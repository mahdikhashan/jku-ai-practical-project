import torch
import triton
import triton.language as tl

# Helper for ceil division
def cdiv(x, y):
    return (x + y - 1) // y

@triton.jit
def _swa_fwd_kernel(
    Q, K, V, Out,
    stride_qm, stride_qk,  # Q strides
    stride_kn, stride_kk,  # K strides
    stride_vn, stride_vk,  # V strides
    stride_om, stride_on,  # Output strides
    Z, H, N_CTX,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr,
    HEAD_DIM: tl.constexpr,
    WINDOW_BWD: tl.constexpr, WINDOW_FWD: tl.constexpr
):
    # Grid: (Z * H * (N_CTX // BLOCK_M))
    # PID mapping
    pid_m = tl.program_id(0)
    pid_z = tl.program_id(1)
    pid_h = tl.program_id(2)

    # 1. Setup Pointers for Q
    # We want to load a block of Q: [BLOCK_M, HEAD_DIM]
    # Offset of the current block in Q sequence
    start_m = pid_m * BLOCK_M
    off_m = start_m + tl.arange(0, BLOCK_M)
    off_k = tl.arange(0, HEAD_DIM)
    
    # Q_ptr = Base + (batch * stride_z) + (head * stride_h) + (row * stride_m) + (col * stride_k)
    # We assume standard (B, H, S, D) layout for calculations below, but use strides passed in
    Q_base = Q + (pid_z * H * N_CTX * stride_qm) + (pid_h * N_CTX * stride_qm) # Correction: logic simplified for batch/head handling outside
    # Let's rely on the caller passing correct pointers or simplified strides.
    # To be safe/standard, we usually assume inputs are (Batch, Heads, Seq, Dim).
    
    # Advanced pointer arithmetic:
    # Q is (B, H, S, D). Stride_qm handles S. stride_qk handles D.
    # We advance to the specific batch/head first.
    Q_ptr = Q + (pid_z * H * N_CTX * HEAD_DIM) + (pid_h * N_CTX * HEAD_DIM) \
              + (off_m[:, None] * stride_qm + off_k[None, :] * stride_qk)
    
    K_base = K + (pid_z * H * N_CTX * HEAD_DIM) + (pid_h * N_CTX * HEAD_DIM)
    V_base = V + (pid_z * H * N_CTX * HEAD_DIM) + (pid_h * N_CTX * HEAD_DIM)
    Out_ptr = Out + (pid_z * H * N_CTX * HEAD_DIM) + (pid_h * N_CTX * HEAD_DIM) \
                  + (off_m[:, None] * stride_om + off_k[None, :] * stride_on)

    # Load Q
    # Mask checks if we are out of bounds for sequence length
    q = tl.load(Q_ptr, mask=off_m[:, None] < N_CTX, other=0.0)

    # Initialize Accumulators
    acc = tl.zeros([BLOCK_M, HEAD_DIM], dtype=tl.float32)
    l_i = tl.zeros([BLOCK_M], dtype=tl.float32) - float("inf") # Log-sum-exp
    m_i = tl.zeros([BLOCK_M], dtype=tl.float32) - float("inf") # Max logic

    # --- Sliding Window Logic ---
    # Determine the range of K blocks to load based on the current Q block
    
    # Range of Q indices in this block: [start_m, start_m + BLOCK_M]
    # We care about K indices in range: [start_m - WINDOW_BWD, start_m + BLOCK_M + WINDOW_FWD]
    
    min_k = start_m - WINDOW_BWD
    max_k = start_m + BLOCK_M + WINDOW_FWD
    
    # Align min_k and max_k to BLOCK_N boundaries
    start_n = (min_k // BLOCK_N) * BLOCK_N
    if start_n < 0: start_n = 0
    end_n = ((max_k + BLOCK_N - 1) // BLOCK_N) * BLOCK_N
    if end_n > N_CTX: end_n = N_CTX

    # Loop over relevant K blocks
    for start_n_idx in range(start_n, end_n, BLOCK_N):
        cols = start_n_idx + tl.arange(0, BLOCK_N)
        
        # Load K. Shape required for dot: (HEAD_DIM, BLOCK_N) usually, or (BLOCK_N, HEAD_DIM)
        # If we do dot(q, k.T), we load K as (BLOCK_N, HEAD_DIM).
        # K pointers: 
        k_ptrs = K_base + (cols[:, None] * stride_kn + off_k[None, :] * stride_kk)
        k = tl.load(k_ptrs, mask=cols[:, None] < N_CTX, other=0.0)
        
        # Load V. Shape (BLOCK_N, HEAD_DIM)
        v_ptrs = V_base + (cols[:, None] * stride_vn + off_k[None, :] * stride_vk)
        v = tl.load(v_ptrs, mask=cols[:, None] < N_CTX, other=0.0)

        # Compute QK^T. q is (M, D), k is (N, D).
        # We need k transposed implicitly or explicitly.
        # tl.dot(q, k.T) -> (M, N)
        qk = tl.dot(q, tl.trans(k))
        
        # --- Masking ---
        # row_idx is (BLOCK_M, 1), col_idx is (1, BLOCK_N)
        row_idx = off_m[:, None]
        col_idx = cols[None, :]
        
        # Window Condition: row - bwd <= col <= row + fwd
        mask = (row_idx >= (col_idx - WINDOW_FWD)) & (row_idx <= (col_idx + WINDOW_BWD))
        
        # Apply Mask
        qk = tl.where(mask, qk, float("-inf"))

        # --- Online Softmax (Flash Attention Logic) ---
        m_i_new = tl.maximum(m_i, tl.max(qk, 1))
        alpha = tl.exp(m_i - m_i_new)
        p = tl.exp(qk - m_i_new[:, None])
        
        # Update accumulator
        # acc * alpha + p * v
        # p is (M, N), v is (N, D) -> (M, D)
        acc = acc * alpha[:, None] + tl.dot(p.to(tl.float16), v)
        
        # Update LogSumExp
        # l_i = m_new + log( sum(p) + exp(l_old - m_new) )
        l_i = m_i_new + tl.log(tl.sum(p, 1) + tl.exp(l_i - m_i_new))
        m_i = m_i_new

    # Finalize Output
    # acc / sum(exp(qk))
    # We need to be careful with division by zero if masked out entirely, but usually safe in attention
    acc = acc / tl.exp(l_i - m_i)[:, None]
    tl.store(Out_ptr, acc.to(tl.float16), mask=off_m[:, None] < N_CTX)

def swa_triton(q, k, v, window_sizes):
    bwd, fwd = window_sizes
    BATCH, HEADS, SEQ, DIM = q.shape
    
    # Tuning: BLOCK_M=64/128 usually good. Titan V has decent SRAM.
    BLOCK_M = 64
    BLOCK_N = 64
    
    # Output buffer
    o = torch.empty_like(q)
    
    grid = (cdiv(SEQ, BLOCK_M), BATCH, HEADS)
    
    _swa_fwd_kernel[grid](
        q, k, v, o,
        q.stride(2), q.stride(3),
        k.stride(2), k.stride(3),
        v.stride(2), v.stride(3),
        o.stride(2), o.stride(3),
        BATCH, HEADS, SEQ,
        BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N,
        HEAD_DIM=DIM,
        WINDOW_BWD=bwd, WINDOW_FWD=fwd
    )
    return o

# --- Benchmark ---
if __name__ == "__main__":
    from torch.profiler import profile, record_function, ProfilerActivity
    
    print(f"Device: {torch.cuda.get_device_name(0)}")
    
    # SETTINGS FOR THE BENCHMARK
    batch_size = 16
    seq_len = 32768
    num_heads = 32
    head_dim = 16
    window_sizes = (256, 256)
    device = "cuda:0"
    dtype = torch.float16

    print(f"Benchmarking SWA Triton | N={seq_len} | Window={window_sizes}")

    q = torch.randn(batch_size, num_heads, seq_len, head_dim, device=device, dtype=dtype)
    k = torch.randn(batch_size, num_heads, seq_len, head_dim, device=device, dtype=dtype)
    v = torch.randn(batch_size, num_heads, seq_len, head_dim, device=device, dtype=dtype)

    # 1. Warmup
    print("Warming up...")
    for _ in range(10):
        swa_triton(q, k, v, window_sizes)
    torch.cuda.synchronize()
    print("Warmup done.")

    # 2. Timing
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    
    start.record()
    for _ in range(100):
        swa_triton(q, k, v, window_sizes)
    end.record()
    torch.cuda.synchronize()
    
    latency = start.elapsed_time(end) / 100
    print(f"Avg Latency: {latency:.3f} ms")
    print(f"Throughput:  {1000/latency:.2f} runs/sec")

    # 3. Profile to verify fused kernel count
    with profile(activities=[ProfilerActivity.CUDA], record_shapes=True) as prof:
        for _ in range(10):
            swa_triton(q, k, v, window_sizes)
        torch.cuda.synchronize()

    print(prof.key_averages().table(sort_by="cuda_time_total", row_limit=10))
