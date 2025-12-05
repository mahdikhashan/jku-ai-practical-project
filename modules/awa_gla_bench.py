import torch
import triton
import triton.language as tl
import sys

# Try importing FLA (Gated Linear Attention)
try:
    from fla.layers import GatedLinearAttention
    FLA_AVAILABLE = True
except ImportError:
    print("Error: 'fla' library not found. Please install it to run the GLA benchmark.")
    print("pip install git+https://github.com/sustcsonglin/flash-linear-attention")
    sys.exit(1)

# ============================================================================
# 1. ANCHOR WINDOW ATTENTION (AWA) KERNELS
# ============================================================================

@triton.jit
def anchor_window_fwd_kernel(
    Q, K, V, Out,
    seq_len, d_head,
    local_window, num_meta_tokens,
    stride_qb, stride_qh, stride_qs, stride_qd,
    stride_kb, stride_kh, stride_ks, stride_kd,
    stride_vb, stride_vh, stride_vs, stride_vd,
    stride_ob, stride_oh, stride_os, stride_od,
    BLOCK_M: tl.constexpr,
    BLOCK_DMODEL: tl.constexpr,
):
    pid_seq = tl.program_id(0)
    pid_batch = tl.program_id(1)
    pid_head = tl.program_id(2)
    
    if pid_seq >= seq_len: return
    
    q_offset = pid_batch * stride_qb + pid_head * stride_qh + pid_seq * stride_qs
    d_offsets = tl.arange(0, BLOCK_DMODEL)
    d_mask = d_offsets < d_head
    
    q_ptrs = Q + q_offset + d_offsets * stride_qd
    q = tl.load(q_ptrs, mask=d_mask, other=0.0).to(tl.float32)
    
    m_i = float("-inf")
    l_i = 0.0
    acc = tl.zeros([BLOCK_DMODEL], dtype=tl.float32)
    
    # Phase 1: Local Window
    local_start = tl.maximum(0, pid_seq - local_window)
    local_end = tl.minimum(seq_len, pid_seq + local_window + 1)
    
    k_block_start = local_start
    while k_block_start < local_end:
        k_offsets = k_block_start + tl.arange(0, BLOCK_M)
        k_valid = k_offsets < local_end
        
        k_base = pid_batch * stride_kb + pid_head * stride_kh
        k_ptrs = K + k_base + k_offsets[:, None] * stride_ks + d_offsets[None, :] * stride_kd
        k = tl.load(k_ptrs, mask=k_valid[:, None] & d_mask[None, :], other=0.0).to(tl.float32)
        
        qk = tl.sum(q[None, :] * k, axis=1)
        qk = tl.where(k_valid, qk, float("-inf"))
        
        m_ij = tl.max(qk, axis=0)
        m_ij = tl.maximum(m_i, m_ij)
        alpha = tl.exp(m_i - m_ij)
        p = tl.exp(qk - m_ij)
        l_ij = tl.sum(p, axis=0)
        
        v_base = pid_batch * stride_vb + pid_head * stride_vh
        v_ptrs = V + v_base + k_offsets[:, None] * stride_vs + d_offsets[None, :] * stride_vd
        v = tl.load(v_ptrs, mask=k_valid[:, None] & d_mask[None, :], other=0.0).to(tl.float32)
        
        acc = acc * alpha + tl.sum(p[:, None] * v, axis=0)
        l_i = l_i * alpha + l_ij
        m_i = m_ij
        k_block_start += BLOCK_M
    
    # Phase 2: Meta Tokens
    if num_meta_tokens > 0:
        meta_end = tl.minimum(num_meta_tokens, seq_len)
        meta_idx = 0
        while meta_idx < meta_end:
            k_offsets = meta_idx + tl.arange(0, BLOCK_M)
            k_valid = (k_offsets < meta_end) & ((k_offsets < local_start) | (k_offsets >= local_end))
            
            k_base = pid_batch * stride_kb + pid_head * stride_kh
            k_ptrs = K + k_base + k_offsets[:, None] * stride_ks + d_offsets[None, :] * stride_kd
            k = tl.load(k_ptrs, mask=k_valid[:, None] & d_mask[None, :], other=0.0).to(tl.float32)
            
            qk = tl.sum(q[None, :] * k, axis=1)
            qk = tl.where(k_valid, qk, float("-inf"))
            
            has_valid = tl.max(qk) > float("-inf")
            if has_valid:
                m_ij = tl.max(qk, axis=0)
                m_ij = tl.maximum(m_i, m_ij)
                alpha = tl.exp(m_i - m_ij)
                p = tl.exp(qk - m_ij)
                l_ij = tl.sum(p, axis=0)
                
                v_base = pid_batch * stride_vb + pid_head * stride_vh
                v_ptrs = V + v_base + k_offsets[:, None] * stride_vs + d_offsets[None, :] * stride_vd
                v = tl.load(v_ptrs, mask=k_valid[:, None] & d_mask[None, :], other=0.0).to(tl.float32)
                
                acc = acc * alpha + tl.sum(p[:, None] * v, axis=0)
                l_i = l_i * alpha + l_ij
                m_i = m_ij
            meta_idx += BLOCK_M
    
    acc = acc / l_i
    o_offset = pid_batch * stride_ob + pid_head * stride_oh + pid_seq * stride_os
    o_ptrs = Out + o_offset + d_offsets * stride_od
    tl.store(o_ptrs, acc, mask=d_mask)

def run_awa(q, k, v, local_window=64, num_meta_tokens=4):
    """AWA Wrapper."""
    batch, n_heads, seq_len, d_head = q.shape
    q, k, v = q.contiguous(), k.contiguous(), v.contiguous()
    out = torch.empty_like(q)
    
    if d_head <= 32: BLOCK_M = 64
    elif d_head <= 64: BLOCK_M = 16
    else: BLOCK_M = 16
    BLOCK_DMODEL = triton.next_power_of_2(d_head)
    
    grid = (seq_len, batch, n_heads)
    anchor_window_fwd_kernel[grid](
        q, k, v, out,
        seq_len, d_head,
        local_window, num_meta_tokens,
        q.stride(0), q.stride(1), q.stride(2), q.stride(3),
        k.stride(0), k.stride(1), k.stride(2), k.stride(3),
        v.stride(0), v.stride(1), v.stride(2), v.stride(3),
        out.stride(0), out.stride(1), out.stride(2), out.stride(3),
        BLOCK_M=BLOCK_M, BLOCK_DMODEL=BLOCK_DMODEL,
    )
    return out

# ============================================================================
# 2. BENCHMARKING UTILS
# ============================================================================

def benchmark_latency(name, func, args, n_warmup=10, n_iters=100):
    """
    Unified benchmarking function using torch.cuda.Event for high precision.
    """
    # Warmup
    for _ in range(n_warmup):
        func(*args)
    torch.cuda.synchronize()

    # Measurement
    start_event = torch.cuda.Event(enable_timing=True)
    end_event = torch.cuda.Event(enable_timing=True)

    start_event.record()
    for _ in range(n_iters):
        func(*args)
    end_event.record()
    torch.cuda.synchronize()

    total_ms = start_event.elapsed_time(end_event)
    avg_ms = total_ms / n_iters
    return avg_ms

# ============================================================================
# 3. MAIN EXECUTION
# ============================================================================

if __name__ == "__main__":
    print(f"Device: {torch.cuda.get_device_name(0)}")
    
    # --- Configuration ---
    BATCH_SIZE = 16
    SEQ_LEN = 2048
    HIDDEN_SIZE = 512
    NUM_HEADS = 32
    D_HEAD = HIDDEN_SIZE // NUM_HEADS  # 512 / 32 = 16
    DTYPE = torch.float16
    DEVICE = "cuda:0"
    
    # AWA Params
    WINDOW_SIZE = 64
    META_TOKENS = 4
    
    print("-" * 60)
    print(f"Config: B={BATCH_SIZE}, S={SEQ_LEN}, H_dim={HIDDEN_SIZE}, Heads={NUM_HEADS}")
    print(f"Derived: Head_Dim={D_HEAD}")
    print("-" * 60)

    # --- Setup GLA ---
    # Shape: (Batch, Seq, Hidden) -> (16, 2048, 512)
    gla_layer = GatedLinearAttention(
        hidden_size=HIDDEN_SIZE,
        num_heads=NUM_HEADS,
        mode='fused_recurrent'
    ).to(device=DEVICE, dtype=DTYPE)
    
    # Input for GLA
    x_gla = torch.randn(BATCH_SIZE, SEQ_LEN, HIDDEN_SIZE, device=DEVICE, dtype=DTYPE)

    # --- Setup AWA ---
    # Shape: (Batch, Heads, Seq, Head_Dim) -> (16, 32, 2048, 16)
    # We project x_gla into q, k, v to simulate a real attention layer
    # For benchmarking simple kernel speed, random tensors are fine.
    q_awa = torch.randn(BATCH_SIZE, NUM_HEADS, SEQ_LEN, D_HEAD, device=DEVICE, dtype=DTYPE)
    k_awa = torch.randn(BATCH_SIZE, NUM_HEADS, SEQ_LEN, D_HEAD, device=DEVICE, dtype=DTYPE)
    v_awa = torch.randn(BATCH_SIZE, NUM_HEADS, SEQ_LEN, D_HEAD, device=DEVICE, dtype=DTYPE)

    # --- RUN BENCHMARKS ---
    
    print("Running GLA Benchmark...")
    # GLA takes (x), returns (y)
    gla_time = benchmark_latency("GLA", gla_layer, (x_gla,))
    
    print("Running AWA Benchmark...")
    # AWA takes (q, k, v), returns (out)
    awa_time = benchmark_latency("AWA", run_awa, (q_awa, k_awa, v_awa, WINDOW_SIZE, META_TOKENS))

    # --- DISPLAY RESULTS ---
    
    print("\n" + "=" * 60)
    print(f"{'KERNEL / METHOD':<30} | {'TIME (ms)':<12} | {'SPEEDUP (vs AWA)':<15}")
    print("=" * 60)
    
    # AWA Stats
    print(f"{'Anchor Window Attn (yours)':<30} | {awa_time:<12.4f} | {'1.00x':<15}")
    
    # GLA Stats
    gla_speedup = awa_time / gla_time
    print(f"{'GLA (fused_recurrent)':<30} | {gla_time:<12.4f} | {f'{gla_speedup:.2f}x':<15}")
    print("=" * 60)
    
    # --- MEMORY CHECK ---
    print(f"\nPeak Memory Allocated: {torch.cuda.max_memory_allocated() / 1024**2:.2f} MB")
