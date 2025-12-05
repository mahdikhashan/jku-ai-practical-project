import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl

# =============================================================================
#  TRITON KERNEL (Fixed correctness)
# =============================================================================

@triton.autotune(
    configs=[
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 64, 'GROUP_SIZE_M': 8}, num_warps=4, num_stages=3),
        triton.Config({'BLOCK_M': 64,  'BLOCK_N': 64, 'GROUP_SIZE_M': 8}, num_warps=4, num_stages=3),
        triton.Config({'BLOCK_M': 32,  'BLOCK_N': 64, 'GROUP_SIZE_M': 8}, num_warps=4, num_stages=2),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 64, 'GROUP_SIZE_M': 8}, num_warps=8, num_stages=3),
    ],
    key=['N_CTX', 'HEAD_DIM']
)
@triton.jit
def _swa_fwd_kernel(
    Q, K, V, Out,
    stride_qm, stride_qk, stride_kn, stride_kk, stride_vn, stride_vk, stride_om, stride_ok,
    Z, H, N_CTX,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr,
    HEAD_DIM: tl.constexpr,
    WINDOW_BWD: tl.constexpr, WINDOW_FWD: tl.constexpr,
    GROUP_SIZE_M: tl.constexpr
):
    pid = tl.program_id(0)
    num_pid_m = tl.cdiv(N_CTX, BLOCK_M)
    num_pid_in_group = GROUP_SIZE_M * num_pid_m
    group_id = pid // num_pid_in_group
    first_pid_m = group_id * GROUP_SIZE_M
    group_size_m = min(num_pid_m - first_pid_m, GROUP_SIZE_M)
    pid_m = first_pid_m + (pid % group_size_m)
    
    pid_z = tl.program_id(1)
    pid_h = tl.program_id(2)

    start_m = pid_m * BLOCK_M
    off_m = start_m + tl.arange(0, BLOCK_M)
    off_k = tl.arange(0, HEAD_DIM)

    Q_ptr = Q + (pid_z * H * N_CTX * stride_qm) + (pid_h * N_CTX * stride_qm) + \
            (off_m[:, None] * stride_qm + off_k[None, :] * stride_qk)
    K_base = K + (pid_z * H * N_CTX * stride_kn) + (pid_h * N_CTX * stride_kn)
    V_base = V + (pid_z * H * N_CTX * stride_vn) + (pid_h * N_CTX * stride_vn)
    Out_ptr = Out + (pid_z * H * N_CTX * stride_om) + (pid_h * N_CTX * stride_om) + \
              (off_m[:, None] * stride_om + off_k[None, :] * stride_ok)

    # Load query
    q = tl.load(Q_ptr, mask=off_m[:, None] < N_CTX, other=0.0)

    # Initialize accumulators
    acc = tl.zeros([BLOCK_M, HEAD_DIM], dtype=tl.float32)
    m_i = tl.zeros([BLOCK_M], dtype=tl.float32) - float("inf")
    l_i = tl.zeros([BLOCK_M], dtype=tl.float32)

    # Compute block range for this query block
    # Only process blocks that can possibly be in the window
    for start_n in range(0, N_CTX, BLOCK_N):
        cols = start_n + tl.arange(0, BLOCK_N)
        
        # Load key
        k = tl.load(K_base + (cols[:, None] * stride_kn + off_k[None, :] * stride_kk), 
                    mask=cols[:, None] < N_CTX, other=0.0)
        
        # Compute QK^T
        qk = tl.dot(q, tl.trans(k))
        
        # Apply window mask
        # For each query position i and key position j:
        # Valid if: i - WINDOW_BWD <= j <= i + WINDOW_FWD
        row_idx = off_m[:, None]
        col_idx = cols[None, :]
        
        # Window condition: j >= i - WINDOW_BWD AND j <= i + WINDOW_FWD
        valid_lower = col_idx >= (row_idx - WINDOW_BWD)
        valid_upper = col_idx <= (row_idx + WINDOW_FWD)
        window_mask = valid_lower & valid_upper
        
        # Also mask out invalid positions (beyond sequence length)
        valid_positions = (row_idx < N_CTX) & (col_idx < N_CTX)
        window_mask = window_mask & valid_positions
        
        qk = tl.where(window_mask, qk, float("-inf"))
        
        # Online softmax computation
        # Compute new max
        m_ij = tl.max(qk, 1)
        m_i_new = tl.maximum(m_i, m_ij)
        
        # Rescale previous accumulator
        alpha = tl.exp(m_i - m_i_new)
        
        # Compute attention weights for current block
        p = tl.exp(qk - m_i_new[:, None])
        
        # Load values
        v = tl.load(V_base + (cols[:, None] * stride_vn + off_k[None, :] * stride_vk),
                    mask=cols[:, None] < N_CTX, other=0.0)
        
        # Update accumulator
        acc = acc * alpha[:, None]
        acc = acc + tl.dot(p.to(tl.float16), v)
        
        # Update denominator (sum of exp)
        l_i = l_i * alpha + tl.sum(p, 1)
        
        # Update max
        m_i = m_i_new

    # Final normalization
    acc = acc / l_i[:, None]
    
    # Store output
    tl.store(Out_ptr, acc.to(tl.float16), mask=off_m[:, None] < N_CTX)

# =============================================================================
#  WRAPPER & MODULE
# =============================================================================

def sliding_window_attention_fwd(q, k, v, window_sizes):
    bwd, fwd = window_sizes
    BATCH, HEADS, SEQ, DIM = q.shape
    assert DIM in {16, 32, 64, 128}, f"HEAD_DIM must be 16, 32, 64, or 128, got {DIM}"
    q, k, v = q.contiguous(), k.contiguous(), v.contiguous()
    o = torch.empty_like(q)
    grid = lambda META: (triton.cdiv(SEQ, META['BLOCK_M']), BATCH, HEADS)
    _swa_fwd_kernel[grid](
        q, k, v, o,
        q.stride(2), q.stride(3), k.stride(2), k.stride(3), v.stride(2), v.stride(3), o.stride(2), o.stride(3),
        BATCH, HEADS, SEQ, HEAD_DIM=DIM, WINDOW_BWD=bwd, WINDOW_FWD=fwd
    )
    return o

class SlidingWindowAttention(nn.Module):
    def __init__(self, window_size: tuple[int, int]):
        super().__init__()
        self.window_size = window_size
    
    def forward(self, q, k, v):
        return sliding_window_attention_fwd(q, k, v, self.window_size)

# =============================================================================
#  BENCHMARK
# =============================================================================
if __name__ == "__main__":
    print(f"Device: {torch.cuda.get_device_name(0)}")
    
    # --- Correctness Check ---
    print("\n--- 1. Verifying Correctness ---")
    torch.manual_seed(42)
    B, H, N, D = 1, 4, 1024, 64
    window = (256, 128)
    dtype = torch.float16
    q = torch.randn(B, H, N, D, device='cuda', dtype=dtype)
    k = torch.randn(B, H, N, D, device='cuda', dtype=dtype)
    v = torch.randn(B, H, N, D, device='cuda', dtype=dtype)

    out_triton = sliding_window_attention_fwd(q, k, v, window)
    
    # Reference implementation
    row_idx = torch.arange(N, device='cuda').unsqueeze(-1)
    col_idx = torch.arange(N, device='cuda').unsqueeze(-2)
    # Window mask: i - WINDOW_BWD <= j <= i + WINDOW_FWD
    mask = (col_idx >= row_idx - window[0]) & (col_idx <= row_idx + window[1])
    out_ref = F.scaled_dot_product_attention(q, k, v, attn_mask=mask)

    diff = (out_triton - out_ref).abs().max()
    mean_diff = (out_triton - out_ref).abs().mean()
    print(f"Max Diff: {diff.item():.6f}")
    print(f"Mean Diff: {mean_diff.item():.6f}")
    print(f"Has NaN (Triton): {torch.isnan(out_triton).any()}")
    print(f"Has NaN (Ref): {torch.isnan(out_ref).any()}")
    
    if diff < 0.1:  # FP16 tolerance
        print("✅ Correctness Verified!")
    else:
        print("❌ Mismatch Detected!")
        # Print sample values for debugging
        print(f"\nSample Triton output: {out_triton[0, 0, :5, :5]}")
        print(f"Sample Ref output: {out_ref[0, 0, :5, :5]}")

    # --- Benchmark ---
    print("\n--- 2. Benchmarking (Sequence Length = 16k) ---")
    B, H, N, D = 16, 32, 16384, 16
    window = (256, 256)
    q = torch.randn(B, H, N, D, device='cuda', dtype=dtype)
    k = torch.randn(B, H, N, D, device='cuda', dtype=dtype)
    v = torch.randn(B, H, N, D, device='cuda', dtype=dtype)
    
    print("Compiling & Autotuning...")
    for _ in range(10): 
        sliding_window_attention_fwd(q, k, v, window)
    torch.cuda.synchronize()
    
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    for _ in range(100): 
        sliding_window_attention_fwd(q, k, v, window)
    end.record()
    torch.cuda.synchronize()
    print(f"Avg Latency: {start.elapsed_time(end)/100:.2f} ms")

    