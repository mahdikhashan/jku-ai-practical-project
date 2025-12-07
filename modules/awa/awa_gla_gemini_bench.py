import torch
import triton
import triton.language as tl
import sys

# Try importing FLA
try:
    from fla.layers import GatedLinearAttention
    FLA_AVAILABLE = True
except ImportError:
    print("Error: 'fla' library not found.")
    sys.exit(1)

# ============================================================================
# KERNEL (Optimized block sizes for Large D)
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
    
    acc = acc / l_i
    o_offset = pid_batch * stride_ob + pid_head * stride_oh + pid_seq * stride_os
    o_ptrs = Out + o_offset + d_offsets * stride_od
    tl.store(o_ptrs, acc, mask=d_mask)

def run_awa(q, k, v, local_window, num_meta_tokens):
    batch, n_heads, seq_len, d_head = q.shape
    q, k, v = q.contiguous(), k.contiguous(), v.contiguous()
    out = torch.empty_like(q)
    
    # Tuned for Large D
    BLOCK_M = 16 # Keep small to reduce register pressure
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

def benchmark_latency(func, args, n_iters=50):
    # Warmup
    for _ in range(5): func(*args)
    torch.cuda.synchronize()
    
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    
    start.record()
    for _ in range(n_iters): func(*args)
    end.record()
    torch.cuda.synchronize()
    
    return start.elapsed_time(end) / n_iters

if __name__ == "__main__":
    print(f"Device: {torch.cuda.get_device_name(0)}")
    
    # --- The "AWA Win Condition" Config ---
    # We use a large head dimension (128) 
    # and a window size smaller than D (64)
    
    BATCH_SIZE = 4
    SEQ_LEN = 4096 
    NUM_HEADS = 8
    D_HEAD = 128           # <--- CRITICAL: Large D hurts GLA
    HIDDEN_SIZE = NUM_HEADS * D_HEAD
    DTYPE = torch.float16
    DEVICE = "cuda:0"
    
    WINDOW_SIZE = 64       # <--- CRITICAL: Window < D_Head
    META_TOKENS = 0
    
    print("-" * 60)
    print(f"Config for 'AWA Win Condition':")
    print(f"Head_Dim={D_HEAD} | Window={WINDOW_SIZE}")
    print(f"GLA Complexity ~ D^2 ({D_HEAD**2})")
    print(f"AWA Complexity ~ D*W ({D_HEAD*WINDOW_SIZE})")
    print("-" * 60)

    # Setup GLA
    gla_layer = GatedLinearAttention(
        hidden_size=HIDDEN_SIZE,
        num_heads=NUM_HEADS,
        mode='fused_recurrent' 
    ).to(device=DEVICE, dtype=DTYPE)
    x_gla = torch.randn(BATCH_SIZE, SEQ_LEN, HIDDEN_SIZE, device=DEVICE, dtype=DTYPE)

    # Setup AWA
    q = torch.randn(BATCH_SIZE, NUM_HEADS, SEQ_LEN, D_HEAD, device=DEVICE, dtype=DTYPE)
    k = torch.randn(BATCH_SIZE, NUM_HEADS, SEQ_LEN, D_HEAD, device=DEVICE, dtype=DTYPE)
    v = torch.randn(BATCH_SIZE, NUM_HEADS, SEQ_LEN, D_HEAD, device=DEVICE, dtype=DTYPE)

    gla_time = benchmark_latency(gla_layer, (x_gla,))
    awa_time = benchmark_latency(run_awa, (q, k, v, WINDOW_SIZE, META_TOKENS))

    print(f"{'Method':<30} | {'Time (ms)':<10}")
    print("-" * 45)
    print(f"{'GLA (Recurrent)':<30} | {gla_time:<10.4f}")
    print(f"{'AWA (Sliding Win)':<30} | {awa_time:<10.4f}")
    
    if awa_time < gla_time:
        print(f"\nResult: AWA is {gla_time/awa_time:.2f}x FASTER (Prediction Correct)")
    else:
        print(f"\nResult: GLA is {awa_time/gla_time:.2f}x FASTER")
