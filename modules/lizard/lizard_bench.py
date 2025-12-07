import torch
import triton
import triton.language as tl
import sys
import matplotlib.pyplot as plt

# Try importing FLA
try:
    from fla.layers import GatedLinearAttention
    FLA_AVAILABLE = True
except ImportError:
    print("Error: 'fla' library not found. GLA and Lizard benchmarks will fail.")
    FLA_AVAILABLE = False

# ============================================================================
# TRITON KERNEL (AWA)
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

# ============================================================================
# RUNNERS
# ============================================================================

def run_awa(q, k, v, local_window, num_meta_tokens):
    batch, n_heads, seq_len, d_head = q.shape
    q, k, v = q.contiguous(), k.contiguous(), v.contiguous()
    out = torch.empty_like(q)
    
    BLOCK_M = 16 
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

def run_lizard(gla_layer, x_gla, q, k, v, local_window, num_meta_tokens):
    """
    Lizard = GLA + AWA
    """
    # 1. Run GLA
    # FLA returns tuple (output, last_state) usually, we take index 0
    gla_out, _ = gla_layer(x_gla)
    
    # 2. Run AWA
    awa_out = run_awa(q, k, v, local_window, num_meta_tokens)
    
    # 3. Sum (Fusion)
    # AWA is [Batch, Heads, Seq, D], GLA is [Batch, Seq, Hidden]
    # We must reshape AWA to match GLA
    B, H, L, D = awa_out.shape
    awa_flat = awa_out.permute(0, 2, 1, 3).reshape(B, L, H*D)
    
    return gla_out + awa_flat

def benchmark_latency(func, args, n_iters=10):
    # Warmup
    for _ in range(2): func(*args)
    torch.cuda.synchronize()
    
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    
    start.record()
    for _ in range(n_iters): func(*args)
    end.record()
    torch.cuda.synchronize()
    
    return start.elapsed_time(end) / n_iters

# ============================================================================
# MAIN BENCHMARK
# ============================================================================

if __name__ == "__main__":
    if not torch.cuda.is_available():
        print("CUDA not detected. This script requires a GPU.")
        sys.exit(1)

    print(f"Device: {torch.cuda.get_device_name(0)}")
    
    # --- Config ---
    BATCH_SIZE = 4
    NUM_HEADS = 8
    D_HEAD = 128
    HIDDEN_SIZE = NUM_HEADS * D_HEAD
    DTYPE = torch.float16
    DEVICE = "cuda:0"
    WINDOW_SIZE = 64
    META_TOKENS = 0
    
    # 1k to 32k
    SEQ_LENS = [1024, 2048, 4096, 8192, 16384, 32768]
    
    data = {
        'seq': [],
        'gla_lat': [], 'awa_lat': [], 'liz_lat': [],
        'gla_tp': [], 'awa_tp': [], 'liz_tp': []
    }

    print(f"{'Seq':<8} | {'GLA (ms)':<10} | {'AWA (ms)':<10} | {'Lizard (ms)':<10}")
    print("-" * 50)

    for seq_len in SEQ_LENS:
        data['seq'].append(seq_len)
        
        # Prepare Data Containers
        t_gla, t_awa, t_liz = None, None, None
        
        # 1. Benchmark GLA
        if FLA_AVAILABLE:
            try:
                torch.cuda.empty_cache()
                gla_layer = GatedLinearAttention(hidden_size=HIDDEN_SIZE, num_heads=NUM_HEADS, mode='fused_recurrent').to(DEVICE, dtype=DTYPE)
                x = torch.randn(BATCH_SIZE, seq_len, HIDDEN_SIZE, device=DEVICE, dtype=DTYPE)
                t_gla = benchmark_latency(lambda a: gla_layer(a), (x,))
                del gla_layer, x
            except Exception: t_gla = None
        
        # 2. Benchmark AWA
        try:
            torch.cuda.empty_cache()
            q = torch.randn(BATCH_SIZE, NUM_HEADS, seq_len, D_HEAD, device=DEVICE, dtype=DTYPE)
            k = torch.randn(BATCH_SIZE, NUM_HEADS, seq_len, D_HEAD, device=DEVICE, dtype=DTYPE)
            v = torch.randn(BATCH_SIZE, NUM_HEADS, seq_len, D_HEAD, device=DEVICE, dtype=DTYPE)
            t_awa = benchmark_latency(run_awa, (q, k, v, WINDOW_SIZE, META_TOKENS))
            # Keep q,k,v for Lizard if possible, else recreate
            del q, k, v
        except Exception: t_awa = None

        # 3. Benchmark Lizard (GLA + AWA)
        if FLA_AVAILABLE:
            try:
                torch.cuda.empty_cache()
                # Re-allocate everything for Lizard (highest memory pressure)
                gla_layer = GatedLinearAttention(hidden_size=HIDDEN_SIZE, num_heads=NUM_HEADS, mode='fused_recurrent').to(DEVICE, dtype=DTYPE)
                x = torch.randn(BATCH_SIZE, seq_len, HIDDEN_SIZE, device=DEVICE, dtype=DTYPE)
                q = torch.randn(BATCH_SIZE, NUM_HEADS, seq_len, D_HEAD, device=DEVICE, dtype=DTYPE)
                k = torch.randn(BATCH_SIZE, NUM_HEADS, seq_len, D_HEAD, device=DEVICE, dtype=DTYPE)
                v = torch.randn(BATCH_SIZE, NUM_HEADS, seq_len, D_HEAD, device=DEVICE, dtype=DTYPE)
                
                t_liz = benchmark_latency(run_lizard, (gla_layer, x, q, k, v, WINDOW_SIZE, META_TOKENS))
                del gla_layer, x, q, k, v
            except Exception: t_liz = None

        # Store Results
        data['gla_lat'].append(t_gla)
        data['awa_lat'].append(t_awa)
        data['liz_lat'].append(t_liz)
        
        tokens = BATCH_SIZE * seq_len
        data['gla_tp'].append(tokens / (t_gla/1000) if t_gla else None)
        data['awa_tp'].append(tokens / (t_awa/1000) if t_awa else None)
        data['liz_tp'].append(tokens / (t_liz/1000) if t_liz else None)

        # Print row
        s_gla = f"{t_gla:.2f}" if t_gla else "OOM"
        s_awa = f"{t_awa:.2f}" if t_awa else "OOM"
        s_liz = f"{t_liz:.2f}" if t_liz else "OOM"
        print(f"{seq_len:<8} | {s_gla:<10} | {s_awa:<10} | {s_liz:<10}")

    # ============================================================================
    # PLOTTING
    # ============================================================================
    
    def get_clean(key):
        x, y = [], []
        for s, v in zip(data['seq'], data[key]):
            if v is not None:
                x.append(s)
                y.append(v)
        return x, y

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

    # Latency
    x, y = get_clean('gla_lat')
    ax1.plot(x, y, 'o-', label='GLA (Recurrent)')
    x, y = get_clean('awa_lat')
    ax1.plot(x, y, 's-', label='AWA (Sliding)')
    x, y = get_clean('liz_lat')
    ax1.plot(x, y, '^-', label='Lizard (Sum)', color='green')
    
    ax1.set_title("Latency (ms) - Linear Scale")
    ax1.set_xlabel("Sequence Length")
    ax1.set_ylabel("Time (ms)")
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # Throughput
    x, y = get_clean('gla_tp')
    ax2.plot(x, y, 'o-', label='GLA')
    x, y = get_clean('awa_tp')
    ax2.plot(x, y, 's-', label='AWA')
    x, y = get_clean('liz_tp')
    ax2.plot(x, y, '^-', label='Lizard', color='green')

    ax2.set_title("Throughput (tok/s) - Linear Scale")
    ax2.set_xlabel("Sequence Length")
    ax2.set_ylabel("Tokens / Sec")
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig('benchmark_lizard.png')
    print("\nSaved chart to 'benchmark_lizard.png'")
