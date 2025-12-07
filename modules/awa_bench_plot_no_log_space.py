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
    print("Error: 'fla' library not found. Please install it to benchmark GLA.")
    FLA_AVAILABLE = False

# ============================================================================
# KERNEL (Optimized for Large D)
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

def benchmark_latency(func, args, n_iters=20):
    # Warmup
    for _ in range(3): func(*args)
    torch.cuda.synchronize()
    
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    
    start.record()
    for _ in range(n_iters): func(*args)
    end.record()
    torch.cuda.synchronize()
    
    return start.elapsed_time(end) / n_iters

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
    
    results = {
        'seq_len': [],
        'gla_latency': [],
        'awa_latency': [],
        'gla_throughput': [],
        'awa_throughput': []
    }

    print(f"{'Seq Len':<10} | {'GLA Time (ms)':<15} | {'AWA Time (ms)':<15} | {'Notes':<20}")
    print("-" * 70)

    for seq_len in SEQ_LENS:
        results['seq_len'].append(seq_len)
        current_gla_time = None
        current_awa_time = None
        
        # 1. Benchmark GLA
        if FLA_AVAILABLE:
            try:
                torch.cuda.empty_cache()
                gla_layer = GatedLinearAttention(
                    hidden_size=HIDDEN_SIZE,
                    num_heads=NUM_HEADS,
                    mode='fused_recurrent'
                ).to(device=DEVICE, dtype=DTYPE)
                x_gla = torch.randn(BATCH_SIZE, seq_len, HIDDEN_SIZE, device=DEVICE, dtype=DTYPE)
                
                current_gla_time = benchmark_latency(gla_layer, (x_gla,))
                del gla_layer, x_gla # cleanup
            except torch.cuda.OutOfMemoryError:
                current_gla_time = float('nan')
            except Exception as e:
                print(f"\nGLA Error at {seq_len}: {e}")
                current_gla_time = float('nan')
        else:
            current_gla_time = float('nan')

        # 2. Benchmark AWA
        try:
            torch.cuda.empty_cache()
            q = torch.randn(BATCH_SIZE, NUM_HEADS, seq_len, D_HEAD, device=DEVICE, dtype=DTYPE)
            k = torch.randn(BATCH_SIZE, NUM_HEADS, seq_len, D_HEAD, device=DEVICE, dtype=DTYPE)
            v = torch.randn(BATCH_SIZE, NUM_HEADS, seq_len, D_HEAD, device=DEVICE, dtype=DTYPE)
            
            current_awa_time = benchmark_latency(run_awa, (q, k, v, WINDOW_SIZE, META_TOKENS))
            del q, k, v # cleanup
        except torch.cuda.OutOfMemoryError:
            current_awa_time = float('nan')
        except Exception as e:
            print(f"\nAWA Error at {seq_len}: {e}")
            current_awa_time = float('nan')

        # Store and Print
        results['gla_latency'].append(current_gla_time)
        results['awa_latency'].append(current_awa_time)
        
        # Calculate Throughput (Tokens / Sec)
        tokens = BATCH_SIZE * seq_len
        
        if current_gla_time and not torch.isnan(torch.tensor(current_gla_time)):
            gla_tput = tokens / (current_gla_time / 1000.0)
            results['gla_throughput'].append(gla_tput)
            gla_str = f"{current_gla_time:.2f}"
        else:
            results['gla_throughput'].append(None)
            gla_str = "OOM/Err"

        if current_awa_time and not torch.isnan(torch.tensor(current_awa_time)):
            awa_tput = tokens / (current_awa_time / 1000.0)
            results['awa_throughput'].append(awa_tput)
            awa_str = f"{current_awa_time:.2f}"
        else:
            results['awa_throughput'].append(None)
            awa_str = "OOM/Err"

        print(f"{seq_len:<10} | {gla_str:<15} | {awa_str:<15} |")

    # ============================================================================
    # PLOTTING (LINEAR SCALES)
    # ============================================================================
    
    valid_seq = results['seq_len']
    
    def get_valid_points(key):
        x = []
        y = []
        for s, val in zip(valid_seq, results[key]):
            if val is not None and not torch.isnan(torch.tensor(val)):
                x.append(s)
                y.append(val)
        return x, y

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
    
    # 1. Latency Plot (Linear Scale)
    x_gla, y_gla = get_valid_points('gla_latency')
    x_awa, y_awa = get_valid_points('awa_latency')
    
    ax1.plot(x_gla, y_gla, 'o-', linewidth=2, label='GLA (Recurrent)')
    ax1.plot(x_awa, y_awa, 's-', linewidth=2, label='AWA (Sliding Win)')
    ax1.set_xlabel('Sequence Length')
    ax1.set_ylabel('Latency (ms)')
    ax1.set_title(f'Latency vs Sequence Length (Linear)\n(Batch={BATCH_SIZE}, D={D_HEAD})')
    # Force linear (default) but add grid for readability
    ax1.grid(True, linestyle='--', alpha=0.7)
    ax1.legend()
    
    # 2. Throughput Plot (Linear Scale)
    x_gla_tp, y_gla_tp = get_valid_points('gla_throughput')
    x_awa_tp, y_awa_tp = get_valid_points('awa_throughput')
    
    ax2.plot(x_gla_tp, y_gla_tp, 'o-', linewidth=2, label='GLA (Recurrent)')
    ax2.plot(x_awa_tp, y_awa_tp, 's-', linewidth=2, label='AWA (Sliding Win)')
    ax2.set_xlabel('Sequence Length')
    ax2.set_ylabel('Throughput (tokens/sec)')
    ax2.set_title('Throughput vs Sequence Length (Linear)')
    ax2.grid(True, linestyle='--', alpha=0.7)
    ax2.legend()
    
    plt.tight_layout()
    plt.savefig('benchmark_results_linear.png')
    print("\nChart saved to 'benchmark_results_linear.png'")
