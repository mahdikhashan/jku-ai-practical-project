import torch
import torch.nn as nn
import triton
import triton.language as tl
import matplotlib.pyplot as plt
import sys

# Try importing FLA
try:
    from fla.layers import GatedLinearAttention
    FLA_AVAILABLE = True
except ImportError:
    FLA_AVAILABLE = False

# ============================================================================
# 1. OPTIMIZED AWA KERNEL (Fixed for 16k/32k Scaling)
# ============================================================================

@triton.jit
def anchor_window_fwd_kernel_optimized(
    Q, K, V, Out,
    # --- CRITICAL FIX: Explicit 64-bit Strides ---
    stride_qb: tl.int64, stride_qh: tl.int64, stride_qs: tl.int64, stride_qd: tl.int64,
    stride_kb: tl.int64, stride_kh: tl.int64, stride_ks: tl.int64, stride_kd: tl.int64,
    stride_vb: tl.int64, stride_vh: tl.int64, stride_vs: tl.int64, stride_vd: tl.int64,
    stride_ob: tl.int64, stride_oh: tl.int64, stride_os: tl.int64, stride_od: tl.int64,
    seq_len, d_head, window_size,
    BLOCK_Q: tl.constexpr, BLOCK_K: tl.constexpr, BLOCK_DMODEL: tl.constexpr
):
    # Force 64-bit indexing for large grids
    pid_m = tl.program_id(0).to(tl.int64)
    pid_b = tl.program_id(1).to(tl.int64)
    pid_h = tl.program_id(2).to(tl.int64)

    # Offsets for Q
    off_m = pid_m * BLOCK_Q + tl.arange(0, BLOCK_Q)
    off_d = tl.arange(0, BLOCK_DMODEL)
    mask_m = off_m < seq_len
    
    # Load Q (BF16)
    # Calculation strictly in 64-bit to avoid overflow
    q_ptr = Q + (pid_b * stride_qb + pid_h * stride_qh) + \
            off_m[:, None] * stride_qs + off_d[None, :] * stride_qd
    q = tl.load(q_ptr, mask=mask_m[:, None], other=0.0)

    # Accumulators (FP32)
    acc = tl.zeros([BLOCK_Q, BLOCK_DMODEL], dtype=tl.float32)
    l_i = tl.zeros([BLOCK_Q], dtype=tl.float32)
    m_i = tl.full([BLOCK_Q], float("-inf"), dtype=tl.float32)

    start_idx = pid_m * BLOCK_Q
    min_k_idx = tl.maximum(0, start_idx - window_size)
    start_block_k = min_k_idx // BLOCK_K
    
    # Loop over K/V blocks
    for block_k in range(start_block_k, pid_m + 1):
        off_n = block_k * BLOCK_K + tl.arange(0, BLOCK_K)
        mask_n = off_n < seq_len
        
        # Load K, V (BF16)
        # Note: K loaded as [BLOCK_K, D]
        k_ptr = K + (pid_b * stride_kb + pid_h * stride_kh) + \
                off_n[:, None] * stride_ks + off_d[None, :] * stride_kd
        v_ptr = V + (pid_b * stride_vb + pid_h * stride_vh) + \
                off_n[:, None] * stride_vs + off_d[None, :] * stride_vd
        
        k = tl.load(k_ptr, mask=mask_n[:, None], other=0.0)
        v = tl.load(v_ptr, mask=mask_n[:, None], other=0.0)
        
        # Compute Q @ K.T
        qk = tl.dot(q, tl.trans(k))
        
        # Masking
        diff = off_m[:, None] - off_n[None, :]
        window_mask = (diff >= 0) & (diff <= window_size)
        qk = tl.where(window_mask & mask_m[:, None] & mask_n[None, :], qk, float("-inf"))
        
        # Softmax
        m_curr = tl.max(qk, 1)
        m_new = tl.maximum(m_i, m_curr)
        alpha = tl.exp(m_i - m_new)
        p = tl.exp(qk - m_new[:, None])
        
        # Update Accumulator
        acc = acc * alpha[:, None] + tl.dot(p.to(v.dtype), v)
        l_i = l_i * alpha + tl.sum(p, 1)
        m_i = m_new

    # Finalize
    acc = acc / l_i[:, None]
    
    # Store
    out_ptr = Out + (pid_b * stride_ob + pid_h * stride_oh) + \
              off_m[:, None] * stride_os + off_d[None, :] * stride_od
    tl.store(out_ptr, acc, mask=mask_m[:, None])

# ============================================================================
# 2. GLA KERNEL (Unchanged - Already Robust)
# ============================================================================

@triton.jit
def gla_chunk_fwd_kernel(
    Q, K, V, G, Out, State_in, State_out,
    seq_len, d_model, chunk_size,
    stride_qb: tl.int64, stride_qh: tl.int64, stride_qs: tl.int64, stride_qd: tl.int64,
    stride_kb: tl.int64, stride_kh: tl.int64, stride_ks: tl.int64, stride_kd: tl.int64,
    stride_vb: tl.int64, stride_vh: tl.int64, stride_vs: tl.int64, stride_vd: tl.int64,
    stride_gb: tl.int64, stride_gh: tl.int64, stride_gs: tl.int64,
    stride_ob: tl.int64, stride_oh: tl.int64, stride_os: tl.int64, stride_od: tl.int64,
    stride_sb: tl.int64, stride_sh: tl.int64, stride_sd: tl.int64,
    BLOCK_D: tl.constexpr, BLOCK_CHUNK: tl.constexpr,
):
    pid_batch = tl.program_id(0).to(tl.int64)
    pid_head = tl.program_id(1).to(tl.int64)
    pid_chunk = tl.program_id(2).to(tl.int64)
    
    chunk_start = pid_chunk * BLOCK_CHUNK
    chunk_end = tl.minimum(chunk_start + BLOCK_CHUNK, seq_len)
    if (chunk_end - chunk_start) <= 0: return

    seq_offsets = chunk_start + tl.arange(0, BLOCK_CHUNK)
    d_offsets = tl.arange(0, BLOCK_D)
    seq_mask = seq_offsets < chunk_end
    d_mask = d_offsets < d_model
    
    q_off = pid_batch * stride_qb + pid_head * stride_qh
    k_off = pid_batch * stride_kb + pid_head * stride_kh
    v_off = pid_batch * stride_vb + pid_head * stride_vh
    g_off = pid_batch * stride_gb + pid_head * stride_gh
    
    Q_c = tl.load(Q + q_off + seq_offsets[:, None]*stride_qs + d_offsets[None, :]*stride_qd, mask=seq_mask[:, None]&d_mask[None, :], other=0.0).to(tl.float32)
    K_c = tl.load(K + k_off + seq_offsets[:, None]*stride_ks + d_offsets[None, :]*stride_kd, mask=seq_mask[:, None]&d_mask[None, :], other=0.0).to(tl.float32)
    V_c = tl.load(V + v_off + seq_offsets[:, None]*stride_vs + d_offsets[None, :]*stride_vd, mask=seq_mask[:, None]&d_mask[None, :], other=0.0).to(tl.float32)
    gates = tl.load(G + g_off + seq_offsets*stride_gs, mask=seq_mask, other=0.0).to(tl.float32)
    
    state_off = pid_batch*stride_sb + pid_head*stride_sh + pid_chunk*stride_sd
    prev_state = tl.load(State_in + state_off + d_offsets, mask=d_mask, other=0.0).to(tl.float32)
    
    scores = tl.dot(Q_c, tl.trans(K_c))
    causal_mask = seq_offsets[:, None] >= seq_offsets[None, :]
    scores = tl.where(causal_mask & seq_mask[:, None] & seq_mask[None, :], scores, float("-inf"))
    attn = tl.exp(scores - tl.max(scores, 1)[:, None])
    attn = attn / tl.sum(attn, 1)[:, None]
    
    out_local = tl.dot(attn * gates[:, None], V_c)
    cumulative_gates = tl.cumprod(gates)
    state_contrib = prev_state[None, :] * cumulative_gates[:, None]
    out_chunk = out_local + state_contrib
    
    last_mask = seq_offsets == (chunk_end - 1)
    new_state = tl.sum(tl.where(last_mask[:, None], out_chunk, 0.0), axis=0)
    
    state_out_off = pid_batch*stride_sb + pid_head*stride_sh + (pid_chunk + 1)*stride_sd
    tl.store(State_out + state_out_off + d_offsets, new_state, mask=d_mask)
    tl.store(Out + pid_batch*stride_ob + pid_head*stride_oh + seq_offsets[:, None]*stride_os + d_offsets[None, :]*stride_od, out_chunk, mask=seq_mask[:, None] & d_mask[None, :])


# ============================================================================
# 3. LIZARD LAYER
# ============================================================================

class LizardLayer(nn.Module):
    def __init__(self, d_model, n_heads, window_size=64, chunk_size=64):
        super().__init__()
        self.d_model = d_model
        self.n_heads = n_heads
        self.d_head = d_model // n_heads
        self.window_size = window_size
        self.chunk_size = chunk_size
        
    def forward(self, q, k, v, g):
        return self.run_awa(q, k, v) + self.run_gla(q, k, v, g)

    def run_awa(self, q, k, v):
        batch, heads, seq, d = q.shape
        out = torch.empty_like(q)
        
        BLOCK_Q = 64
        BLOCK_K = 64
        BLOCK_D = triton.next_power_of_2(self.d_head)
        
        grid = (triton.cdiv(seq, BLOCK_Q), batch, heads)
        
        anchor_window_fwd_kernel_optimized[grid](
            q, k, v, out,
            q.stride(0), q.stride(1), q.stride(2), q.stride(3),
            k.stride(0), k.stride(1), k.stride(2), k.stride(3),
            v.stride(0), v.stride(1), v.stride(2), v.stride(3),
            out.stride(0), out.stride(1), out.stride(2), out.stride(3),
            seq, self.d_head, self.window_size,
            BLOCK_Q=BLOCK_Q, BLOCK_K=BLOCK_K, BLOCK_DMODEL=BLOCK_D,
            num_stages=2, num_warps=4
        )
        return out

    def run_gla(self, q, k, v, g):
        batch, heads, seq, d = q.shape
        out = torch.empty_like(q)
        num_chunks = triton.cdiv(seq, self.chunk_size)
        state_in = torch.zeros(batch, heads, num_chunks + 1, d, device=q.device, dtype=q.dtype)
        state_out = torch.zeros_like(state_in)
        BLOCK_D = triton.next_power_of_2(self.d_head)
        grid = (batch, heads, num_chunks)
        
        gla_chunk_fwd_kernel[grid](
            q, k, v, g, out, state_in, state_out,
            seq, self.d_head, self.chunk_size,
            q.stride(0), q.stride(1), q.stride(2), q.stride(3),
            k.stride(0), k.stride(1), k.stride(2), k.stride(3),
            v.stride(0), v.stride(1), v.stride(2), v.stride(3),
            g.stride(0), g.stride(1), g.stride(2),
            out.stride(0), out.stride(1), out.stride(2), out.stride(3),
            state_in.stride(0), state_in.stride(1), state_in.stride(3),
            BLOCK_D=BLOCK_D, BLOCK_CHUNK=self.chunk_size
        )
        return out

# ============================================================================
# 4. BENCHMARK
# ============================================================================

def benchmark_final_robust():
    if not torch.cuda.is_available(): return
    
    BATCH = 4
    HEADS = 8
    DIM = 128
    D_MODEL = HEADS * DIM
    # Now that we have 64-bit pointers, 16k and 32k should be safe
    SEQ_LENS = [1024, 2048, 4096, 8192, 16384, 32768]
    DTYPE = torch.bfloat16
    
    lizard = LizardLayer(D_MODEL, HEADS).cuda().to(DTYPE)
    
    if FLA_AVAILABLE:
        fla_model = GatedLinearAttention(hidden_size=D_MODEL, num_heads=HEADS, mode='fused_chunk').cuda().to(DTYPE)
    
    print(f"{'Seq':<8} | {'FLA (Ref)':<12} | {'AWA (New)':<12} | {'Lizard Total':<12}")
    print("-" * 60)
    
    results = {'seq':[], 'fla':[], 'awa':[], 'liz':[]}
    
    for s in SEQ_LENS:
        q = torch.randn(BATCH, HEADS, s, DIM, device='cuda', dtype=DTYPE)
        k = torch.randn(BATCH, HEADS, s, DIM, device='cuda', dtype=DTYPE)
        v = torch.randn(BATCH, HEADS, s, DIM, device='cuda', dtype=DTYPE)
        g = torch.sigmoid(torch.randn(BATCH, HEADS, s, device='cuda', dtype=torch.float32)).to(DTYPE)
        x_fla = torch.randn(BATCH, s, D_MODEL, device='cuda', dtype=DTYPE)
        
        def get_time(fn, *args):
            for _ in range(2): fn(*args)
            torch.cuda.synchronize()
            start = torch.cuda.Event(enable_timing=True)
            end = torch.cuda.Event(enable_timing=True)
            start.record()
            for _ in range(5): fn(*args)
            end.record()
            torch.cuda.synchronize()
            return start.elapsed_time(end) / 5

        try:
            t_awa = get_time(lizard.run_awa, q, k, v)
            t_liz = get_time(lizard.forward, q, k, v, g)
            if FLA_AVAILABLE:
                t_fla = get_time(lambda x: fla_model(x), x_fla)
            else:
                t_fla = 0.0
                
            print(f"{s:<8} | {t_fla:<12.2f} | {t_awa:<12.2f} | {t_liz:<12.2f}")
            
            results['seq'].append(s)
            results['fla'].append(t_fla)
            results['awa'].append(t_awa)
            results['liz'].append(t_liz)
            
        except torch.cuda.OutOfMemoryError:
            print(f"{s:<8} | OOM")
            break
        except Exception as e:
            print(f"{s:<8} | Error: {e}")
            break

    plt.figure(figsize=(10,6))
    if FLA_AVAILABLE:
        plt.plot(results['seq'], results['fla'], 'k--', linewidth=2, label='FLA (Reference)')
    plt.plot(results['seq'], results['awa'], 's-', label='Optimized AWA')
    plt.plot(results['seq'], results['liz'], '^-', linewidth=3, label='Optimized Lizard')
    plt.title(f'Lizard vs FLA (32k Support)\nBatch={BATCH}, D={DIM}x{HEADS}')
    plt.xlabel('Sequence Length')
    plt.ylabel('Latency (ms)')
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.savefig('lizard_final_32k.png')
    print("\nSaved chart to 'lizard_final_32k.png'")

if __name__ == "__main__":
    benchmark_final_robust()