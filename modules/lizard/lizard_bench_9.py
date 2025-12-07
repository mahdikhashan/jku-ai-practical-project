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
# 1. ROBUST AWA KERNEL (64-bit Pointers + BFloat16 Support)
# ============================================================================

@triton.jit
def anchor_window_fwd_kernel(
    Q, K, V, Out,
    seq_len, d_head,
    local_window, 
    # STRIDES MUST BE INT64 TO PREVENT OVERFLOW AT 16k+
    stride_qb: tl.int64, stride_qh: tl.int64, stride_qs: tl.int64, stride_qd: tl.int64,
    stride_kb: tl.int64, stride_kh: tl.int64, stride_ks: tl.int64, stride_kd: tl.int64,
    stride_vb: tl.int64, stride_vh: tl.int64, stride_vs: tl.int64, stride_vd: tl.int64,
    stride_ob: tl.int64, stride_oh: tl.int64, stride_os: tl.int64, stride_od: tl.int64,
    BLOCK_M: tl.constexpr, BLOCK_DMODEL: tl.constexpr,
):
    # Force 64-bit PID calculation
    pid_seq = tl.program_id(0).to(tl.int64)
    pid_batch = tl.program_id(1).to(tl.int64)
    pid_head = tl.program_id(2).to(tl.int64)
    
    if pid_seq >= seq_len: return
    
    # 64-bit Pointer Arithmetic
    q_offset = pid_batch * stride_qb + pid_head * stride_qh + pid_seq * stride_qs
    d_offsets = tl.arange(0, BLOCK_DMODEL)
    d_mask = d_offsets < d_head
    
    q_ptrs = Q + q_offset + d_offsets * stride_qd
    # tl.load handles bf16 automatically. We cast to fp32 for precision in exp/sum
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
    # Auto-cast back to bf16 on store if Out pointer is bf16
    tl.store(o_ptrs, acc, mask=d_mask)

# ============================================================================
# 2. ROBUST GLA KERNEL (64-bit + BFloat16)
# ============================================================================

@triton.jit
def gla_chunk_fwd_kernel(
    Q, K, V, G, Out, State_in, State_out,
    seq_len, d_model, chunk_size,
    # STRIDES MUST BE INT64
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
    actual_chunk_size = chunk_end - chunk_start
    if actual_chunk_size <= 0: return

    seq_offsets = chunk_start + tl.arange(0, BLOCK_CHUNK)
    d_offsets = tl.arange(0, BLOCK_D)
    seq_mask = seq_offsets < chunk_end
    d_mask = d_offsets < d_model
    
    q_offset = pid_batch * stride_qb + pid_head * stride_qh
    k_offset = pid_batch * stride_kb + pid_head * stride_kh
    v_offset = pid_batch * stride_vb + pid_head * stride_vh
    g_offset = pid_batch * stride_gb + pid_head * stride_gh
    
    q_ptrs = Q + q_offset + seq_offsets[:, None] * stride_qs + d_offsets[None, :] * stride_qd
    k_ptrs = K + k_offset + seq_offsets[:, None] * stride_ks + d_offsets[None, :] * stride_kd
    v_ptrs = V + v_offset + seq_offsets[:, None] * stride_vs + d_offsets[None, :] * stride_vd
    g_ptrs = G + g_offset + seq_offsets * stride_gs
    
    # Load bf16 -> cast to fp32 for math
    Q_c = tl.load(q_ptrs, mask=seq_mask[:, None] & d_mask[None, :], other=0.0).to(tl.float32)
    K_c = tl.load(k_ptrs, mask=seq_mask[:, None] & d_mask[None, :], other=0.0).to(tl.float32)
    V_c = tl.load(v_ptrs, mask=seq_mask[:, None] & d_mask[None, :], other=0.0).to(tl.float32)
    gates = tl.load(g_ptrs, mask=seq_mask, other=0.0).to(tl.float32)
    
    state_off = pid_batch * stride_sb + pid_head * stride_sh + pid_chunk * stride_sd
    prev_state = tl.load(State_in + state_off + d_offsets, mask=d_mask, other=0.0).to(tl.float32)
    
    # Tensor Core Dot (bf16 inputs supported, but here we used fp32 accumulator logic)
    # Note: On L4, tl.dot with fp32 inputs will use TF32 or standard CUDA cores.
    # To strictly use BF16 Tensor Cores, inputs to dot should be bf16.
    # However, kept as fp32 for maximum precision stability in this demo.
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
    
    state_out_off = pid_batch * stride_sb + pid_head * stride_sh + (pid_chunk + 1) * stride_sd
    tl.store(State_out + state_out_off + d_offsets, new_state, mask=d_mask)
    
    o_off = pid_batch * stride_ob + pid_head * stride_oh
    o_ptrs = Out + o_off + seq_offsets[:, None] * stride_os + d_offsets[None, :] * stride_od
    tl.store(o_ptrs, out_chunk, mask=seq_mask[:, None] & d_mask[None, :])

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
        # Hybrid: Local + Global
        return self.run_awa(q, k, v) + self.run_gla(q, k, v, g)

    def run_awa(self, q, k, v):
        batch, heads, seq, d = q.shape
        out = torch.empty_like(q)
        BLOCK_M = 16
        BLOCK_D = triton.next_power_of_2(self.d_head)
        grid = (seq, batch, heads)
        anchor_window_fwd_kernel[grid](
            q, k, v, out,
            seq, self.d_head, self.window_size,
            q.stride(0), q.stride(1), q.stride(2), q.stride(3),
            k.stride(0), k.stride(1), k.stride(2), k.stride(3),
            v.stride(0), v.stride(1), v.stride(2), v.stride(3),
            out.stride(0), out.stride(1), out.stride(2), out.stride(3),
            BLOCK_M=BLOCK_M, BLOCK_DMODEL=BLOCK_D
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
# 4. BENCHMARK WITH BFLOAT16
# ============================================================================

def benchmark_final():
    if not torch.cuda.is_available(): return
    
    BATCH = 4
    HEADS = 8
    DIM = 128
    D_MODEL = HEADS * DIM
    SEQ_LENS = [1024, 2048, 4096, 8192, 16384, 32768]
    DTYPE = torch.bfloat16  # <--- CHANGED TO BFLOAT16
    
    print(f"Benchmarking with DTYPE: {DTYPE}")

    my_lizard = LizardLayer(D_MODEL, HEADS).cuda().to(DTYPE)
    
    if FLA_AVAILABLE:
        fla_model = GatedLinearAttention(hidden_size=D_MODEL, num_heads=HEADS, mode='fused_chunk').cuda().to(DTYPE)
    
    print(f"{'Seq':<8} | {'FLA (Ref)':<12} | {'My GLA':<12} | {'My Lizard':<12} | {'Status':<10}")
    print("-" * 65)
    
    results = {'seq':[], 'fla':[], 'my_gla':[], 'my_liz':[]}
    
    for s in SEQ_LENS:
        # Generate BFLOAT16 Data
        q = torch.randn(BATCH, HEADS, s, DIM, device='cuda', dtype=DTYPE)
        k = torch.randn(BATCH, HEADS, s, DIM, device='cuda', dtype=DTYPE)
        v = torch.randn(BATCH, HEADS, s, DIM, device='cuda', dtype=DTYPE)
        g = torch.sigmoid(torch.randn(BATCH, HEADS, s, device='cuda', dtype=torch.float32)).to(DTYPE) # sig works best in fp32 then cast
        
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
            t_my_gla = get_time(my_lizard.run_gla, q, k, v, g)
            t_my_liz = get_time(my_lizard.forward, q, k, v, g)
            if FLA_AVAILABLE:
                t_fla = get_time(lambda x: fla_model(x), x_fla)
            else:
                t_fla = 0.0
                
            print(f"{s:<8} | {t_fla:<12.2f} | {t_my_gla:<12.2f} | {t_my_liz:<12.2f} | OK")
            
            results['seq'].append(s)
            results['fla'].append(t_fla)
            results['my_gla'].append(t_my_gla)
            results['my_liz'].append(t_my_liz)
            
        except torch.cuda.OutOfMemoryError:
            print(f"{s:<8} | {'-':<12} | {'-':<12} | {'-':<12} | OOM")
            break
        except Exception as e:
            print(f"{s:<8} | Error: {e}")
            break

    plt.figure(figsize=(10,6))
    if FLA_AVAILABLE:
        plt.plot(results['seq'], results['fla'], 'k--', linewidth=2, label='FLA (Reference)')
    plt.plot(results['seq'], results['my_gla'], 's-', label='My GLA (Custom)')
    plt.plot(results['seq'], results['my_liz'], '^-', linewidth=3, label='My Lizard (Total)')
    plt.title(f'Performance (BFloat16): Lizard vs FLA\nBatch={BATCH}, D={DIM}x{HEADS}')
    plt.xlabel('Sequence Length')
    plt.ylabel('Latency (ms)')
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.savefig('comparison_bf16.png')
    print("\nSaved chart to 'comparison_bf16.png'")

if __name__ == "__main__":
    benchmark_final()
