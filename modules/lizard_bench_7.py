import torch
import torch.nn as nn
import triton
import triton.language as tl
import matplotlib.pyplot as plt
import sys

# ============================================================================
# 1. KERNEL: AWA (Sliding Window / Local)
# ============================================================================

@triton.jit
def anchor_window_fwd_kernel(
    Q, K, V, Out,
    seq_len, d_head,
    local_window, 
    stride_qb, stride_qh, stride_qs, stride_qd,
    stride_kb, stride_kh, stride_ks, stride_kd,
    stride_vb, stride_vh, stride_vs, stride_vd,
    stride_ob, stride_oh, stride_os, stride_od,
    BLOCK_M: tl.constexpr, BLOCK_DMODEL: tl.constexpr,
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
# 2. KERNEL: GLA (Recurrent / Global) with Tensor Cores
# ============================================================================

@triton.jit
def gla_chunk_fwd_kernel(
    Q, K, V, G, Out, State_in, State_out,
    seq_len, d_model, chunk_size,
    stride_qb, stride_qh, stride_qs, stride_qd,
    stride_kb, stride_kh, stride_ks, stride_kd,
    stride_vb, stride_vh, stride_vs, stride_vd,
    stride_gb, stride_gh, stride_gs,
    stride_ob, stride_oh, stride_os, stride_od,
    stride_sb, stride_sh, stride_sd,
    BLOCK_D: tl.constexpr, BLOCK_CHUNK: tl.constexpr,
):
    pid_batch = tl.program_id(0)
    pid_head = tl.program_id(1)
    pid_chunk = tl.program_id(2)
    
    chunk_start = pid_chunk * BLOCK_CHUNK
    chunk_end = tl.minimum(chunk_start + BLOCK_CHUNK, seq_len)
    actual_chunk_size = chunk_end - chunk_start
    if actual_chunk_size <= 0: return

    seq_offsets = chunk_start + tl.arange(0, BLOCK_CHUNK)
    d_offsets = tl.arange(0, BLOCK_D)
    seq_mask = seq_offsets < chunk_end
    d_mask = d_offsets < d_model
    
    # Strides & Pointers
    q_offset = pid_batch * stride_qb + pid_head * stride_qh
    k_offset = pid_batch * stride_kb + pid_head * stride_kh
    v_offset = pid_batch * stride_vb + pid_head * stride_vh
    g_offset = pid_batch * stride_gb + pid_head * stride_gh
    
    q_ptrs = Q + q_offset + seq_offsets[:, None] * stride_qs + d_offsets[None, :] * stride_qd
    k_ptrs = K + k_offset + seq_offsets[:, None] * stride_ks + d_offsets[None, :] * stride_kd
    v_ptrs = V + v_offset + seq_offsets[:, None] * stride_vs + d_offsets[None, :] * stride_vd
    g_ptrs = G + g_offset + seq_offsets * stride_gs
    
    # Load Data
    Q_c = tl.load(q_ptrs, mask=seq_mask[:, None] & d_mask[None, :], other=0.0).to(tl.float32)
    K_c = tl.load(k_ptrs, mask=seq_mask[:, None] & d_mask[None, :], other=0.0).to(tl.float32)
    V_c = tl.load(v_ptrs, mask=seq_mask[:, None] & d_mask[None, :], other=0.0).to(tl.float32)
    gates = tl.load(g_ptrs, mask=seq_mask, other=0.0).to(tl.float32)
    
    # Load State
    state_off = pid_batch * stride_sb + pid_head * stride_sh + pid_chunk * stride_sd
    prev_state = tl.load(State_in + state_off + d_offsets, mask=d_mask, other=0.0).to(tl.float32)
    
    # Compute: Attention
    scores = tl.dot(Q_c, tl.trans(K_c)) # Tensor Core
    causal_mask = seq_offsets[:, None] >= seq_offsets[None, :]
    scores = tl.where(causal_mask & seq_mask[:, None] & seq_mask[None, :], scores, float("-inf"))
    
    attn = tl.exp(scores - tl.max(scores, 1)[:, None])
    attn = attn / tl.sum(attn, 1)[:, None]
    
    # Compute: Output
    out_local = tl.dot(attn * gates[:, None], V_c) # Tensor Core
    
    # Recurrence: Add decaying state
    # Simplified decay for this demo (cumprod of gates)
    cumulative_gates = tl.cumprod(gates) # 1D approximation
    state_contrib = prev_state[None, :] * cumulative_gates[:, None]
    out_chunk = out_local + state_contrib
    
    # Save State
    last_mask = seq_offsets == (chunk_end - 1)
    new_state = tl.sum(tl.where(last_mask[:, None], out_chunk, 0.0), axis=0)
    
    state_out_off = pid_batch * stride_sb + pid_head * stride_sh + (pid_chunk + 1) * stride_sd
    tl.store(State_out + state_out_off + d_offsets, new_state, mask=d_mask)
    
    # Store Output
    o_off = pid_batch * stride_ob + pid_head * stride_oh
    o_ptrs = Out + o_off + seq_offsets[:, None] * stride_os + d_offsets[None, :] * stride_od
    tl.store(o_ptrs, out_chunk, mask=seq_mask[:, None] & d_mask[None, :])

# ============================================================================
# 3. LIZARD LAYER MODULE
# ============================================================================

class LizardLayer(nn.Module):
    def __init__(self, d_model, n_heads, window_size=64, chunk_size=64):
        super().__init__()
        self.d_model = d_model
        self.n_heads = n_heads
        self.d_head = d_model // n_heads
        self.window_size = window_size
        self.chunk_size = chunk_size
        
        # In a real model, you'd have projection layers here (Wq, Wk, Wv)
        # We assume pre-projected inputs for the benchmark kernel
        
    def forward(self, q, k, v, g):
        # 1. Run Local Branch (AWA)
        local_out = self.run_awa(q, k, v)
        
        # 2. Run Global Branch (GLA)
        global_out = self.run_gla(q, k, v, g)
        
        # 3. Combine
        return local_out + global_out

    def run_awa(self, q, k, v):
        # AWA Wrapper
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
        # GLA Wrapper
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
# 4. FINAL BENCHMARK
# ============================================================================

def benchmark_all():
    if not torch.cuda.is_available(): return
    
    BATCH = 4
    HEADS = 8
    DIM = 128
    D_MODEL = HEADS * DIM
    SEQ_LENS = [1024, 2048, 4096, 8192, 16384, 32768]
    
    lizard = LizardLayer(D_MODEL, HEADS).cuda().half()
    
    print(f"{'Seq':<8} | {'Local(AWA)':<12} | {'Global(GLA)':<12} | {'LIZARD Total':<12}")
    print("-" * 55)
    
    results = {'seq':[], 'awa':[], 'gla':[], 'liz':[]}
    
    for s in SEQ_LENS:
        # Data Setup
        q = torch.randn(BATCH, HEADS, s, DIM, device='cuda', dtype=torch.float16)
        k = torch.randn(BATCH, HEADS, s, DIM, device='cuda', dtype=torch.float16)
        v = torch.randn(BATCH, HEADS, s, DIM, device='cuda', dtype=torch.float16)
        g = torch.sigmoid(torch.randn(BATCH, HEADS, s, device='cuda', dtype=torch.float16))
        
        # Benchmark Helper
        def get_time(fn, *args):
            # Warmup
            for _ in range(3): fn(*args)
            torch.cuda.synchronize()
            start = torch.cuda.Event(enable_timing=True)
            end = torch.cuda.Event(enable_timing=True)
            start.record()
            for _ in range(10): fn(*args)
            end.record()
            torch.cuda.synchronize()
            return start.elapsed_time(end) / 10
            
        try:
            t_awa = get_time(lizard.run_awa, q, k, v)
            t_gla = get_time(lizard.run_gla, q, k, v, g)
            t_liz = get_time(lizard.forward, q, k, v, g)
            
            print(f"{s:<8} | {t_awa:<12.2f} | {t_gla:<12.2f} | {t_liz:<12.2f}")
            
            results['seq'].append(s)
            results['awa'].append(t_awa)
            results['gla'].append(t_gla)
            results['liz'].append(t_liz)
            
        except torch.cuda.OutOfMemoryError:
            print(f"{s:<8} | OOM")
            break

    # Plot
    plt.figure(figsize=(10,6))
    plt.plot(results['seq'], results['awa'], 'o-', label='Local (AWA)')
    plt.plot(results['seq'], results['gla'], 's-', label='Global (GLA)')
    plt.plot(results['seq'], results['liz'], '^-', linewidth=3, label='Lizard (Total)')
    plt.title('Lizard Model: Component Latency')
    plt.xlabel('Sequence Length')
    plt.ylabel('Latency (ms)')
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.savefig('lizard_final_bench.png')
    print("\nSaved 'lizard_final_bench.png'")

if __name__ == "__main__":
    benchmark_all()
