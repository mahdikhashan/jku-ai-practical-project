import torch
import triton
import triton.language as tl
import sys

# ============================================================================
# 1. TRITON KERNEL DEFINITION
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
    BLOCK_D: tl.constexpr,
    BLOCK_CHUNK: tl.constexpr,
):
    # Grid handling
    pid_batch = tl.program_id(0)
    pid_head = tl.program_id(1)
    pid_chunk = tl.program_id(2)
    
    chunk_start = pid_chunk * BLOCK_CHUNK
    chunk_end = tl.minimum(chunk_start + BLOCK_CHUNK, seq_len)
    
    # Range setup
    seq_offsets = chunk_start + tl.arange(0, BLOCK_CHUNK)
    d_offsets = tl.arange(0, BLOCK_D)
    seq_mask = seq_offsets < chunk_end
    d_mask = d_offsets < d_model
    
    # 1. Load Q, K, V (Tensor Core Friendly)
    q_offset = pid_batch * stride_qb + pid_head * stride_qh
    k_offset = pid_batch * stride_kb + pid_head * stride_kh
    v_offset = pid_batch * stride_vb + pid_head * stride_vh
    
    q_ptrs = Q + q_offset + seq_offsets[:, None] * stride_qs + d_offsets[None, :] * stride_qd
    k_ptrs = K + k_offset + seq_offsets[:, None] * stride_ks + d_offsets[None, :] * stride_kd
    v_ptrs = V + v_offset + seq_offsets[:, None] * stride_vs + d_offsets[None, :] * stride_vd
    
    # fp32 accumulator for precision, but loaded as whatever dtype is passed (usually fp16)
    Q_chunk = tl.load(q_ptrs, mask=seq_mask[:, None] & d_mask[None, :], other=0.0).to(tl.float32)
    K_chunk = tl.load(k_ptrs, mask=seq_mask[:, None] & d_mask[None, :], other=0.0).to(tl.float32)
    V_chunk = tl.load(v_ptrs, mask=seq_mask[:, None] & d_mask[None, :], other=0.0).to(tl.float32)
    
    # 2. Load Gates
    g_offset = pid_batch * stride_gb + pid_head * stride_gh
    g_ptrs = G + g_offset + seq_offsets * stride_gs
    gates = tl.load(g_ptrs, mask=seq_mask, other=0.0).to(tl.float32)
    
    # 3. Compute Attention (Chunk-Local)
    # Transpose K for [M, K] x [K, N] -> [M, N] multiplication
    K_T = tl.trans(K_chunk)
    
    # Tensor Core Operation 1: Q @ K.T
    scores = tl.dot(Q_chunk, K_T)
    
    # Causal Masking
    # chunk_offsets[i] >= chunk_offsets[j]
    causal_mask = seq_offsets[:, None] >= seq_offsets[None, :]
    scores = tl.where(causal_mask & seq_mask[:, None] & seq_mask[None, :], scores, float("-inf"))
    
    # Softmax
    scores_max = tl.max(scores, axis=1)
    scores_exp = tl.exp(scores - scores_max[:, None])
    scores_sum = tl.sum(scores_exp, axis=1)
    attn = scores_exp / scores_sum[:, None]
    
    # Apply Gates (Row-wise / Query-wise)
    gated_attn = attn * gates[:, None]
    
    # Tensor Core Operation 2: Attn @ V
    out_chunk = tl.dot(gated_attn, V_chunk)
    
    # 4. Store Result
    o_offset = pid_batch * stride_ob + pid_head * stride_oh
    o_ptrs = Out + o_offset + seq_offsets[:, None] * stride_os + d_offsets[None, :] * stride_od
    tl.store(o_ptrs, out_chunk, mask=seq_mask[:, None] & d_mask[None, :])

# ============================================================================
# 2. PYTHON WRAPPER
# ============================================================================

def gla_tensor_core(q, k, v, gates, chunk_size=64):
    """
    Stateless wrapper for correctness checking. 
    It runs the kernel but ignores the recurrent state logic (passing zeros/dummies).
    """
    batch, n_heads, seq_len, d_model = q.shape
    
    # Ensure Contiguity for stride calculations
    q, k, v, gates = q.contiguous(), k.contiguous(), v.contiguous(), gates.contiguous()
    out = torch.empty_like(q)
    
    # Dummies for state (required by kernel signature)
    num_chunks = triton.cdiv(seq_len, chunk_size)
    dummy_state = torch.zeros(batch, n_heads, num_chunks + 1, d_model, device=q.device, dtype=q.dtype)
    
    BLOCK_D = triton.next_power_of_2(d_model)
    
    grid = (batch, n_heads, num_chunks)
    
    gla_chunk_fwd_kernel[grid](
        q, k, v, gates, out, dummy_state, dummy_state,
        seq_len, d_model, chunk_size,
        q.stride(0), q.stride(1), q.stride(2), q.stride(3),
        k.stride(0), k.stride(1), k.stride(2), k.stride(3),
        v.stride(0), v.stride(1), v.stride(2), v.stride(3),
        gates.stride(0), gates.stride(1), gates.stride(2),
        out.stride(0), out.stride(1), out.stride(2), out.stride(3),
        dummy_state.stride(0), dummy_state.stride(1), dummy_state.stride(3),
        BLOCK_D=BLOCK_D,
        BLOCK_CHUNK=chunk_size,
    )
    return out

# ============================================================================
# 3. CORRECTNESS TEST RUNNER
# ============================================================================

def run_correctness_test():
    if not torch.cuda.is_available():
        print("No CUDA device found.")
        return

    print("\n" + "="*80)
    print("VERIFYING NUMERICAL CORRECTNESS")
    print("="*80)

    # Small Config for precision checks
    B, H, S, D = 2, 4, 1024, 128
    CHUNK = 64
    dtype = torch.float16
    device = "cuda"

    print(f"Checking Inputs: [{B}, {H}, {S}, {D}] | Chunk={CHUNK}")

    # 1. Setup Data
    torch.manual_seed(42)
    q = torch.randn(B, H, S, D, device=device, dtype=dtype)
    k = torch.randn(B, H, S, D, device=device, dtype=dtype)
    v = torch.randn(B, H, S, D, device=device, dtype=dtype)
    gates = torch.sigmoid(torch.randn(B, H, S, device=device, dtype=dtype))

    # 2. Define Ground Truth (PyTorch Native)
    # Matches the math: Block-Diagonal Attention + Query Gating
    def naive_chunked_gla(q, k, v, gates):
        # [Batch, Head, NumChunks, ChunkSize, D]
        q_c = q.reshape(B, H, S // CHUNK, CHUNK, D)
        k_c = k.reshape(B, H, S // CHUNK, CHUNK, D)
        v_c = v.reshape(B, H, S // CHUNK, CHUNK, D)
        g_c = gates.reshape(B, H, S // CHUNK, CHUNK)
        
        out_c = torch.zeros_like(q_c)
        
        for i in range(S // CHUNK):
            qc = q_c[:, :, i] 
            kc = k_c[:, :, i]
            vc = v_c[:, :, i]
            gc = g_c[:, :, i]
            
            # Q @ K.T
            scores = torch.matmul(qc, kc.transpose(-1, -2))
            
            # Causal Mask
            mask = torch.tril(torch.ones(CHUNK, CHUNK, device=device, dtype=torch.bool))
            scores.masked_fill_(~mask, float("-inf"))
            
            attn = torch.softmax(scores, dim=-1)
            
            # Apply Gates to Queries (broadcasting across K)
            # gc is [B,H,Chunk], unsqueeze -> [B,H,Chunk,1]
            gated_attn = attn * gc.unsqueeze(-1)
            
            # Attn @ V
            out_c[:, :, i] = torch.matmul(gated_attn, vc)
            
        return out_c.reshape(B, H, S, D)

    # 3. Run Both
    ref_out = naive_chunked_gla(q, k, v, gates)
    tri_out = gla_tensor_core(q, k, v, gates, chunk_size=CHUNK)

    # 4. Compare
    # Note: FP16 matmul accumulation usually results in errors ~1e-3
    diff = (ref_out - tri_out).abs()
    max_diff = diff.max().item()
    mean_diff = diff.mean().item()

    print("-" * 50)
    print(f"Max Difference:  {max_diff:.6f}")
    print(f"Mean Difference: {mean_diff:.6f}")
    print("-" * 50)

    # Tolerance check (approx 1e-2 is standard for FP16 attention)
    if max_diff < 0.05: 
        print("✅ PASS: Triton output matches PyTorch baseline!")
    else:
        print("❌ FAIL: Differences are too high.")

if __name__ == "__main__":
    run_correctness_test()
    