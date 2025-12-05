import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl

@triton.jit
def _swa_fwd_kernel(
    Q, K, V, Out,
    stride_qz, stride_qh, stride_qm, stride_qk,
    stride_kz, stride_kh, stride_kn, stride_kk,
    stride_vz, stride_vh, stride_vn, stride_vk,
    stride_oz, stride_oh, stride_om, stride_ok,
    N_CTX, HEAD_DIM: tl.constexpr,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr,
    WINDOW_BWD: tl.constexpr, WINDOW_FWD: tl.constexpr,
):
    pid_m = tl.program_id(0)
    pid_batch = tl.program_id(1)
    pid_head = tl.program_id(2)
    
    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_d = tl.arange(0, HEAD_DIM)
    
    q_offset = pid_batch * stride_qz + pid_head * stride_qh
    k_offset = pid_batch * stride_kz + pid_head * stride_kh
    v_offset = pid_batch * stride_vz + pid_head * stride_vh
    o_offset = pid_batch * stride_oz + pid_head * stride_oh
    
    q_ptrs = Q + q_offset + offs_m[:, None] * stride_qm + offs_d[None, :] * stride_qk
    q = tl.load(q_ptrs, mask=offs_m[:, None] < N_CTX, other=0.0)
    
    acc = tl.zeros([BLOCK_M, HEAD_DIM], dtype=tl.float32)
    m_i = tl.zeros([BLOCK_M], dtype=tl.float32) - float("inf")
    l_i = tl.zeros([BLOCK_M], dtype=tl.float32)
    
    num_blocks = tl.cdiv(N_CTX, BLOCK_N)
    for block_n in range(num_blocks):
        start_n = block_n * BLOCK_N
        offs_n_curr = start_n + tl.arange(0, BLOCK_N)
        
        k_ptrs = K + k_offset + offs_n_curr[:, None] * stride_kn + offs_d[None, :] * stride_kk
        v_ptrs = V + v_offset + offs_n_curr[:, None] * stride_vn + offs_d[None, :] * stride_vk
        
        k = tl.load(k_ptrs, mask=offs_n_curr[:, None] < N_CTX, other=0.0)
        v = tl.load(v_ptrs, mask=offs_n_curr[:, None] < N_CTX, other=0.0)
        
        qk = tl.dot(q, tl.trans(k))
        
        mask = (offs_n_curr[None, :] >= offs_m[:, None] - WINDOW_BWD) & \
               (offs_n_curr[None, :] <= offs_m[:, None] + WINDOW_FWD) & \
               (offs_m[:, None] < N_CTX) & (offs_n_curr[None, :] < N_CTX)
        
        qk = tl.where(mask, qk, float("-inf"))
        
        m_ij = tl.max(qk, axis=1)
        
        # Only update if current block has valid values
        block_has_valid = m_ij > float("-inf")
        m_i_new = tl.where(block_has_valid, tl.maximum(m_i, m_ij), m_i)
        
        # Compute alpha - but only if both m_i and m_i_new are finite
        # If current block is all masked (m_ij = -inf), skip it entirely
        alpha = tl.where(block_has_valid, tl.exp(m_i - m_i_new), 1.0)
        
        # Compute p - only non-zero if block has valid values
        p = tl.where(block_has_valid[:, None], tl.exp(qk - m_i_new[:, None]), 0.0)
        
        # Update accumulator
        acc = acc * alpha[:, None]
        acc = acc + tl.dot(p.to(v.dtype), v)
        
        # Update denominator
        l_i = l_i * alpha + tl.sum(p, axis=1)
        
        # Update max
        m_i = m_i_new
    
    # Final normalization
    valid_rows = l_i > 0
    acc = tl.where(valid_rows[:, None], acc / l_i[:, None], 0.0)
    
    o_ptrs = Out + o_offset + offs_m[:, None] * stride_om + offs_d[None, :] * stride_ok
    tl.store(o_ptrs, acc.to(Out.dtype.element_ty), mask=offs_m[:, None] < N_CTX)


def sliding_window_attention_fwd(q, k, v, window_sizes):
    BATCH, HEADS, SEQ, HEAD_DIM = q.shape
    bwd, fwd = window_sizes
    
    assert HEAD_DIM in {16, 32, 64, 128}, f"HEAD_DIM must be 16/32/64/128, got {HEAD_DIM}"
    
    BLOCK_M = 64
    BLOCK_N = 64
    
    q, k, v = q.contiguous(), k.contiguous(), v.contiguous()
    o = torch.empty_like(q)
    
    grid = (triton.cdiv(SEQ, BLOCK_M), BATCH, HEADS)
    
    _swa_fwd_kernel[grid](
        q, k, v, o,
        q.stride(0), q.stride(1), q.stride(2), q.stride(3),
        k.stride(0), k.stride(1), k.stride(2), k.stride(3),
        v.stride(0), v.stride(1), v.stride(2), v.stride(3),
        o.stride(0), o.stride(1), o.stride(2), o.stride(3),
        SEQ, HEAD_DIM,
        BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N,
        WINDOW_BWD=bwd, WINDOW_FWD=fwd,
    )
    
    return o


if __name__ == "__main__":
    print(f"Device: {torch.cuda.get_device_name(0)}")
    
    print("\n--- Simple Test ---")
    torch.manual_seed(0)
    B, H, N, D = 1, 1, 128, 64
    window = (32, 32)
    
    q = torch.randn(B, H, N, D, device='cuda', dtype=torch.float32)
    k = torch.randn(B, H, N, D, device='cuda', dtype=torch.float32)
    v = torch.randn(B, H, N, D, device='cuda', dtype=torch.float32)
    
    out_triton = sliding_window_attention_fwd(q, k, v, window)
    
    qk = torch.matmul(q, k.transpose(-2, -1))
    row_idx = torch.arange(N, device='cuda').unsqueeze(-1)
    col_idx = torch.arange(N, device='cuda').unsqueeze(-2)
    mask = (col_idx >= row_idx - window[0]) & (col_idx <= row_idx + window[1])
    qk = qk.masked_fill(~mask, float('-inf'))
    attn = torch.softmax(qk, dim=-1)
    out_ref = torch.matmul(attn, v)
    
    diff = (out_triton - out_ref).abs()
    print(f"Max Diff: {diff.max().item():.6f}")
    print(f"Mean Diff: {diff.mean().item():.6f}")
    
    if diff.max() < 1e-4:
        print("✅ Correctness verified!")
    else:
        print("❌ Still has errors")
        # Debug: print attention weights to see if softmax is correct
        print("\nDebugging row 10:")
        print(f"Triton output[0,0,10,:10]: {out_triton[0,0,10,:10]}")
        print(f"Ref output[0,0,10,:10]: {out_ref[0,0,10,:10]}")
