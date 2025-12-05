import torch
import triton
import triton.language as tl


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
    """
    Anchor Window Attention forward kernel.
    Each token attends to:
    1. Local window (nearby tokens)
    2. First num_meta_tokens tokens (positions 0, 1, 2, ... num_meta_tokens-1)
    """
    pid_seq = tl.program_id(0)
    pid_batch = tl.program_id(1)
    pid_head = tl.program_id(2)
    
    # Early exit for empty windows
    if pid_seq >= seq_len:
        return
    
    # Load query vector [d_head]
    q_offset = pid_batch * stride_qb + pid_head * stride_qh + pid_seq * stride_qs
    d_offsets = tl.arange(0, BLOCK_DMODEL)
    d_mask = d_offsets < d_head
    
    q_ptrs = Q + q_offset + d_offsets * stride_qd
    q = tl.load(q_ptrs, mask=d_mask, other=0.0).to(tl.float32)
    
    # Initialize for online softmax
    m_i = float("-inf")
    l_i = 0.0
    acc = tl.zeros([BLOCK_DMODEL], dtype=tl.float32)
    
    # Phase 1: Process local window
    local_start = tl.maximum(0, pid_seq - local_window)
    local_end = tl.minimum(seq_len, pid_seq + local_window + 1)
    
    k_block_start = local_start
    while k_block_start < local_end:
        k_offsets = k_block_start + tl.arange(0, BLOCK_M)
        k_valid = k_offsets < local_end
        
        # Load K block
        k_base = pid_batch * stride_kb + pid_head * stride_kh
        k_ptrs = K + k_base + k_offsets[:, None] * stride_ks + d_offsets[None, :] * stride_kd
        k = tl.load(k_ptrs, mask=k_valid[:, None] & d_mask[None, :], other=0.0).to(tl.float32)
        
        # Compute scores
        qk = tl.sum(q[None, :] * k, axis=1)
        qk = tl.where(k_valid, qk, float("-inf"))
        
        # Online softmax
        m_ij = tl.max(qk, axis=0)
        m_ij = tl.maximum(m_i, m_ij)
        alpha = tl.exp(m_i - m_ij)
        p = tl.exp(qk - m_ij)
        l_ij = tl.sum(p, axis=0)
        
        # Load V block
        v_base = pid_batch * stride_vb + pid_head * stride_vh
        v_ptrs = V + v_base + k_offsets[:, None] * stride_vs + d_offsets[None, :] * stride_vd
        v = tl.load(v_ptrs, mask=k_valid[:, None] & d_mask[None, :], other=0.0).to(tl.float32)
        
        # Update accumulator
        acc = acc * alpha + tl.sum(p[:, None] * v, axis=0)
        l_i = l_i * alpha + l_ij
        m_i = m_ij
        
        k_block_start += BLOCK_M
    
    # Phase 2: Process meta tokens (first num_meta_tokens positions)
    # Only process if they're not already in the local window
    if num_meta_tokens > 0:
        meta_end = tl.minimum(num_meta_tokens, seq_len)
        
        # Process meta tokens in blocks
        meta_idx = 0
        while meta_idx < meta_end:
            k_offsets = meta_idx + tl.arange(0, BLOCK_M)
            # Meta tokens that are outside the local window
            k_valid = (k_offsets < meta_end) & ((k_offsets < local_start) | (k_offsets >= local_end))
            
            # Load K block for meta tokens (zeros for invalid positions)
            k_base = pid_batch * stride_kb + pid_head * stride_kh
            k_ptrs = K + k_base + k_offsets[:, None] * stride_ks + d_offsets[None, :] * stride_kd
            k = tl.load(k_ptrs, mask=k_valid[:, None] & d_mask[None, :], other=0.0).to(tl.float32)
            
            # Compute scores (will be -inf for invalid positions)
            qk = tl.sum(q[None, :] * k, axis=1)
            qk = tl.where(k_valid, qk, float("-inf"))
            
            # Check if all values are -inf (no valid meta tokens)
            has_valid = tl.max(qk) > float("-inf")
            
            # Only update softmax if we have valid meta tokens
            if has_valid:
                # Online softmax
                m_ij = tl.max(qk, axis=0)
                m_ij = tl.maximum(m_i, m_ij)
                alpha = tl.exp(m_i - m_ij)
                p = tl.exp(qk - m_ij)
                l_ij = tl.sum(p, axis=0)
                
                # Load V block
                v_base = pid_batch * stride_vb + pid_head * stride_vh
                v_ptrs = V + v_base + k_offsets[:, None] * stride_vs + d_offsets[None, :] * stride_vd
                v = tl.load(v_ptrs, mask=k_valid[:, None] & d_mask[None, :], other=0.0).to(tl.float32)
                
                # Update accumulator
                acc = acc * alpha + tl.sum(p[:, None] * v, axis=0)
                l_i = l_i * alpha + l_ij
                m_i = m_ij
            
            meta_idx += BLOCK_M
    
    # Final normalization
    acc = acc / l_i
    
    # Store output
    o_offset = pid_batch * stride_ob + pid_head * stride_oh + pid_seq * stride_os
    o_ptrs = Out + o_offset + d_offsets * stride_od
    tl.store(o_ptrs, acc, mask=d_mask)


def anchor_window_attention(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    local_window: int = 64,
    num_meta_tokens: int = 4,
) -> torch.Tensor:
    """
    Anchor Window Attention using Triton.
    
    Each token attends to:
    - Local window: tokens within ±local_window
    - Meta tokens: first num_meta_tokens positions (e.g., 2, 4, 6, or 8)
    
    Args:
        q: Query tensor [batch, n_heads, seq_len, d_head]
        k: Key tensor [batch, n_heads, seq_len, d_head]
        v: Value tensor [batch, n_heads, seq_len, d_head]
        local_window: Size of local attention window
        num_meta_tokens: Number of meta tokens at the beginning (2, 4, 6, or 8)
        
    Returns:
        Output tensor [batch, n_heads, seq_len, d_head]
    """
    batch, n_heads, seq_len, d_head = q.shape
    
    q = q.contiguous()
    k = k.contiguous()
    v = v.contiguous()
    
    out = torch.empty_like(q)
    
    # Adjust BLOCK_M based on d_head
    if d_head <= 32:
        BLOCK_M = 64
    elif d_head <= 64:
        BLOCK_M = 16
    else:
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
        BLOCK_M=BLOCK_M,
        BLOCK_DMODEL=BLOCK_DMODEL,
    )
    
    return out


# ============================================================================
# Import sliding window attention from previous implementation
# ============================================================================

@triton.jit
def swa_fwd_kernel(
    Q, K, V, Out,
    seq_len, d_head,
    fwd_win_size, bwd_win_size,
    stride_qb, stride_qh, stride_qs, stride_qd,
    stride_kb, stride_kh, stride_ks, stride_kd,
    stride_vb, stride_vh, stride_vs, stride_vd,
    stride_ob, stride_oh, stride_os, stride_od,
    BLOCK_M: tl.constexpr,
    BLOCK_DMODEL: tl.constexpr,
):
    """Sliding Window Attention kernel."""
    pid_seq = tl.program_id(0)
    pid_batch = tl.program_id(1)
    pid_head = tl.program_id(2)
    
    win_start = tl.maximum(0, pid_seq - fwd_win_size)
    win_end = tl.minimum(seq_len, pid_seq + bwd_win_size + 1)
    
    if win_end <= win_start:
        return
    
    q_offset = pid_batch * stride_qb + pid_head * stride_qh + pid_seq * stride_qs
    d_offsets = tl.arange(0, BLOCK_DMODEL)
    d_mask = d_offsets < d_head
    
    q_ptrs = Q + q_offset + d_offsets * stride_qd
    q = tl.load(q_ptrs, mask=d_mask, other=0.0).to(tl.float32)
    
    m_i = float("-inf")
    l_i = 0.0
    acc = tl.zeros([BLOCK_DMODEL], dtype=tl.float32)
    
    k_block_start = win_start
    
    while k_block_start < win_end:
        k_offsets = k_block_start + tl.arange(0, BLOCK_M)
        k_valid = k_offsets < win_end
        
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


def sliding_window_attention(q, k, v, window_sizes=(31, 32)):
    """Sliding window attention wrapper."""
    batch, n_heads, seq_len, d_head = q.shape
    fwd_win_size, bwd_win_size = window_sizes
    
    q = q.contiguous()
    k = k.contiguous()
    v = v.contiguous()
    
    out = torch.empty_like(q)
    
    if d_head <= 32:
        BLOCK_M = 64
    elif d_head <= 64:
        BLOCK_M = 16
    else:
        BLOCK_M = 16
    
    BLOCK_DMODEL = triton.next_power_of_2(d_head)
    grid = (seq_len, batch, n_heads)
    
    swa_fwd_kernel[grid](
        q, k, v, out,
        seq_len, d_head,
        fwd_win_size, bwd_win_size,
        q.stride(0), q.stride(1), q.stride(2), q.stride(3),
        k.stride(0), k.stride(1), k.stride(2), k.stride(3),
        v.stride(0), v.stride(1), v.stride(2), v.stride(3),
        out.stride(0), out.stride(1), out.stride(2), out.stride(3),
        BLOCK_M=BLOCK_M,
        BLOCK_DMODEL=BLOCK_DMODEL,
    )
    
    return out


# ============================================================================
# Naive reference implementations
# ============================================================================

def anchor_window_naive(q, k, v, local_window=64, num_meta_tokens=4):
    """Naive implementation of anchor window attention with meta tokens."""
    batch, n_heads, seq_len, d_head = q.shape
    
    # Create attention mask
    row_idx = torch.arange(seq_len, device=q.device).unsqueeze(-1)
    col_idx = torch.arange(seq_len, device=q.device).unsqueeze(-2)
    
    # Local window mask
    local_mask = (col_idx >= row_idx - local_window) & (col_idx <= row_idx + local_window)
    
    # Meta tokens mask: first num_meta_tokens positions
    meta_mask = torch.zeros(seq_len, seq_len, dtype=torch.bool, device=q.device)
    if num_meta_tokens > 0:
        meta_mask[:, :num_meta_tokens] = True
    
    # Combined mask
    valid_mask = local_mask | meta_mask
    
    qk = q @ k.transpose(-1, -2)
    qk_masked = torch.masked_fill(qk, ~valid_mask, -float("inf"))
    a = torch.softmax(qk_masked, dim=-1)
    return a @ v


# ============================================================================
# Testing and benchmarking
# ============================================================================

def test_correctness():
    """Test anchor window attention."""
    print("=" * 80)
    print("Testing Anchor Window Attention with Meta Tokens")
    print("=" * 80)
    
    torch.manual_seed(42)
    
    batch, n_heads, seq_len, d_head = 2, 4, 512, 64
    local_window = 64
    num_meta_tokens = 4  # Can be 2, 4, 6, or 8
    
    q = torch.randn(batch, n_heads, seq_len, d_head, device='cuda')
    k = torch.randn(batch, n_heads, seq_len, d_head, device='cuda')
    v = torch.randn(batch, n_heads, seq_len, d_head, device='cuda')
    
    out_ref = anchor_window_naive(q, k, v, local_window, num_meta_tokens)
    out_triton = anchor_window_attention(q, k, v, local_window, num_meta_tokens)
    
    # Manual verification on random queries
    print(f"\nConfig: local_window={local_window}, num_meta_tokens={num_meta_tokens}")
    print("=== Verifying correctness with manual computation ===")
    errors = []
    for _ in range(10):
        b = torch.randint(0, batch, (1,)).item()
        h = torch.randint(0, n_heads, (1,)).item()
        s = torch.randint(0, seq_len, (1,)).item()
        
        # Build attention indices manually
        local_start = max(0, s - local_window)
        local_end = min(seq_len, s + local_window + 1)
        local_indices = list(range(local_start, local_end))
        
        # Add meta token indices (first num_meta_tokens positions)
        meta_indices = list(range(min(num_meta_tokens, seq_len)))
        meta_indices = [i for i in meta_indices if i not in local_indices]
        
        all_indices = sorted(local_indices + meta_indices)
        
        # Manual computation
        q_vec = q[b, h, s].cpu()
        k_window = k[b, h, all_indices].cpu()
        v_window = v[b, h, all_indices].cpu()
        
        scores = q_vec @ k_window.T
        attn = torch.softmax(scores, dim=-1)
        manual_out = attn @ v_window
        
        triton_out = out_triton[b, h, s].cpu()
        error = (manual_out - triton_out).abs().max().item()
        errors.append(error)
    
    max_error = max(errors)
    print(f"Max error vs manual: {max_error:.6f}")
    
    if max_error < 1e-3:
        print("✓ Anchor window attention is correct!")
    else:
        print(f"✗ Error too large: {max_error}")
        
    # Compare with naive
    max_diff = (out_ref - out_triton).abs().max().item()
    print(f"Max diff vs naive PyTorch: {max_diff:.6f}")
    
    # Show effective attention pattern
    print(f"\nExample: Token at position 200 attends to:")
    print(f"  Local: [{max(0, 200-local_window)}, {min(seq_len, 200+local_window+1)}) = {2*local_window+1} tokens")
    print(f"  Meta: [0, {num_meta_tokens}) = {num_meta_tokens} tokens")
    print(f"  Total keys: ~{2*local_window+1+num_meta_tokens} tokens")


def benchmark_comparison():
    """Compare sliding window vs anchor window attention."""
    import time
    
    print("\n" + "=" * 80)
    print("Benchmarking: Sliding Window vs Anchor Window (with Meta Tokens)")
    print("=" * 80)
    
    # Test different numbers of meta tokens
    meta_token_configs = [2, 4, 6, 8]
    
    configs = [
        # (batch, heads, seq_len, d_head, local_win)
        (2, 8, 2048, 64, 64),
        (2, 8, 4096, 64, 64),
        (2, 8, 8192, 64, 64),
        (1, 8, 16384, 64, 64),
    ]
    
    n_iters = 30
    
    for batch, n_heads, seq_len, d_head, local_win in configs:
        print(f"\n{'='*80}")
        print(f"Config: B={batch}, H={n_heads}, S={seq_len}, local_window={local_win}")
        print(f"{'='*80}")
        print(f"{'Method':<30} {'Time (ms)':<12} {'Keys/Query':<12} {'Rel. Speed':<12}")
        print("-" * 80)
        
        q = torch.randn(batch, n_heads, seq_len, d_head, device='cuda')
        k = torch.randn(batch, n_heads, seq_len, d_head, device='cuda')
        v = torch.randn(batch, n_heads, seq_len, d_head, device='cuda')
        
        # Benchmark sliding window (baseline)
        for _ in range(5):
            _ = sliding_window_attention(q, k, v, (local_win, local_win))
        torch.cuda.synchronize()
        
        start = time.time()
        for _ in range(n_iters):
            _ = sliding_window_attention(q, k, v, (local_win, local_win))
        torch.cuda.synchronize()
        swa_time = (time.time() - start) * 1000 / n_iters
        swa_keys = 2 * local_win + 1
        
        print(f"{'Sliding Window (baseline)':<30} {swa_time:<12.3f} {swa_keys:<12} {1.0:<12.2f}x")
        
        # Benchmark anchor window with different meta token counts
        for num_meta in meta_token_configs:
            for _ in range(5):
                _ = anchor_window_attention(q, k, v, local_win, num_meta)
            torch.cuda.synchronize()
            
            start = time.time()
            for _ in range(n_iters):
                _ = anchor_window_attention(q, k, v, local_win, num_meta)
            torch.cuda.synchronize()
            anchor_time = (time.time() - start) * 1000 / n_iters
            
            anchor_keys = (2 * local_win + 1) + num_meta
            rel_speed = swa_time / anchor_time
            
            print(f"{'  + ' + str(num_meta) + ' meta tokens':<30} {anchor_time:<12.3f} {anchor_keys:<12} {rel_speed:<12.2f}x")


if __name__ == "__main__":
    test_correctness()
    benchmark_comparison()
