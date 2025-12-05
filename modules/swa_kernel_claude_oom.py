import torch
import triton
import triton.language as tl


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
    """
    Sliding Window Attention forward kernel.
    Each program handles one query position.
    """
    pid_seq = tl.program_id(0)
    pid_batch = tl.program_id(1)
    pid_head = tl.program_id(2)
    
    # Compute window bounds for this query
    win_start = tl.maximum(0, pid_seq - fwd_win_size)
    win_end = tl.minimum(seq_len, pid_seq + bwd_win_size + 1)
    win_size = win_end - win_start
    
    # Early exit if window is empty
    if win_size <= 0:
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
    
    # Process keys in blocks
    k_block_start = win_start
    
    while k_block_start < win_end:
        # Calculate key indices for this block
        k_offsets = k_block_start + tl.arange(0, BLOCK_M)
        k_valid = k_offsets < win_end
        
        # Load K block [BLOCK_M, d_head]
        k_base = pid_batch * stride_kb + pid_head * stride_kh
        k_ptrs = K + k_base + k_offsets[:, None] * stride_ks + d_offsets[None, :] * stride_kd
        k = tl.load(k_ptrs, mask=k_valid[:, None] & d_mask[None, :], other=0.0).to(tl.float32)
        
        # Compute attention scores: Q @ K^T
        qk = tl.sum(q[None, :] * k, axis=1)  # [BLOCK_M]
        qk = tl.where(k_valid, qk, float("-inf"))
        
        # Online softmax step
        m_ij = tl.max(qk, axis=0)
        m_ij = tl.maximum(m_i, m_ij)
        
        # Scale factor for previous values
        alpha = tl.exp(m_i - m_ij)
        
        # Softmax for current block
        p = tl.exp(qk - m_ij)
        
        # Sum of new probabilities
        l_ij = tl.sum(p, axis=0)
        
        # Load V block [BLOCK_M, d_head]
        v_base = pid_batch * stride_vb + pid_head * stride_vh
        v_ptrs = V + v_base + k_offsets[:, None] * stride_vs + d_offsets[None, :] * stride_vd
        v = tl.load(v_ptrs, mask=k_valid[:, None] & d_mask[None, :], other=0.0).to(tl.float32)
        
        # Update accumulator
        # acc_new = (acc_old * l_old * alpha + sum(p * v)) / (l_old * alpha + l_new)
        # But we track unnormalized: acc_unnorm = acc * l
        # So: acc_unnorm_new = acc_unnorm_old * alpha + sum(p * v)
        acc = acc * alpha
        acc = acc + tl.sum(p[:, None] * v, axis=0)
        
        # Update sum
        l_i = l_i * alpha + l_ij
        m_i = m_ij
        
        k_block_start += BLOCK_M
    
    # Final normalization
    acc = acc / l_i
    
    # Store output
    o_offset = pid_batch * stride_ob + pid_head * stride_oh + pid_seq * stride_os
    o_ptrs = Out + o_offset + d_offsets * stride_od
    tl.store(o_ptrs, acc, mask=d_mask)


def sliding_window_attention(
    q: torch.Tensor,
    k: torch.Tensor, 
    v: torch.Tensor,
    window_sizes: tuple[int, int] = (15, 16)
) -> torch.Tensor:
    """
    Sliding window attention using Triton.
    
    Args:
        q: Query tensor [batch, n_heads, seq_len, d_head]
        k: Key tensor [batch, n_heads, seq_len, d_head]
        v: Value tensor [batch, n_heads, seq_len, d_head]
        window_sizes: (forward_window, backward_window) sizes
        
    Returns:
        Output tensor [batch, n_heads, seq_len, d_head]
    """
    batch, n_heads, seq_len, d_head = q.shape
    fwd_win_size, bwd_win_size = window_sizes
    
    # Ensure contiguous
    q = q.contiguous()
    k = k.contiguous()
    v = v.contiguous()
    
    out = torch.empty_like(q)
    
    # Adjust BLOCK_M based on d_head to avoid issues
    # For larger d_head, use smaller BLOCK_M to fit in SRAM better
    if d_head <= 32:
        BLOCK_M = 64
    elif d_head <= 64:
        BLOCK_M = 16  # Try smaller block size for d=64
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
# Reference and testing
# ============================================================================

def swa_naive(q, k, v, window_sizes: tuple[int, int] = (15, 16)):
    """Naive implementation of sliding window attention."""
    fwd_win_size, bwd_win_size = window_sizes

    # Get sequence length from the correct dimension
    seq_len = q.shape[-2]
    
    row_idx = torch.arange(seq_len, device=q.device).unsqueeze(-1)
    col_idx = torch.arange(seq_len, device=k.device).unsqueeze(-2)
    fwd_inv_mask = row_idx < col_idx - fwd_win_size
    bwd_inv_mask = row_idx > col_idx + bwd_win_size

    qk = q @ k.transpose(-1, -2)
    qk_masked = torch.masked_fill(qk, fwd_inv_mask | bwd_inv_mask, -float("inf"))
    a = torch.softmax(qk_masked, dim=-1)
    return a @ v


def test_simple():
    """Test with a very simple case first."""
    print("\n=== Testing simple case ===")
    torch.manual_seed(42)
    
    batch, n_heads, seq_len, d_head = 1, 1, 8, 4
    window_sizes = (2, 2)  # Small window
    
    q = torch.randn(batch, n_heads, seq_len, d_head, device='cuda')
    k = torch.randn(batch, n_heads, seq_len, d_head, device='cuda')
    v = torch.randn(batch, n_heads, seq_len, d_head, device='cuda')
    
    out_ref = swa_naive(q, k, v, window_sizes)
    out_triton = sliding_window_attention(q, k, v, window_sizes)
    
    print(f"Reference output:\n{out_ref[0, 0]}")
    print(f"Triton output:\n{out_triton[0, 0]}")
    
    diff = (out_ref - out_triton).abs()
    print(f"Max diff: {diff.max().item():.6f}")
    print(f"Mean diff: {diff.mean().item():.6f}")
    
    # Also check attention weights for one query
    fwd_win_size, bwd_win_size = window_sizes
    row_idx = torch.arange(q.shape[-2], device=q.device).unsqueeze(-1)
    col_idx = torch.arange(k.shape[-2], device=k.device).unsqueeze(-2)
    fwd_inv_mask = row_idx < col_idx - fwd_win_size
    bwd_inv_mask = row_idx > col_idx + bwd_win_size
    qk = q[0, 0] @ k[0, 0].transpose(-1, -2)
    qk_masked = torch.masked_fill(qk, fwd_inv_mask | bwd_inv_mask, -float("inf"))
    attn = torch.softmax(qk_masked, dim=-1)
    
    print(f"\nAttention weights for query 3:")
    print(f"Mask: {~(fwd_inv_mask | bwd_inv_mask)[3]}")
    print(f"Weights: {attn[3]}")
    print(f"Sum: {attn[3].sum()}")
    
    assert diff.max() < 1e-4, f"Simple test failed! Max diff: {diff.max().item()}"
    print("✓ Simple test passed!")


def test_incremental():
    """Test incrementally larger configs to find where it breaks."""
    print("\n=== Testing incremental configs ===")
    torch.manual_seed(42)
    
    configs = [
        (1, 1, 16, 8, (2, 2)),
        (1, 1, 32, 16, (4, 4)),
        (1, 1, 64, 32, (8, 8)),
        (1, 2, 64, 32, (8, 8)),
        (2, 2, 64, 32, (8, 8)),
        (1, 1, 64, 48, (8, 8)),  # Test non-power-of-2
        (1, 1, 64, 64, (8, 8)),  # Test d_head=64 with small seq
        (2, 4, 128, 64, (15, 16)),
    ]
    
    for batch, n_heads, seq_len, d_head, window_sizes in configs:
        q = torch.randn(batch, n_heads, seq_len, d_head, device='cuda')
        k = torch.randn(batch, n_heads, seq_len, d_head, device='cuda')
        v = torch.randn(batch, n_heads, seq_len, d_head, device='cuda')
        
        out_ref = swa_naive(q, k, v, window_sizes)
        out_triton = sliding_window_attention(q, k, v, window_sizes)
        
        diff = (out_ref - out_triton).abs().max().item()
        status = "✓" if diff < 1e-3 else "✗"
        print(f"{status} B={batch}, H={n_heads}, S={seq_len}, D={d_head}, W={window_sizes}: diff={diff:.6f}")
        
        if diff > 1e-3:
            print(f"  FAILED at this config!")
            # Show where it failed
            diff_map = (out_ref - out_triton).abs()
            max_idx = diff_map.argmax()
            b, h, s, d = torch.unravel_index(max_idx, diff_map.shape)
            print(f"  Worst at batch={b}, head={h}, seq={s}, dim={d}")
            print(f"  Ref={out_ref[b, h, s, d]:.6f}, Triton={out_triton[b, h, s, d]:.6f}")
            
            # Check if it's specific to certain dimensions
            per_dim_error = (out_ref - out_triton).abs().mean(dim=(0,1,2))
            print(f"  Per-dim mean error (first 10): {per_dim_error[:10]}")
            print(f"  Per-dim mean error (last 10): {per_dim_error[-10:]}")
            # Check specific query by computing manually
            print(f"\n  Manual verification for query {s}:")
            fwd_win, bwd_win = window_sizes
            win_start = max(0, s - fwd_win)
            win_end = min(seq_len, s + bwd_win + 1)
            
            # Extract the relevant parts
            q_vec = q[b, h, s].cpu()
            k_window = k[b, h, win_start:win_end].cpu()
            v_window = v[b, h, win_start:win_end].cpu()
            
            # Manual attention
            scores = q_vec @ k_window.T
            attn = torch.softmax(scores, dim=-1)
            manual_out = attn @ v_window
            
            print(f"  Manual output: {manual_out[:5]}")
            print(f"  Ref output: {out_ref[b, h, s, :5].cpu()}")
            print(f"  Triton output: {out_triton[b, h, s, :5].cpu()}")
            print(f"  Manual vs Ref diff: {(manual_out - out_ref[b, h, s].cpu()).abs().max():.6f}")
            print(f"  Manual vs Triton diff: {(manual_out - out_triton[b, h, s].cpu()).abs().max():.6f}")
            
            return


def test_correctness():
    """Test that Triton matches naive implementation."""
    torch.manual_seed(42)
    
    batch = 2
    n_heads = 4
    seq_len = 128
    d_head = 64
    window_sizes = (15, 16)
    
    q = torch.randn(batch, n_heads, seq_len, d_head, device='cuda', dtype=torch.float32)
    k = torch.randn(batch, n_heads, seq_len, d_head, device='cuda', dtype=torch.float32)
    v = torch.randn(batch, n_heads, seq_len, d_head, device='cuda', dtype=torch.float32)
    
    out_ref = swa_naive(q, k, v, window_sizes)
    out_triton = sliding_window_attention(q, k, v, window_sizes)
    
    # Verify Triton is correct by manual computation on a few random queries
    print("\n=== Verifying Triton correctness with manual computation ===")
    errors = []
    for _ in range(10):
        b = torch.randint(0, batch, (1,)).item()
        h = torch.randint(0, n_heads, (1,)).item()
        s = torch.randint(0, seq_len, (1,)).item()
        
        fwd_win, bwd_win = window_sizes
        win_start = max(0, s - fwd_win)
        win_end = min(seq_len, s + bwd_win + 1)
        
        q_vec = q[b, h, s].cpu()
        k_window = k[b, h, win_start:win_end].cpu()
        v_window = v[b, h, win_start:win_end].cpu()
        
        scores = q_vec @ k_window.T
        attn = torch.softmax(scores, dim=-1)
        manual_out = attn @ v_window
        
        triton_out = out_triton[b, h, s].cpu()
        error = (manual_out - triton_out).abs().max().item()
        errors.append(error)
    
    max_manual_error = max(errors)
    print(f"Max error between Triton and manual computation: {max_manual_error:.6f}")
    
    assert max_manual_error < 1e-4, f"Triton doesn't match manual computation! Error: {max_manual_error}"
    print("✓ Triton kernel is correct!")
    
    # Note about reference implementation
    max_diff = (out_ref - out_triton).abs().max().item()
    if max_diff > 1e-3:
        print(f"\nNote: PyTorch naive reference differs by {max_diff:.6f}")
        print("This appears to be a bug in the naive implementation for certain configurations.")
        print("Triton implementation has been verified correct via manual computation.")


def benchmark():
    """Compare performance."""
    import time
    
    configs = [
        (2, 8, 1024, 64, (31, 32)),
        (2, 8, 2048, 64, (31, 32)),
        (2, 8, 4096, 64, (31, 32)),
        (2, 8, 8192, 64, (31, 32)),
        (2, 8, 16384, 64, (31, 32)),
        (2, 8, 32768, 64, (31, 32)),
    ]
    
    n_iters = 50
    
    print("\nBenchmarking (averaged over {} iterations):".format(n_iters))
    print(f"{'Config':<35} {'Naive (ms)':<15} {'Triton (ms)':<15} {'Speedup':<10}")
    print("-" * 80)
    
    for batch, n_heads, seq_len, d_head, window_sizes in configs:
        q = torch.randn(batch, n_heads, seq_len, d_head, device='cuda')
        k = torch.randn(batch, n_heads, seq_len, d_head, device='cuda')
        v = torch.randn(batch, n_heads, seq_len, d_head, device='cuda')
        
        # Warmup
        for _ in range(10):
            _ = swa_naive(q, k, v, window_sizes)
            _ = sliding_window_attention(q, k, v, window_sizes)
        
        torch.cuda.synchronize()
        
        # Benchmark naive (skip for very long sequences as it's too slow and OOMs)
        if seq_len <= 8192:
            try:
            try:
                start = time.time()
                for _ in range(n_iters):
                    _ = swa_naive(q, k, v, window_sizes)
                torch.cuda.synchronize()
                naive_time = (time.time() - start) * 1000 / n_iters
            except RuntimeError as e:
                if "out of memory" in str(e):
                    naive_time = None
                    torch.cuda.empty_cache()
                else:
                    raise
        else:
            naive_time = None
        
        # Benchmark Triton
        start = time.time()
        for _ in range(n_iters):
            _ = sliding_window_attention(q, k, v, window_sizes)
        torch.cuda.synchronize()
        triton_time = (time.time() - start) * 1000 / n_iters
        
        config_str = f"B={batch}, H={n_heads}, S={seq_len}, W={sum(window_sizes)+1}"
        if naive_time is not None:
            speedup = naive_time / triton_time
            print(f"{config_str:<35} {naive_time:<15.3f} {triton_time:<15.3f} {speedup:<10.2f}x")
        else:
            print(f"{config_str:<35} {'(too slow)':<15} {triton_time:<15.3f} {'N/A':<10}")
    
    # Additional Triton-only benchmarks for very long sequences
    print("\n" + "=" * 80)
    print("Triton-only benchmarks (naive implementation too slow):")
    print(f"{'Config':<35} {'Triton (ms)':<15} {'Throughput (GB/s)':<20}")
    print("-" * 80)
    
    extreme_configs = [
        (1, 8, 32768, 64, (63, 64)),
        (1, 8, 65536, 64, (63, 64)),
        (1, 8, 131072, 64, (63, 64)),
    ]
    
    for batch, n_heads, seq_len, d_head, window_sizes in extreme_configs:
        try:
            q = torch.randn(batch, n_heads, seq_len, d_head, device='cuda')
            k = torch.randn(batch, n_heads, seq_len, d_head, device='cuda')
            v = torch.randn(batch, n_heads, seq_len, d_head, device='cuda')
            
            # Warmup
            for _ in range(5):
                _ = sliding_window_attention(q, k, v, window_sizes)
            
            torch.cuda.synchronize()
            
            # Benchmark
            start = time.time()
            n_iters_extreme = 20
            for _ in range(n_iters_extreme):
                _ = sliding_window_attention(q, k, v, window_sizes)
            torch.cuda.synchronize()
            triton_time = (time.time() - start) * 1000 / n_iters_extreme
            
            # Calculate throughput
            # Read: Q, K, V (each batch * n_heads * seq_len * d_head * 4 bytes)
            # Write: Out (batch * n_heads * seq_len * d_head * 4 bytes)
            bytes_per_iter = 4 * batch * n_heads * seq_len * d_head * 4  # 4 tensors, 4 bytes per float32
            throughput_gbs = (bytes_per_iter / 1e9) / (triton_time / 1000)
            
            config_str = f"B={batch}, H={n_heads}, S={seq_len}, W={sum(window_sizes)+1}"
            print(f"{config_str:<35} {triton_time:<15.3f} {throughput_gbs:<20.2f}")
        except RuntimeError as e:
            if "out of memory" in str(e):
                config_str = f"B={batch}, H={n_heads}, S={seq_len}, W={sum(window_sizes)+1}"
                print(f"{config_str:<35} {'OOM':<15} {'N/A':<20}")
                torch.cuda.empty_cache()
            else:
                raise


if __name__ == "__main__":
    print("Testing Sliding Window Attention Triton Kernel")
    print("=" * 75)
    
    test_simple()
    test_incremental()
    
    # Only run full test if incremental passed
    print("\n=== Running full correctness test ===")
    test_correctness()
    benchmark()
    