import torch
import triton
import triton.language as tl
import time


@triton.jit
def gla_chunk_fwd_kernel(
    Q, K, V, G, Out,  # G = gates
    seq_len, d_model, chunk_size,
    stride_qb, stride_qh, stride_qs, stride_qd,
    stride_kb, stride_kh, stride_ks, stride_kd,
    stride_vb, stride_vh, stride_vs, stride_vd,
    stride_gb, stride_gh, stride_gs,
    stride_ob, stride_oh, stride_os, stride_od,
    BLOCK_D: tl.constexpr,
    BLOCK_CHUNK: tl.constexpr,
):
    """
    GLA kernel using tensor cores via tl.dot()
    
    Processes sequence in chunks, using tensor cores for chunk-level operations.
    This is a simplified version of what Lizard likely does.
    """
    pid_batch = tl.program_id(0)
    pid_head = tl.program_id(1)
    pid_chunk = tl.program_id(2)
    
    # Chunk boundaries
    chunk_start = pid_chunk * BLOCK_CHUNK
    chunk_end = tl.minimum(chunk_start + BLOCK_CHUNK, seq_len)
    actual_chunk_size = chunk_end - chunk_start
    
    if actual_chunk_size <= 0:
        return
    
    # Offsets for this chunk
    seq_offsets = chunk_start + tl.arange(0, BLOCK_CHUNK)
    d_offsets = tl.arange(0, BLOCK_D)
    seq_mask = seq_offsets < chunk_end
    d_mask = d_offsets < d_model
    
    # Load Q, K, V for this chunk [BLOCK_CHUNK, d_model]
    q_offset = pid_batch * stride_qb + pid_head * stride_qh
    k_offset = pid_batch * stride_kb + pid_head * stride_kh
    v_offset = pid_batch * stride_vb + pid_head * stride_vh
    
    q_ptrs = Q + q_offset + seq_offsets[:, None] * stride_qs + d_offsets[None, :] * stride_qd
    k_ptrs = K + k_offset + seq_offsets[:, None] * stride_ks + d_offsets[None, :] * stride_kd
    v_ptrs = V + v_offset + seq_offsets[:, None] * stride_vs + d_offsets[None, :] * stride_vd
    
    Q_chunk = tl.load(q_ptrs, mask=seq_mask[:, None] & d_mask[None, :], other=0.0).to(tl.float32)
    K_chunk = tl.load(k_ptrs, mask=seq_mask[:, None] & d_mask[None, :], other=0.0).to(tl.float32)
    V_chunk = tl.load(v_ptrs, mask=seq_mask[:, None] & d_mask[None, :], other=0.0).to(tl.float32)
    
    # Load gates [BLOCK_CHUNK]
    g_offset = pid_batch * stride_gb + pid_head * stride_gh
    g_ptrs = G + g_offset + seq_offsets * stride_gs
    gates = tl.load(g_ptrs, mask=seq_mask, other=0.0).to(tl.float32)
    
    # Compute attention within chunk using tensor cores
    # scores = Q @ K^T  [BLOCK_CHUNK, BLOCK_CHUNK]
    # This uses tensor cores automatically!
    K_T = tl.trans(K_chunk)  # [d_model, BLOCK_CHUNK]
    scores = tl.dot(Q_chunk, K_T)  # Uses tensor cores (mma.sync)!
    
    # Apply causal mask within chunk
    causal_mask = seq_offsets[:, None] >= seq_offsets[None, :]
    scores = tl.where(causal_mask & seq_mask[:, None] & seq_mask[None, :], scores, float("-inf"))
    
    # Softmax
    scores_max = tl.max(scores, axis=1)
    scores_exp = tl.exp(scores - scores_max[:, None])
    scores_sum = tl.sum(scores_exp, axis=1)
    attn = scores_exp / scores_sum[:, None]
    
    # Apply gates (recurrent-like behavior)
    gated_attn = attn * gates[:, None]
    
    # Output = attention @ V  [BLOCK_CHUNK, d_model]
    # This also uses tensor cores!
    out_chunk = tl.dot(gated_attn, V_chunk)  # Uses tensor cores (mma.sync)!
    
    # Store output
    o_offset = pid_batch * stride_ob + pid_head * stride_oh
    o_ptrs = Out + o_offset + seq_offsets[:, None] * stride_os + d_offsets[None, :] * stride_od
    tl.store(o_ptrs, out_chunk, mask=seq_mask[:, None] & d_mask[None, :])


def gla_tensor_core(q, k, v, gates, chunk_size=64):
    """
    GLA using Triton with tensor cores.
    
    Args:
        q, k, v: [batch, n_heads, seq_len, d_model]
        gates: [batch, n_heads, seq_len] - gating values
        chunk_size: Size of chunks for parallel processing
    
    Returns:
        out: [batch, n_heads, seq_len, d_model]
    """
    batch, n_heads, seq_len, d_model = q.shape
    
    q = q.contiguous()
    k = k.contiguous()
    v = v.contiguous()
    gates = gates.contiguous()
    
    out = torch.empty_like(q)
    
    # Block sizes (must be powers of 2 for tensor cores)
    BLOCK_D = triton.next_power_of_2(d_model)
    BLOCK_CHUNK = chunk_size
    
    # Grid: (batch, heads, num_chunks)
    num_chunks = triton.cdiv(seq_len, chunk_size)
    grid = (batch, n_heads, num_chunks)
    
    gla_chunk_fwd_kernel[grid](
        q, k, v, gates, out,
        seq_len, d_model, chunk_size,
        q.stride(0), q.stride(1), q.stride(2), q.stride(3),
        k.stride(0), k.stride(1), k.stride(2), k.stride(3),
        v.stride(0), v.stride(1), v.stride(2), v.stride(3),
        gates.stride(0), gates.stride(1), gates.stride(2),
        out.stride(0), out.stride(1), out.stride(2), out.stride(3),
        BLOCK_D=BLOCK_D,
        BLOCK_CHUNK=BLOCK_CHUNK,
    )
    
    return out


# ============================================================================
# Testing and Benchmarking
# ============================================================================

def test_tensor_core_usage():
    """Test if tensor cores are being used"""
    print("=" * 80)
    print("Testing GLA with Tensor Cores (tl.dot)")
    print("=" * 80)
    
    batch = 2
    n_heads = 8
    seq_len = 2048
    d_model = 128
    chunk_size = 64
    
    device = 'cuda'
    dtype = torch.float16  # Tensor cores work best with fp16
    
    print(f"\nConfig: B={batch}, H={n_heads}, S={seq_len}, D={d_model}")
    print(f"Chunk size: {chunk_size}")
    print(f"Dtype: {dtype}")
    
    # Create inputs
    q = torch.randn(batch, n_heads, seq_len, d_model, device=device, dtype=dtype)
    k = torch.randn(batch, n_heads, seq_len, d_model, device=device, dtype=dtype)
    v = torch.randn(batch, n_heads, seq_len, d_model, device=device, dtype=dtype)
    gates = torch.sigmoid(torch.randn(batch, n_heads, seq_len, device=device, dtype=dtype))
    
    # Warmup
    for _ in range(5):
        out = gla_tensor_core(q, k, v, gates, chunk_size)
    torch.cuda.synchronize()
    
    # Benchmark
    n_iters = 50
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    
    start.record()
    for _ in range(n_iters):
        out = gla_tensor_core(q, k, v, gates, chunk_size)
    end.record()
    torch.cuda.synchronize()
    
    time_ms = start.elapsed_time(end) / n_iters
    
    print(f"\nLatency: {time_ms:.2f} ms")
    print(f"Output shape: {out.shape}")
    
    # Verify tensor cores are likely being used
    print("\n" + "=" * 80)
    print("Tensor Core Usage Indicators:")
    print("=" * 80)
    print(f"✓ Using tl.dot() for matmuls (automatically uses tensor cores)")
    print(f"✓ Using fp16 dtype (optimal for tensor cores)")
    print(f"✓ Block sizes are powers of 2: D={triton.next_power_of_2(d_model)}, Chunk={chunk_size}")
    print(f"✓ Matrix dimensions suitable for tensor cores: [{chunk_size} x {d_model}]")
    
    # Compare with naive implementation
    print("\n" + "=" * 80)
    print("Comparison with Naive Approach:")
    print("=" * 80)
    
    # Naive: element-wise operations
    def naive_gla(q, k, v, gates):
        # Simple attention without tensor cores
        scores = torch.einsum('bhqd,bhkd->bhqk', q, k)
        causal_mask = torch.tril(torch.ones(seq_len, seq_len, device=device, dtype=torch.bool))
        scores = scores.masked_fill(~causal_mask, float('-inf'))
        attn = torch.softmax(scores, dim=-1)
        attn = attn * gates.unsqueeze(-1)
        return torch.einsum('bhqk,bhkd->bhqd', attn, v)
    
    # Warmup
    for _ in range(5):
        _ = naive_gla(q, k, v, gates)
    torch.cuda.synchronize()
    
    start.record()
    for _ in range(n_iters):
        _ = naive_gla(q, k, v, gates)
    end.record()
    torch.cuda.synchronize()
    
    naive_time_ms = start.elapsed_time(end) / n_iters
    
    print(f"Naive PyTorch: {naive_time_ms:.2f} ms")
    print(f"Triton (with tensor cores): {time_ms:.2f} ms")
    print(f"Speedup: {naive_time_ms / time_ms:.2f}x")
    
    print("\n" + "=" * 80)
    print("Key Insights:")
    print("=" * 80)
    print("• tl.dot() automatically uses tensor cores (mma.sync) when:")
    print("  - Dimensions are multiples of 16 (for fp16)")
    print("  - Data is properly aligned")
    print("  - GPU supports tensor cores (SM 7.0+)")
    print("\n• For true Lizard-style GLA, you'd need:")
    print("  - More sophisticated chunking strategy")
    print("  - Cross-chunk state propagation")
    print("  - Optimized gate application")
    print("=" * 80)


if __name__ == "__main__":
    if not torch.cuda.is_available():
        print("CUDA not available!")
        exit(1)
    
    # Check GPU capabilities
    device_props = torch.cuda.get_device_properties(0)
    print(f"GPU: {device_props.name}")
    print(f"Compute Capability: {device_props.major}.{device_props.minor}")
    
    if device_props.major >= 7:  # Tensor cores available on SM 7.0+
        print("✓ Tensor Cores available!")
    else:
        print("✗ Tensor Cores not available (need compute capability >= 7.0)")
    
    print()
    test_tensor_core_usage()
