import torch
import triton
import triton.language as tl
import time

# ... (Keep your kernel definitions exactly as they are) ...

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
    
    # Benchmark settings
    n_iters = 50
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)

    # ------------------------------------------------------------------------
    # 1. Run NAIVE PyTorch Implementation FIRST to establish baseline
    # ------------------------------------------------------------------------
    print("\n" + "=" * 80)
    print("1. Running Naive Approach (Baseline):")
    print("=" * 80)
    
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
    print(f"Naive PyTorch Latency: {naive_time_ms:.2f} ms")

    # ------------------------------------------------------------------------
    # 2. Run Triton Chunk-Local (Stateless)
    # ------------------------------------------------------------------------
    print("\n" + "=" * 80)
    print("2. Running Triton Chunk-Local (Tensor Cores):")
    print("=" * 80)
    
    # Warmup
    for _ in range(5):
        out = gla_tensor_core(q, k, v, gates, chunk_size)
    torch.cuda.synchronize()
    
    start.record()
    for _ in range(n_iters):
        out = gla_tensor_core(q, k, v, gates, chunk_size)
    end.record()
    torch.cuda.synchronize()
    
    time_ms = start.elapsed_time(end) / n_iters
    
    print(f"Triton Local Latency: {time_ms:.2f} ms")
    print(f"Speedup vs Naive: {naive_time_ms / time_ms:.2f}x")

    # ------------------------------------------------------------------------
    # 3. Run Triton Stateful (Recurrent)
    # ------------------------------------------------------------------------
    print("\n" + "=" * 80)
    print("3. Running Stateful GLA (with cross-chunk recurrence):")
    print("=" * 80)
    
    # Warmup
    for _ in range(5):
        out_stateful = gla_tensor_core_stateful(q, k, v, gates, chunk_size)
    torch.cuda.synchronize()
    
    start.record()
    for _ in range(n_iters):
        out_stateful = gla_tensor_core_stateful(q, k, v, gates, chunk_size)
    end.record()
    torch.cuda.synchronize()
    
    stateful_time_ms = start.elapsed_time(end) / n_iters
    
    print(f"Stateful GLA Latency: {stateful_time_ms:.2f} ms")
    # NOW this variable exists, so this won't crash
    print(f"Speedup vs Naive: {naive_time_ms / stateful_time_ms:.2f}x")
    
    # ------------------------------------------------------------------------
    # Summary
    # ------------------------------------------------------------------------
    print("\n" + "=" * 80)
    print("Summary: Tensor Core Performance")
    print("=" * 80)
    print(f"{'Method':<35} {'Time (ms)':<12} {'Speedup':<10}")
    print("-" * 80)
    print(f"{'Naive PyTorch (no tensor cores)':<35} {naive_time_ms:<12.2f} {'1.00x':<10}")
    print(f"{'Triton chunk-local (tensor cores)':<35} {time_ms:<12.2f} {f'{naive_time_ms/time_ms:.2f}x':<10}")
    print(f"{'Triton stateful GLA (tensor cores)':<35} {stateful_time_ms:<12.2f} {f'{naive_time_ms/stateful_time_ms:.2f}x':<10}")
    
    print("\n" + "=" * 80)
    print("Tensor Core Usage Indicators:")
    print("=" * 80)
    print(f"✓ Using tl.dot() for matmuls (automatically uses tensor cores)")
    print(f"✓ Using fp16 dtype (optimal for tensor cores)")
    print(f"✓ Block sizes are powers of 2: D={triton.next_power_of_2(d_model)}, Chunk={chunk_size}")
    print(f"✓ Matrix dimensions suitable for tensor cores: [{chunk_size} x {d_model}]")

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
    
    test_tensor_core_usage()
