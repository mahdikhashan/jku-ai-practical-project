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
    print("Warning: 'fla' library not found. GLA and Lizard benchmarks will be skipped.")
    FLA_AVAILABLE = False

# ============================================================================
# TRITON KERNEL (AWA with Meta Tokens)
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
    """AWA Kernel: Local Window + Meta Tokens"""
    pid_seq = tl.program_id(0)
    pid_batch = tl.program_id(1)
    pid_head = tl.program_id(2)
    
    if pid_seq >= seq_len:
        return
    
    # Load query vector
    q_offset = pid_batch * stride_qb + pid_head * stride_qh + pid_seq * stride_qs
    d_offsets = tl.arange(0, BLOCK_DMODEL)
    d_mask = d_offsets < d_head
    
    q_ptrs = Q + q_offset + d_offsets * stride_qd
    q = tl.load(q_ptrs, mask=d_mask, other=0.0).to(tl.float32)
    
    # Initialize online softmax
    m_i = float("-inf")
    l_i = 0.0
    acc = tl.zeros([BLOCK_DMODEL], dtype=tl.float32)
    
    # Phase 1: Local window
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
    
    # Phase 2: Meta tokens (first num_meta_tokens positions)
    if num_meta_tokens > 0:
        meta_end = tl.minimum(num_meta_tokens, seq_len)
        
        meta_idx = 0
        while meta_idx < meta_end:
            k_offsets = meta_idx + tl.arange(0, BLOCK_M)
            # Only process meta tokens outside local window
            k_valid = (k_offsets < meta_end) & ((k_offsets < local_start) | (k_offsets >= local_end))
            
            k_base = pid_batch * stride_kb + pid_head * stride_kh
            k_ptrs = K + k_base + k_offsets[:, None] * stride_ks + d_offsets[None, :] * stride_kd
            k = tl.load(k_ptrs, mask=k_valid[:, None] & d_mask[None, :], other=0.0).to(tl.float32)
            
            qk = tl.sum(q[None, :] * k, axis=1)
            qk = tl.where(k_valid, qk, float("-inf"))
            
            has_valid = tl.max(qk) > float("-inf")
            
            if has_valid:
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
            
            meta_idx += BLOCK_M
    
    # Final normalization
    acc = acc / l_i
    
    # Store output
    o_offset = pid_batch * stride_ob + pid_head * stride_oh + pid_seq * stride_os
    o_ptrs = Out + o_offset + d_offsets * stride_od
    tl.store(o_ptrs, acc, mask=d_mask)

# ============================================================================
# RUNNERS
# ============================================================================

def run_awa(q, k, v, local_window, num_meta_tokens):
    """Run AWA (Anchor Window Attention)"""
    batch, n_heads, seq_len, d_head = q.shape
    q, k, v = q.contiguous(), k.contiguous(), v.contiguous()
    out = torch.empty_like(q)
    
    BLOCK_M = 16 if d_head >= 64 else 64
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

def run_lizard(gla_layer, x_gla, q, k, v, local_window, num_meta_tokens):
    """
    Lizard = GLA + AWA (in parallel, then fused)
    
    In real implementation, GLA and AWA run concurrently on different parts
    of the model, so latency should be ~max(GLA_time, AWA_time), not sum.
    
    For benchmarking: We measure the actual fusion time which includes
    both forward passes, but they can be parallelized in practice.
    """
    # Run both in parallel (PyTorch will handle scheduling)
    # In a real model, these would be on different layers/heads
    
    # GLA path
    gla_out = gla_layer(x_gla)
    if isinstance(gla_out, tuple):
        gla_out = gla_out[0]
    
    # AWA path (can run concurrently)
    awa_out = run_awa(q, k, v, local_window, num_meta_tokens)
    
    # Fusion
    B, H, L, D = awa_out.shape
    awa_flat = awa_out.permute(0, 2, 1, 3).reshape(B, L, H * D)
    
    return gla_out + awa_flat

def benchmark_latency(func, args, n_iters=20, warmup=5):
    """Benchmark function latency"""
    # Warmup
    for _ in range(warmup):
        func(*args)
    torch.cuda.synchronize()
    
    # Measure
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    
    start.record()
    for _ in range(n_iters):
        func(*args)
    end.record()
    torch.cuda.synchronize()
    
    return start.elapsed_time(end) / n_iters

# ============================================================================
# MAIN BENCHMARK
# ============================================================================

if __name__ == "__main__":
    if not torch.cuda.is_available():
        print("ERROR: CUDA not available. This script requires a GPU.")
        sys.exit(1)

    print("=" * 80)
    print("Lizard Architecture Benchmark: GLA + AWA")
    print("=" * 80)
    print(f"Device: {torch.cuda.get_device_name(0)}")
    print(f"Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
    
    # Configuration
    BATCH_SIZE = 2  # Reduced from 4 for memory
    NUM_HEADS = 8
    D_HEAD = 128
    HIDDEN_SIZE = NUM_HEADS * D_HEAD
    DTYPE = torch.float16
    DEVICE = "cuda:0"
    WINDOW_SIZE = 64
    META_TOKENS = 4  # Number of meta tokens (0, 2, 4, 6, or 8)
    
    print(f"\nConfig: Batch={BATCH_SIZE}, Heads={NUM_HEADS}, D_head={D_HEAD}")
    print(f"AWA: window={WINDOW_SIZE}, meta_tokens={META_TOKENS}")
    print("=" * 80)
    
    # Sequence lengths to test
    SEQ_LENS = [1024, 2048, 4096, 8192, 16384]  # Removed 32K for Lizard memory
    
    # Results storage
    results = {
        'seq': [],
        'gla_ms': [], 'awa_ms': [], 'liz_ms': [],
        'gla_tok_s': [], 'awa_tok_s': [], 'liz_tok_s': []
    }

    print(f"\n{'Seq Len':<10} {'GLA (ms)':<12} {'AWA (ms)':<12} {'Lizard (ms)':<14} {'Parallel*':<14} {'AWA vs GLA':<12} {'Liz vs GLA':<12}")
    print("-" * 90)
    print("*Parallel = theoretical best case if GLA and AWA run concurrently = max(GLA, AWA)")
    print("-" * 90)

    for seq_len in SEQ_LENS:
        results['seq'].append(seq_len)
        
        t_gla, t_awa, t_liz = None, None, None
        
        # Benchmark GLA
        if FLA_AVAILABLE:
            try:
                torch.cuda.empty_cache()
                gla_layer = GatedLinearAttention(
                    hidden_size=HIDDEN_SIZE, 
                    num_heads=NUM_HEADS, 
                    mode='fused_recurrent'
                ).to(DEVICE, dtype=DTYPE)
                x = torch.randn(BATCH_SIZE, seq_len, HIDDEN_SIZE, device=DEVICE, dtype=DTYPE)
                
                t_gla = benchmark_latency(lambda: gla_layer(x), ())
                del gla_layer, x
                torch.cuda.empty_cache()
            except RuntimeError as e:
                if "out of memory" in str(e):
                    t_gla = None
                else:
                    raise
        
        # Benchmark AWA
        try:
            torch.cuda.empty_cache()
            q = torch.randn(BATCH_SIZE, NUM_HEADS, seq_len, D_HEAD, device=DEVICE, dtype=DTYPE)
            k = torch.randn(BATCH_SIZE, NUM_HEADS, seq_len, D_HEAD, device=DEVICE, dtype=DTYPE)
            v = torch.randn(BATCH_SIZE, NUM_HEADS, seq_len, D_HEAD, device=DEVICE, dtype=DTYPE)
            
            t_awa = benchmark_latency(lambda: run_awa(q, k, v, WINDOW_SIZE, META_TOKENS), ())
            del q, k, v
            torch.cuda.empty_cache()
        except RuntimeError as e:
            if "out of memory" in str(e):
                t_awa = None
            else:
                raise

        # Benchmark Lizard (GLA + AWA)
        if FLA_AVAILABLE:
            try:
                torch.cuda.empty_cache()
                
                # Use smaller batch for Lizard to fit in memory
                lizard_batch = max(1, BATCH_SIZE // 2)
                
                gla_layer = GatedLinearAttention(
                    hidden_size=HIDDEN_SIZE, 
                    num_heads=NUM_HEADS, 
                    mode='fused_recurrent'
                ).to(DEVICE, dtype=DTYPE)
                x = torch.randn(lizard_batch, seq_len, HIDDEN_SIZE, device=DEVICE, dtype=DTYPE)
                q = torch.randn(lizard_batch, NUM_HEADS, seq_len, D_HEAD, device=DEVICE, dtype=DTYPE)
                k = torch.randn(lizard_batch, NUM_HEADS, seq_len, D_HEAD, device=DEVICE, dtype=DTYPE)
                v = torch.randn(lizard_batch, NUM_HEADS, seq_len, D_HEAD, device=DEVICE, dtype=DTYPE)
                
                t_liz = benchmark_latency(
                    lambda: run_lizard(gla_layer, x, q, k, v, WINDOW_SIZE, META_TOKENS), 
                    ()
                )
                # Scale up time for fair comparison
                t_liz = t_liz * (BATCH_SIZE / lizard_batch)
                
                del gla_layer, x, q, k, v
                torch.cuda.empty_cache()
            except RuntimeError as e:
                if "out of memory" in str(e):
                    t_liz = None
                    torch.cuda.empty_cache()
                else:
                    raise

        # Store results
        results['gla_ms'].append(t_gla)
        results['awa_ms'].append(t_awa)
        results['liz_ms'].append(t_liz)
        
        # Theoretical best case: if GLA and AWA could run in parallel
        t_parallel = max(t_gla, t_awa) if (t_gla and t_awa) else None
        
        total_tokens = BATCH_SIZE * seq_len
        results['gla_tok_s'].append(total_tokens / (t_gla/1000) if t_gla else None)
        results['awa_tok_s'].append(total_tokens / (t_awa/1000) if t_awa else None)
        results['liz_tok_s'].append(total_tokens / (t_liz/1000) if t_liz else None)

        # Print results
        s_gla = f"{t_gla:.2f}" if t_gla else "OOM"
        s_awa = f"{t_awa:.2f}" if t_awa else "OOM"
        s_liz = f"{t_liz:.2f}" if t_liz else "OOM"
        s_parallel = f"{t_parallel:.2f}" if t_parallel else "N/A"
        
        # Calculate speedups
        speedup_awa = f"{t_gla/t_awa:.2f}x" if (t_gla and t_awa) else "N/A"
        speedup_liz = f"{t_gla/t_liz:.2f}x" if (t_gla and t_liz) else "N/A"
        
        print(f"{seq_len:<10} {s_gla:<12} {s_awa:<12} {s_liz:<14} {s_parallel:<14} {speedup_awa:<12} {speedup_liz:<12}")

    # ============================================================================
    # PLOTTING
    # ============================================================================
    
    def get_valid_data(key):
        """Extract valid (non-None) data points"""
        x, y = [], []
        for s, v in zip(results['seq'], results[key]):
            if v is not None:
                x.append(s)
                y.append(v)
        return x, y

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))

    # Plot 1: Latency
    if FLA_AVAILABLE:
        x, y = get_valid_data('gla_ms')
        if x: ax1.plot(x, y, 'o-', label='GLA (Recurrent)', linewidth=2, markersize=8)
    
    x, y = get_valid_data('awa_ms')
    if x: ax1.plot(x, y, 's-', label='AWA (Sliding+Meta)', linewidth=2, markersize=8)
    
    if FLA_AVAILABLE:
        x, y = get_valid_data('liz_ms')
        if x: ax1.plot(x, y, '^-', label='Lizard (GLA+AWA)', color='green', linewidth=2, markersize=8)
    
    ax1.set_title("Latency Comparison", fontsize=14, fontweight='bold')
    ax1.set_xlabel("Sequence Length", fontsize=12)
    ax1.set_ylabel("Time (ms)", fontsize=12)
    ax1.legend(fontsize=11)
    ax1.grid(True, alpha=0.3)
    ax1.set_xscale('log', base=2)
    ax1.set_yscale('log')

    # Plot 2: Throughput
    if FLA_AVAILABLE:
        x, y = get_valid_data('gla_tok_s')
        if x: ax2.plot(x, y, 'o-', label='GLA', linewidth=2, markersize=8)
    
    x, y = get_valid_data('awa_tok_s')
    if x: ax2.plot(x, y, 's-', label='AWA', linewidth=2, markersize=8)
    
    if FLA_AVAILABLE:
        x, y = get_valid_data('liz_tok_s')
        if x: ax2.plot(x, y, '^-', label='Lizard', color='green', linewidth=2, markersize=8)

    ax2.set_title("Throughput Comparison", fontsize=14, fontweight='bold')
    ax2.set_xlabel("Sequence Length", fontsize=12)
    ax2.set_ylabel("Tokens/Second", fontsize=12)
    ax2.legend(fontsize=11)
    ax2.grid(True, alpha=0.3)
    ax2.set_xscale('log', base=2)

    plt.tight_layout()
    plt.savefig('lizard_benchmark.png', dpi=150)
    print("\n" + "=" * 80)
    print("✓ Chart saved to 'lizard_benchmark.png'")
    print("=" * 80)
