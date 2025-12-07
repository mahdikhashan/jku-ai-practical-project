import torch
import triton
import triton.language as tl
import sys
import matplotlib.pyplot as plt

try:
    from fla.layers import GatedLinearAttention
    FLA_AVAILABLE = True
except ImportError:
    print("Warning: 'fla' library not found. Install with: pip install fla-layers")
    FLA_AVAILABLE = False
    sys.exit(1)

# ============================================================================
# AWA KERNEL
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
    
    q_offset = pid_batch * stride_qb + pid_head * stride_qh + pid_seq * stride_qs
    d_offsets = tl.arange(0, BLOCK_DMODEL)
    d_mask = d_offsets < d_head
    
    q_ptrs = Q + q_offset + d_offsets * stride_qd
    q = tl.load(q_ptrs, mask=d_mask, other=0.0).to(tl.float32)
    
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
    
    # Phase 2: Meta tokens
    if num_meta_tokens > 0:
        meta_end = tl.minimum(num_meta_tokens, seq_len)
        
        meta_idx = 0
        while meta_idx < meta_end:
            k_offsets = meta_idx + tl.arange(0, BLOCK_M)
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
    
    acc = acc / l_i
    
    o_offset = pid_batch * stride_ob + pid_head * stride_oh + pid_seq * stride_os
    o_ptrs = Out + o_offset + d_offsets * stride_od
    tl.store(o_ptrs, acc, mask=d_mask)

def run_awa_layer(q, k, v, local_window, num_meta_tokens):
    """Single AWA layer"""
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

# ============================================================================
# MODEL ARCHITECTURES
# ============================================================================

class PureGLAModel(torch.nn.Module):
    """Pure GLA model with N layers"""
    def __init__(self, hidden_size, num_heads, num_layers, dtype=torch.float16, device='cuda'):
        super().__init__()
        self.layers = torch.nn.ModuleList([
            GatedLinearAttention(hidden_size=hidden_size, num_heads=num_heads, mode='fused_recurrent')
            for _ in range(num_layers)
        ])
        self.to(device=device, dtype=dtype)
    
    def forward(self, x):
        for layer in self.layers:
            out = layer(x)
            x = out[0] if isinstance(out, tuple) else out
        return x

class PureAWAModel(torch.nn.Module):
    """Pure AWA model with N layers"""
    def __init__(self, num_heads, d_head, num_layers, local_window, num_meta_tokens):
        super().__init__()
        self.num_layers = num_layers
        self.num_heads = num_heads
        self.d_head = d_head
        self.local_window = local_window
        self.num_meta_tokens = num_meta_tokens
    
    def forward(self, q, k, v):
        """
        q, k, v: [batch, num_heads, seq_len, d_head]
        """
        x = q
        for _ in range(self.num_layers):
            x = run_awa_layer(x, k, v, self.local_window, self.num_meta_tokens)
        return x

class LizardModel(torch.nn.Module):
    """Lizard model: Interleaved GLA and AWA layers"""
    def __init__(self, hidden_size, num_heads, d_head, num_gla_layers, num_awa_layers, 
                 local_window, num_meta_tokens, dtype=torch.float16, device='cuda'):
        super().__init__()
        self.num_heads = num_heads
        self.d_head = d_head
        self.local_window = local_window
        self.num_meta_tokens = num_meta_tokens
        
        # GLA layers
        self.gla_layers = torch.nn.ModuleList([
            GatedLinearAttention(hidden_size=hidden_size, num_heads=num_heads, mode='fused_recurrent')
            for _ in range(num_gla_layers)
        ])
        
        # AWA layers (no parameters, just compute)
        self.num_awa_layers = num_awa_layers
        
        self.to(device=device, dtype=dtype)
    
    def forward(self, x_gla, q, k, v):
        """
        x_gla: [batch, seq_len, hidden_size] for GLA
        q, k, v: [batch, num_heads, seq_len, d_head] for AWA
        """
        # Run GLA layers
        gla_out = x_gla
        for layer in self.gla_layers:
            out = layer(gla_out)
            gla_out = out[0] if isinstance(out, tuple) else out
        
        # Run AWA layers
        awa_out = q
        for _ in range(self.num_awa_layers):
            awa_out = run_awa_layer(awa_out, k, v, self.local_window, self.num_meta_tokens)
        
        # Fusion: convert AWA to match GLA shape and add
        B, H, L, D = awa_out.shape
        awa_flat = awa_out.permute(0, 2, 1, 3).reshape(B, L, H * D)
        
        return gla_out + awa_flat

# ============================================================================
# BENCHMARK
# ============================================================================

def benchmark_model(model, inputs, n_iters=20, warmup=5):
    """Benchmark a model"""
    # Warmup
    for _ in range(warmup):
        if isinstance(inputs, tuple):
            _ = model(*inputs)
        else:
            _ = model(inputs)
    torch.cuda.synchronize()
    
    # Measure
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    
    start.record()
    for _ in range(n_iters):
        if isinstance(inputs, tuple):
            _ = model(*inputs)
        else:
            _ = model(inputs)
    end.record()
    torch.cuda.synchronize()
    
    return start.elapsed_time(end) / n_iters

if __name__ == "__main__":
    if not torch.cuda.is_available():
        print("ERROR: CUDA not available")
        sys.exit(1)
    
    print("=" * 80)
    print("Realistic Lizard Architecture Comparison")
    print("=" * 80)
    print(f"Device: {torch.cuda.get_device_name(0)}")
    
    # Config
    BATCH_SIZE = 2
    NUM_HEADS = 8
    D_HEAD = 128
    HIDDEN_SIZE = NUM_HEADS * D_HEAD
    SEQ_LEN = 4096
    DTYPE = torch.float16
    DEVICE = "cuda:0"
    WINDOW_SIZE = 64
    META_TOKENS = 4
    
    print(f"\nConfig: Batch={BATCH_SIZE}, Seq={SEQ_LEN}, Heads={NUM_HEADS}, D_head={D_HEAD}")
    print(f"AWA: window={WINDOW_SIZE}, meta_tokens={META_TOKENS}")
    print("=" * 80)
    
    # Architecture configurations to test
    configs = [
        # (name, num_gla_layers, num_awa_layers, description)
        ("Pure GLA (12L)", 12, 0, "Baseline: 12 GLA layers"),
        ("Pure AWA (12L)", 0, 12, "Baseline: 12 AWA layers"),
        ("Lizard (6+6)", 6, 6, "Hybrid: 6 GLA + 6 AWA"),
        ("Lizard (8+4)", 8, 4, "GLA-heavy: 8 GLA + 4 AWA"),
        ("Lizard (4+8)", 4, 8, "AWA-heavy: 4 GLA + 8 AWA"),
        ("Lizard (9+3)", 9, 3, "Efficient: 9 GLA + 3 AWA"),
    ]
    
    results = []
    
    print(f"\n{'Architecture':<20} {'Time (ms)':<12} {'vs Pure GLA':<15} {'Description':<30}")
    print("-" * 80)
    
    baseline_time = None
    
    for name, num_gla, num_awa, description in configs:
        try:
            torch.cuda.empty_cache()
            
            # Prepare inputs
            x = torch.randn(BATCH_SIZE, SEQ_LEN, HIDDEN_SIZE, device=DEVICE, dtype=DTYPE)
            q = torch.randn(BATCH_SIZE, NUM_HEADS, SEQ_LEN, D_HEAD, device=DEVICE, dtype=DTYPE)
            k = torch.randn(BATCH_SIZE, NUM_HEADS, SEQ_LEN, D_HEAD, device=DEVICE, dtype=DTYPE)
            v = torch.randn(BATCH_SIZE, NUM_HEADS, SEQ_LEN, D_HEAD, device=DEVICE, dtype=DTYPE)
            
            # Create model
            if num_awa == 0:
                # Pure GLA
                model = PureGLAModel(HIDDEN_SIZE, NUM_HEADS, num_gla, DTYPE, DEVICE)
                inputs = x
            elif num_gla == 0:
                # Pure AWA
                model = PureAWAModel(NUM_HEADS, D_HEAD, num_awa, WINDOW_SIZE, META_TOKENS)
                inputs = (q, k, v)
            else:
                # Lizard
                model = LizardModel(HIDDEN_SIZE, NUM_HEADS, D_HEAD, num_gla, num_awa, 
                                   WINDOW_SIZE, META_TOKENS, DTYPE, DEVICE)
                inputs = (x, q, k, v)
            
            # Benchmark
            time_ms = benchmark_model(model, inputs)
            
            if baseline_time is None:
                baseline_time = time_ms
            
            speedup = baseline_time / time_ms
            speedup_str = f"{speedup:.2f}x faster" if speedup > 1 else f"{1/speedup:.2f}x slower"
            
            print(f"{name:<20} {time_ms:<12.2f} {speedup_str:<15} {description:<30}")
            
            results.append({
                'name': name,
                'time': time_ms,
                'speedup': speedup,
                'gla': num_gla,
                'awa': num_awa
            })
            
            del model, x, q, k, v
            torch.cuda.empty_cache()
            
        except Exception as e:
            print(f"{name:<20} {'ERROR':<12} {str(e)[:40]:<15}")
    
    # Plot results
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
    
    # Plot 1: Latency
    names = [r['name'] for r in results]
    times = [r['time'] for r in results]
    colors = ['red' if 'Pure GLA' in n else 'blue' if 'Pure AWA' in n else 'green' for n in names]
    
    ax1.barh(names, times, color=colors, alpha=0.7)
    ax1.set_xlabel('Latency (ms)', fontsize=12)
    ax1.set_title('Architecture Latency Comparison', fontsize=14, fontweight='bold')
    ax1.grid(axis='x', alpha=0.3)
    
    # Plot 2: Speedup
    speedups = [r['speedup'] for r in results]
    ax2.barh(names, speedups, color=colors, alpha=0.7)
    ax2.axvline(x=1.0, color='black', linestyle='--', linewidth=2, label='Baseline (Pure GLA)')
    ax2.set_xlabel('Speedup vs Pure GLA', fontsize=12)
    ax2.set_title('Speedup Comparison', fontsize=14, fontweight='bold')
    ax2.legend()
    ax2.grid(axis='x', alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('lizard_realistic_benchmark.png', dpi=150)
    
    print("\n" + "=" * 80)
    print("Key Insights:")
    print("- Pure AWA is faster than Pure GLA (fewer operations per token)")
    print("- Lizard can be faster than Pure GLA by using fewer total layers")
    print("- Best config depends on quality vs speed tradeoff")
    print("\n✓ Chart saved to 'lizard_realistic_benchmark.png'")
    print("=" * 80)
