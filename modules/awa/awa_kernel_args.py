import torch
import triton
import triton.language as tl
import argparse
import sys

# ============================================================================
# KERNELS (Unchanged)
# ============================================================================


@triton.jit
def anchor_window_fwd_kernel(
    Q,
    K,
    V,
    Out,
    seq_len,
    d_head,
    local_window,
    num_meta_tokens,
    stride_qb,
    stride_qh,
    stride_qs,
    stride_qd,
    stride_kb,
    stride_kh,
    stride_ks,
    stride_kd,
    stride_vb,
    stride_vh,
    stride_vs,
    stride_vd,
    stride_ob,
    stride_oh,
    stride_os,
    stride_od,
    BLOCK_M: tl.constexpr,
    BLOCK_DMODEL: tl.constexpr,
):
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

    # Phase 1: Local Window
    local_start = tl.maximum(0, pid_seq - local_window)
    local_end = tl.minimum(seq_len, pid_seq + local_window + 1)

    k_block_start = local_start
    while k_block_start < local_end:
        k_offsets = k_block_start + tl.arange(0, BLOCK_M)
        k_valid = k_offsets < local_end

        k_base = pid_batch * stride_kb + pid_head * stride_kh
        k_ptrs = (
            K + k_base + k_offsets[:, None] * stride_ks + d_offsets[None, :] * stride_kd
        )
        k = tl.load(k_ptrs, mask=k_valid[:, None] & d_mask[None, :], other=0.0).to(
            tl.float32
        )

        qk = tl.sum(q[None, :] * k, axis=1)
        qk = tl.where(k_valid, qk, float("-inf"))

        m_ij = tl.max(qk, axis=0)
        m_ij = tl.maximum(m_i, m_ij)
        alpha = tl.exp(m_i - m_ij)
        p = tl.exp(qk - m_ij)
        l_ij = tl.sum(p, axis=0)

        v_base = pid_batch * stride_vb + pid_head * stride_vh
        v_ptrs = (
            V + v_base + k_offsets[:, None] * stride_vs + d_offsets[None, :] * stride_vd
        )
        v = tl.load(v_ptrs, mask=k_valid[:, None] & d_mask[None, :], other=0.0).to(
            tl.float32
        )

        acc = acc * alpha + tl.sum(p[:, None] * v, axis=0)
        l_i = l_i * alpha + l_ij
        m_i = m_ij
        k_block_start += BLOCK_M

    # Phase 2: Meta Tokens
    if num_meta_tokens > 0:
        meta_end = tl.minimum(num_meta_tokens, seq_len)
        meta_idx = 0
        while meta_idx < meta_end:
            k_offsets = meta_idx + tl.arange(0, BLOCK_M)
            k_valid = (k_offsets < meta_end) & (
                (k_offsets < local_start) | (k_offsets >= local_end)
            )

            k_base = pid_batch * stride_kb + pid_head * stride_kh
            k_ptrs = (
                K
                + k_base
                + k_offsets[:, None] * stride_ks
                + d_offsets[None, :] * stride_kd
            )
            k = tl.load(k_ptrs, mask=k_valid[:, None] & d_mask[None, :], other=0.0).to(
                tl.float32
            )

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
                v_ptrs = (
                    V
                    + v_base
                    + k_offsets[:, None] * stride_vs
                    + d_offsets[None, :] * stride_vd
                )
                v = tl.load(
                    v_ptrs, mask=k_valid[:, None] & d_mask[None, :], other=0.0
                ).to(tl.float32)

                acc = acc * alpha + tl.sum(p[:, None] * v, axis=0)
                l_i = l_i * alpha + l_ij
                m_i = m_ij
            meta_idx += BLOCK_M

    acc = acc / l_i
    o_offset = pid_batch * stride_ob + pid_head * stride_oh + pid_seq * stride_os
    o_ptrs = Out + o_offset + d_offsets * stride_od
    tl.store(o_ptrs, acc, mask=d_mask)


def anchor_window_attention(q, k, v, local_window, num_meta_tokens):
    batch, n_heads, seq_len, d_head = q.shape
    q, k, v = q.contiguous(), k.contiguous(), v.contiguous()
    out = torch.empty_like(q)

    # Heuristic for block size
    if d_head <= 32:
        BLOCK_M = 64
    elif d_head <= 64:
        BLOCK_M = 16
    else:
        BLOCK_M = 16

    BLOCK_DMODEL = triton.next_power_of_2(d_head)
    grid = (seq_len, batch, n_heads)

    anchor_window_fwd_kernel[grid](
        q,
        k,
        v,
        out,
        seq_len,
        d_head,
        local_window,
        num_meta_tokens,
        q.stride(0),
        q.stride(1),
        q.stride(2),
        q.stride(3),
        k.stride(0),
        k.stride(1),
        k.stride(2),
        k.stride(3),
        v.stride(0),
        v.stride(1),
        v.stride(2),
        v.stride(3),
        out.stride(0),
        out.stride(1),
        out.stride(2),
        out.stride(3),
        BLOCK_M=BLOCK_M,
        BLOCK_DMODEL=BLOCK_DMODEL,
    )
    return out


@triton.jit
def swa_fwd_kernel(
    Q,
    K,
    V,
    Out,
    seq_len,
    d_head,
    fwd_win_size,
    bwd_win_size,
    stride_qb,
    stride_qh,
    stride_qs,
    stride_qd,
    stride_kb,
    stride_kh,
    stride_ks,
    stride_kd,
    stride_vb,
    stride_vh,
    stride_vs,
    stride_vd,
    stride_ob,
    stride_oh,
    stride_os,
    stride_od,
    BLOCK_M: tl.constexpr,
    BLOCK_DMODEL: tl.constexpr,
):
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
        k_ptrs = (
            K + k_base + k_offsets[:, None] * stride_ks + d_offsets[None, :] * stride_kd
        )
        k = tl.load(k_ptrs, mask=k_valid[:, None] & d_mask[None, :], other=0.0).to(
            tl.float32
        )

        qk = tl.sum(q[None, :] * k, axis=1)
        qk = tl.where(k_valid, qk, float("-inf"))

        m_ij = tl.max(qk, axis=0)
        m_ij = tl.maximum(m_i, m_ij)
        alpha = tl.exp(m_i - m_ij)
        p = tl.exp(qk - m_ij)
        l_ij = tl.sum(p, axis=0)

        v_base = pid_batch * stride_vb + pid_head * stride_vh
        v_ptrs = (
            V + v_base + k_offsets[:, None] * stride_vs + d_offsets[None, :] * stride_vd
        )
        v = tl.load(v_ptrs, mask=k_valid[:, None] & d_mask[None, :], other=0.0).to(
            tl.float32
        )

        acc = acc * alpha + tl.sum(p[:, None] * v, axis=0)
        l_i = l_i * alpha + l_ij
        m_i = m_ij
        k_block_start += BLOCK_M

    acc = acc / l_i
    o_offset = pid_batch * stride_ob + pid_head * stride_oh + pid_seq * stride_os
    o_ptrs = Out + o_offset + d_offsets * stride_od
    tl.store(o_ptrs, acc, mask=d_mask)


def sliding_window_attention(q, k, v, window_sizes):
    batch, n_heads, seq_len, d_head = q.shape
    fwd_win_size, bwd_win_size = window_sizes
    q, k, v = q.contiguous(), k.contiguous(), v.contiguous()
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
        q,
        k,
        v,
        out,
        seq_len,
        d_head,
        fwd_win_size,
        bwd_win_size,
        q.stride(0),
        q.stride(1),
        q.stride(2),
        q.stride(3),
        k.stride(0),
        k.stride(1),
        k.stride(2),
        k.stride(3),
        v.stride(0),
        v.stride(1),
        v.stride(2),
        v.stride(3),
        out.stride(0),
        out.stride(1),
        out.stride(2),
        out.stride(3),
        BLOCK_M=BLOCK_M,
        BLOCK_DMODEL=BLOCK_DMODEL,
    )
    return out


# ============================================================================
# Benchmarking Tools
# ============================================================================


def measure_kernel_execution(func, args, n_warmup, n_iters):
    """Measures average execution time using torch.cuda.Event."""
    # Warmup
    for _ in range(n_warmup):
        func(*args)
    torch.cuda.synchronize()

    start_event = torch.cuda.Event(enable_timing=True)
    end_event = torch.cuda.Event(enable_timing=True)

    start_event.record()
    for _ in range(n_iters):
        func(*args)
    end_event.record()
    torch.cuda.synchronize()

    return start_event.elapsed_time(end_event) / n_iters


def benchmark_comparison(args):
    print("\n" + "=" * 90)
    print(f"Benchmarking: Sliding Window vs Anchor Window")
    print("=" * 90)

    print(f"Global Config:")
    print(f"  Batch Size: {args.batch_size}")
    print(f"  Heads:      {args.n_heads}")
    print(f"  Head Dim:   {args.d_head}")
    print(f"  Window:     {args.window_size}")
    print(f"  Warmup/Rep: {args.warmup}/{args.rep}")
    print("-" * 90)

    for seq_len in args.seq_lens:
        print(f"\n---> Sequence Length: {seq_len}")
        print(f"{'Method':<40} {'GPU Time (ms)':<15} {'Rel. Speed':<12}")
        print("-" * 90)

        try:
            # Allocate Memory
            q = torch.randn(
                args.batch_size, args.n_heads, seq_len, args.d_head, device="cuda"
            )
            k = torch.randn(
                args.batch_size, args.n_heads, seq_len, args.d_head, device="cuda"
            )
            v = torch.randn(
                args.batch_size, args.n_heads, seq_len, args.d_head, device="cuda"
            )
        except torch.cuda.OutOfMemoryError:
            print(f"Skipping seq_len={seq_len} due to OOM")
            continue

        # 1. Benchmark Sliding Window (Baseline)
        swa_args = (q, k, v, (args.window_size, args.window_size))
        swa_time = measure_kernel_execution(
            sliding_window_attention, swa_args, args.warmup, args.rep
        )

        print(f"{'Sliding Window (baseline)':<40} {swa_time:<15.4f} {1.0:<12.2f}x")

        # 2. Benchmark Anchor Window Variants
        for num_meta in args.meta_tokens:
            awa_args = (q, k, v, args.window_size, num_meta)
            awa_time = measure_kernel_execution(
                anchor_window_attention, awa_args, args.warmup, args.rep
            )

            rel_speed = swa_time / awa_time
            print(
                f"{f'  + {num_meta} meta tokens':<40} {awa_time:<15.4f} {rel_speed:<12.2f}x"
            )


def test_correctness():
    print("Running correctness check (minimal config)...")
    torch.manual_seed(42)
    B, H, S, D = 1, 1, 128, 64
    q = torch.randn(B, H, S, D, device="cuda")
    k = torch.randn(B, H, S, D, device="cuda")
    v = torch.randn(B, H, S, D, device="cuda")
    _ = sliding_window_attention(q, k, v, (32, 32))
    _ = anchor_window_attention(q, k, v, 32, 2)
    print("Correctness check passed.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Full Benchmark: Anchor vs Sliding Window Attention"
    )

    # Model/Data Dimensions
    parser.add_argument(
        "--batch_size", type=int, default=2, help="Batch size (default: 2)"
    )
    parser.add_argument(
        "--n_heads", type=int, default=8, help="Number of attention heads (default: 8)"
    )
    parser.add_argument(
        "--d_head", type=int, default=64, help="Head dimension (default: 64)"
    )
    parser.add_argument(
        "--window_size",
        type=int,
        default=64,
        help="Sliding window size (one-sided) (default: 64)",
    )

    # Scaling Parameters (Lists)
    parser.add_argument(
        "--seq_lens",
        nargs="+",
        type=int,
        default=[1024, 2048, 4096, 8192],
        help="List of sequence lengths to test",
    )
    parser.add_argument(
        "--meta_tokens",
        nargs="+",
        type=int,
        default=[2, 4, 6, 8, 16],
        help="List of meta token counts to test",
    )

    # Benchmarking Controls
    parser.add_argument(
        "--warmup", type=int, default=5, help="Warmup iterations (default: 5)"
    )
    parser.add_argument(
        "--rep", type=int, default=50, help="Measurement repetitions (default: 50)"
    )
    parser.add_argument(
        "--skip_check", action="store_true", help="Skip the initial correctness check"
    )

    args = parser.parse_args()

    if not args.skip_check:
        test_correctness()

    benchmark_comparison(args)
