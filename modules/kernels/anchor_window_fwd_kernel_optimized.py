import triton  # type: ignore
import triton.language as tl # type: ignore #


@triton.jit
def anchor_window_fwd_kernel_optimized(
    Q,
    K,
    V,
    Out,
    # --- CRITICAL FIX: Explicit 64-bit Strides ---
    stride_qb: tl.int64,
    stride_qh: tl.int64,
    stride_qs: tl.int64,
    stride_qd: tl.int64,
    stride_kb: tl.int64,
    stride_kh: tl.int64,
    stride_ks: tl.int64,
    stride_kd: tl.int64,
    stride_vb: tl.int64,
    stride_vh: tl.int64,
    stride_vs: tl.int64,
    stride_vd: tl.int64,
    stride_ob: tl.int64,
    stride_oh: tl.int64,
    stride_os: tl.int64,
    stride_od: tl.int64,
    seq_len,
    d_head,
    window_size,
    BLOCK_Q: tl.constexpr,
    BLOCK_K: tl.constexpr,
    BLOCK_DMODEL: tl.constexpr,
):
    # Force 64-bit indexing for large grids
    pid_m = tl.program_id(0).to(tl.int64)
    pid_b = tl.program_id(1).to(tl.int64)
    pid_h = tl.program_id(2).to(tl.int64)

    # Offsets for Q
    off_m = pid_m * BLOCK_Q + tl.arange(0, BLOCK_Q)
    off_d = tl.arange(0, BLOCK_DMODEL)
    mask_m = off_m < seq_len

    # Load Q (BF16)
    # Calculation strictly in 64-bit to avoid overflow
    q_ptr = (
        Q
        + (pid_b * stride_qb + pid_h * stride_qh)
        + off_m[:, None] * stride_qs
        + off_d[None, :] * stride_qd
    )
    q = tl.load(q_ptr, mask=mask_m[:, None], other=0.0)

    # Accumulators (FP32)
    acc = tl.zeros([BLOCK_Q, BLOCK_DMODEL], dtype=tl.float32)
    l_i = tl.zeros([BLOCK_Q], dtype=tl.float32)
    m_i = tl.full([BLOCK_Q], float("-inf"), dtype=tl.float32)

    start_idx = pid_m * BLOCK_Q
    min_k_idx = tl.maximum(0, start_idx - window_size)
    start_block_k = min_k_idx // BLOCK_K

    # Loop over K/V blocks
    for block_k in range(start_block_k, pid_m + 1):
        off_n = block_k * BLOCK_K + tl.arange(0, BLOCK_K)
        mask_n = off_n < seq_len

        # Load K, V (BF16)
        # Note: K loaded as [BLOCK_K, D]
        k_ptr = (
            K
            + (pid_b * stride_kb + pid_h * stride_kh)
            + off_n[:, None] * stride_ks
            + off_d[None, :] * stride_kd
        )
        v_ptr = (
            V
            + (pid_b * stride_vb + pid_h * stride_vh)
            + off_n[:, None] * stride_vs
            + off_d[None, :] * stride_vd
        )

        k = tl.load(k_ptr, mask=mask_n[:, None], other=0.0)
        v = tl.load(v_ptr, mask=mask_n[:, None], other=0.0)

        # Compute Q @ K.T
        qk = tl.dot(q, tl.trans(k))

        # Masking
        diff = off_m[:, None] - off_n[None, :]
        window_mask = (diff >= 0) & (diff <= window_size)
        qk = tl.where(
            window_mask & mask_m[:, None] & mask_n[None, :], qk, float("-inf")
        )

        # Softmax
        m_curr = tl.max(qk, 1)
        m_new = tl.maximum(m_i, m_curr)
        alpha = tl.exp(m_i - m_new)
        p = tl.exp(qk - m_new[:, None])

        # Update Accumulator
        acc = acc * alpha[:, None] + tl.dot(p.to(v.dtype), v)
        l_i = l_i * alpha + tl.sum(p, 1)
        m_i = m_new

    # Finalize
    acc = acc / l_i[:, None]

    # Store
    out_ptr = (
        Out
        + (pid_b * stride_ob + pid_h * stride_oh)
        + off_m[:, None] * stride_os
        + off_d[None, :] * stride_od
    )
    tl.store(out_ptr, acc, mask=mask_m[:, None])
