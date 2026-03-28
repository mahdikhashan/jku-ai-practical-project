import triton
import triton.language as tl


@triton.jit
def awa_tiled_kernel(
        Q, K, V, Meta, Out,
        stride_qb, stride_qh, stride_ql, stride_qd,
        stride_kb, stride_kh, stride_kl, stride_kd,
        stride_vb, stride_vh, stride_vl, stride_vd,
        stride_mb, stride_mh, stride_mm, stride_md,
        stride_ob, stride_oh, stride_ol, stride_od,
        B, H, L, D, W,
        BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_D: tl.constexpr,
        BLOCK_M_META: tl.constexpr,
        M: tl.constexpr,
):
    off_m_block = tl.program_id(0)
    off_bh = tl.program_id(1)
    off_h = off_bh % H
    off_b = off_bh // H

    offs_m = off_m_block * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_d = tl.arange(0, BLOCK_D)
    mask_m = offs_m < L

    Q_base = Q + (off_b * stride_qb + off_h * stride_qh)
    K_base = K + (off_b * stride_kb + off_h * stride_kh)
    V_base = V + (off_b * stride_vb + off_h * stride_vh)
    Meta_base = Meta + (off_b * stride_mb + off_h * stride_mh)
    Out_base = Out + (off_b * stride_ob + off_h * stride_oh)

    # Load Query as FP32
    q_ptr = Q_base + (offs_m[:, None] * stride_ql + offs_d[None, :] * stride_qd)
    q = tl.load(q_ptr, mask=mask_m[:, None] & (offs_d[None, :] < D), other=0.0)
    q = q * (1.0 / tl.sqrt(D.to(tl.float32)))

    # Sink
    offs_meta_m = tl.arange(0, BLOCK_M_META)
    meta_ptr = Meta_base + (offs_meta_m[None, :] * stride_mm + offs_d[:, None] * stride_md)
    meta_t = tl.load(meta_ptr, mask=(offs_meta_m[None, :] < M) & (offs_d[:, None] < D), other=0.0)

    # tl.dot(fp32, fp32)
    meta_scores = tl.dot(q, meta_t)
    meta_scores = tl.where(offs_meta_m[None, :] < M, meta_scores, float("-inf"))

    m_i = tl.max(meta_scores, 1)
    l_i = tl.sum(tl.exp(meta_scores - m_i[:, None]), 1)
    acc = tl.zeros([BLOCK_M, BLOCK_D], dtype=tl.float32)

    # --- Sliding Window ---
    start_m_idx = off_m_block * BLOCK_M
    min_k_idx = tl.maximum(0, start_m_idx - W + 1)
    max_k_idx = (off_m_block + 1) * BLOCK_M

    start_block_n = min_k_idx // BLOCK_N
    end_block_n = tl.cdiv(max_k_idx, BLOCK_N)
    offs_n_base = tl.arange(0, BLOCK_N)

    for block_n in range(start_block_n, end_block_n):
        start_n = block_n * BLOCK_N
        offs_n = start_n + offs_n_base
        k_ptr = K_base + (offs_n[None, :] * stride_kl + offs_d[:, None] * stride_kd)
        k_t = tl.load(k_ptr, mask=(offs_n[None, :] < L) & (offs_d[:, None] < D), other=0.0)

        # TF32 Dot
        qk = tl.dot(q, k_t)

        dist = offs_m[:, None] - offs_n[None, :]
        mask_val = (dist >= 0) & (dist < W)
        qk = tl.where(mask_val, qk, float("-inf"))

        m_curr = tl.max(qk, 1)
        m_new = tl.maximum(m_i, m_curr)
        p = tl.exp(qk - m_new[:, None])
        alpha = tl.exp(m_i - m_new)
        l_i = l_i * alpha + tl.sum(p, 1)

        v_ptr = V_base + (offs_n[:, None] * stride_vl + offs_d[None, :] * stride_vd)
        v = tl.load(v_ptr, mask=(offs_n[:, None] < L) & (offs_d[None, :] < D), other=0.0)

        acc = acc * alpha[:, None]
        # TF32 Dot (Accumulate in FP32)
        acc += tl.dot(p.to(tl.float32), v)
        m_i = m_new

    # Sink
    out = acc / (l_i[:, None] + 1e-8)
    out_ptr = Out_base + (offs_m[:, None] * stride_ol + offs_d[None, :] * stride_od)
    tl.store(out_ptr, out, mask=mask_m[:, None] & (offs_d[None, :] < D))
