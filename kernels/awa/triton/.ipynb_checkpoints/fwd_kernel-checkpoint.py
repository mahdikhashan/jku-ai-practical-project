import triton
import triton.language as tl


@triton.jit
def awa_kernel(
    Q,
    K,
    V,
    Meta,
    Out,
    stride_qb,
    stride_qh,
    stride_ql,
    stride_qd,
    stride_kb,
    stride_kh,
    stride_kl,
    stride_kd,
    stride_vb,
    stride_vh,
    stride_vl,
    stride_vd,
    stride_mb,
    stride_mh,
    stride_mm,
    stride_md,
    stride_ob,
    stride_oh,
    stride_ol,
    stride_od,
    B,
    H,
    L,
    D,
    M,
    W,
    BLOCK_D: tl.constexpr,
):
    b, h, i = tl.program_id(0), tl.program_id(1), tl.program_id(2)
    d_range = tl.arange(0, BLOCK_D)
    d_mask = d_range < D

    q = tl.load(
        Q + b * stride_qb + h * stride_qh + i * stride_ql + d_range * stride_qd,
        mask=d_mask,
        other=0.0,
    )
    sqrt_d = tl.sqrt(D.to(tl.float32))

    m_i = -float("inf")
    for m_idx in range(M):
        meta = tl.load(
            Meta + h * stride_mh + m_idx * stride_mm + d_range * stride_md,
            mask=d_mask,
            other=0.0,
        )
        m_i = tl.maximum(m_i, tl.sum(q * meta) / sqrt_d)

    start, end = tl.maximum(0, i - W + 1), tl.minimum(L, i + W)
    for t in range(start, end):
        k = tl.load(
            K + b * stride_kb + h * stride_kh + t * stride_kl + d_range * stride_kd,
            mask=d_mask,
            other=0.0,
        )
        m_i = tl.maximum(m_i, tl.sum(q * k) / sqrt_d)

    num = tl.zeros([BLOCK_D], dtype=tl.float32)
    den = 0.0
    for m_idx in range(M):
        meta = tl.load(
            Meta + h * stride_mh + m_idx * stride_mm + d_range * stride_md,
            mask=d_mask,
            other=0.0,
        )
        den += tl.exp((tl.sum(q * meta) / sqrt_d) - m_i)

    for t in range(start, end):
        k = tl.load(
            K + b * stride_kb + h * stride_kh + t * stride_kl + d_range * stride_kd,
            mask=d_mask,
            other=0.0,
        )
        v = tl.load(
            V + b * stride_vb + h * stride_vh + t * stride_vl + d_range * stride_vd,
            mask=d_mask,
            other=0.0,
        )
        e = tl.exp((tl.sum(q * k) / sqrt_d) - m_i)
        num += e * v
        den += e

    tl.store(
        Out + b * stride_ob + h * stride_oh + i * stride_ol + d_range * stride_od,
        num / (den + 1e-8),
        mask=d_mask,
    )
