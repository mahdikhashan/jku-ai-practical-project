import triton
import triton.language as tl


@triton.jit
def gla_kernel(
    Q,
    K,
    V,
    Gamma,
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
    stride_gb,
    stride_gh,
    stride_gl,
    stride_ob,
    stride_oh,
    stride_ol,
    stride_od,
    B,
    H,
    L,
    D,
    BLOCK_D: tl.constexpr,
):
    b, h = tl.program_id(0), tl.program_id(1)
    d_range = tl.arange(0, BLOCK_D)
    d_mask = d_range < D

    state_num = tl.zeros([BLOCK_D], dtype=tl.float32)
    state_den = 0.0

    for i in range(L):
        q = tl.load(
            Q + b * stride_qb + h * stride_qh + i * stride_ql + d_range * stride_qd,
            mask=d_mask,
            other=0.0,
        )
        g = tl.load(Gamma + b * stride_gb + h * stride_gh + i * stride_gl)
        k = tl.load(
            K + b * stride_kb + h * stride_kh + i * stride_kl + d_range * stride_kd,
            mask=d_mask,
            other=0.0,
        )
        v = tl.load(
            V + b * stride_vb + h * stride_vh + i * stride_vl + d_range * stride_vd,
            mask=d_mask,
            other=0.0,
        )

        qk = tl.sum(q * k)
        state_num = state_num * g + qk * v
        state_den = state_den * g + qk

        tl.store(
            Out + b * stride_ob + h * stride_oh + i * stride_ol + d_range * stride_od,
            state_num / (state_den + 1e-8),
            mask=d_mask,
        )
