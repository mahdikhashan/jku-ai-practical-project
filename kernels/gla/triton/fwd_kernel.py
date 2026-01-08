import triton
import triton.language as tl


@triton.jit
def parallel_gla_kernel(
        Q, K, V, Out,
        stride_qb, stride_ql, stride_qh, stride_qd,
        stride_kb, stride_kl, stride_kh, stride_kd,
        stride_vb, stride_vl, stride_vh, stride_vd,
        stride_ob, stride_ol, stride_oh, stride_od,
        B, L, H,
        BLOCK_M: tl.constexpr,
        BLOCK_N: tl.constexpr,
        D_QK: tl.constexpr,
        D_V: tl.constexpr,
):
    pid_m = tl.program_id(0)
    pid_bh = tl.program_id(1)

    i_b = pid_bh // H
    i_h = pid_bh % H

    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n_base = tl.arange(0, BLOCK_N)

    offs_d_qk = tl.arange(0, D_QK)
    offs_d_v = tl.arange(0, D_V)

    Q_ptr = Q + (i_b * stride_qb + i_h * stride_qh)
    K_ptr = K + (i_b * stride_kb + i_h * stride_kh)
    V_ptr = V + (i_b * stride_vb + i_h * stride_vh)
    Out_ptr = Out + (i_b * stride_ob + i_h * stride_oh)

    q_ptrs = Q_ptr + (offs_m[:, None] * stride_ql + offs_d_qk[None, :] * stride_qd)
    q = tl.load(q_ptrs, mask=offs_m[:, None] < L, other=0.0)

    acc = tl.zeros([BLOCK_M, D_V], dtype=tl.float32)
    loop_end = (pid_m + 1) * BLOCK_M

    for start_n in range(0, loop_end, BLOCK_N):
        offs_n = start_n + offs_n_base

        k_ptrs = K_ptr + (offs_n[None, :] * stride_kl + offs_d_qk[:, None] * stride_kd)
        v_ptrs = V_ptr + (offs_n[:, None] * stride_vl + offs_d_v[None, :] * stride_vd)

        k = tl.load(k_ptrs, mask=offs_n[None, :] < L, other=0.0)
        v = tl.load(v_ptrs, mask=offs_n[:, None] < L, other=0.0)

        qk = tl.dot(q, k)

        mask = offs_m[:, None] >= offs_n[None, :]
        qk = tl.where(mask, qk, 0.0)

        acc += tl.dot(qk, v)

    out_ptrs = Out_ptr + (offs_m[:, None] * stride_ol + offs_d_v[None, :] * stride_od)
    tl.store(out_ptrs, acc.to(Out.dtype.element_ty), mask=offs_m[:, None] < L)
