import triton
import triton.language as tl


@triton.jit
def parallel_gla_kernel(
        Q, K, V, Out,
        stride_qb, stride_qh, stride_ql, stride_qd,
        stride_kb, stride_kh, stride_kl, stride_kd,
        stride_vb, stride_vh, stride_vl, stride_vd,
        stride_ob, stride_oh, stride_ol, stride_od,
        B, H, L,
        D_QK: tl.constexpr,
        D_V: tl.constexpr,
        BLOCK_M: tl.constexpr,
        BLOCK_N: tl.constexpr,
):
    # Grid: (L // BLOCK_M, Batch * Heads)
    pid_m = tl.program_id(0)
    pid_bh = tl.program_id(1)

    i_b = pid_bh // H
    i_h = pid_bh % H

    Q_block_ptr = tl.make_block_ptr(
        base=Q + (i_b * stride_qb + i_h * stride_qh),
        shape=(L, D_QK),
        strides=(stride_ql, stride_qd),
        offsets=(pid_m * BLOCK_M, 0),
        block_shape=(BLOCK_M, D_QK),
        order=(1, 0)
    )

    acc = tl.zeros([BLOCK_M, D_V], dtype=tl.float32)

    for start_n in range(0, (pid_m + 1) * BLOCK_M, BLOCK_N):
        # K ptr: Iterates through history chunks
        K_block_ptr = tl.make_block_ptr(
            base=K + (i_b * stride_kb + i_h * stride_kh),
            shape=(D_QK, L),  # Transposed for dot product
            strides=(stride_kd, stride_kl),
            offsets=(0, start_n),
            block_shape=(D_QK, BLOCK_N),
            order=(0, 1)
        )

        V_block_ptr = tl.make_block_ptr(
            base=V + (i_b * stride_vb + i_h * stride_vh),
            shape=(L, D_V),
            strides=(stride_vl, stride_vd),
            offsets=(start_n, 0),
            block_shape=(BLOCK_N, D_V),
            order=(1, 0)
        )

        q = tl.load(Q_block_ptr)
        k = tl.load(K_block_ptr)
        v = tl.load(V_block_ptr)

        qk = tl.dot(q, k)

        # Causal Mask
        offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
        offs_n = start_n + tl.arange(0, BLOCK_N)

        mask = offs_m[:, None] >= offs_n[None, :]

        qk = tl.where(mask, qk, 0.0)

        acc += tl.dot(qk, v)

    Out_block_ptr = tl.make_block_ptr(
        base=Out + (i_b * stride_ob + i_h * stride_oh),
        shape=(L, D_V),
        strides=(stride_ol, stride_od),
        offsets=(pid_m * BLOCK_M, 0),
        block_shape=(BLOCK_M, D_V),
        order=(1, 0)
    )

    tl.store(Out_block_ptr, acc.to(Out.dtype.element_ty))
