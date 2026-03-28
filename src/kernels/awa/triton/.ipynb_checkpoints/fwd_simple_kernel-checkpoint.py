import triton
import triton.language as tl
import math


@triton.jit
def swa_simple_kernel_contig(
    Q, 
    K, 
    V, 
    Out,
    B: tl.constexpr, 
    H: tl.constexpr, 
    L: tl.constexpr, 
    D: tl.constexpr, 
    W: tl.constexpr,
    BLOCK: tl.constexpr
):
    off_m_block = tl.program_id(0)
    off_bh = tl.program_id(1)
    off_h = off_bh % H
    off_b = off_bh // H

    # Query indices this program handles
    offs_m = off_m_block * BLOCK + tl.arange(0, BLOCK)
    mask_m = offs_m < L
    offs_d = tl.arange(0, D)

    # Load queries
    q = tl.load(Q + (off_b * H * L * D + off_h * L * D + offs_m[:, None] * D + offs_d[None, :]),
                mask=mask_m[:, None],
                other=0.0)
    q = q / tl.sqrt(D.to(tl.float32))

    # Initialize accumulators for attention
    acc = tl.zeros([BLOCK, D], dtype=tl.float32)
    l_i = tl.zeros([BLOCK], dtype=tl.float32)
    m_i = tl.full([BLOCK], float('-inf'), dtype=tl.float32)

    # Sliding window attention
    for k_idx in range(L):
        dist = offs_m - k_idx
        mask_w = (dist >= 0) & (dist < W)

        # Load key and value
        k_t = tl.load(K + (off_b * H * L * D + off_h * L * D + k_idx * D + offs_d[None, :]),
                      mask=mask_w[:, None],
                      other=0.0)
        v_t = tl.load(V + (off_b * H * L * D + off_h * L * D + k_idx * D + offs_d[None, :]),
                      mask=mask_w[:, None],
                      other=0.0)

        # Compute attention
        qk = tl.dot(q, k_t)
        qk = tl.where(mask_w[:, None], qk, float('-inf'))

        # Numerically stable softmax
        m_new = tl.maximum(m_i, qk)
        p = tl.exp(qk - m_new[:, None])
        alpha = tl.exp(m_i - m_new)
        l_i = l_i * alpha + tl.sum(p, 1)
        acc = acc * alpha[:, None] + tl.dot(p, v_t)
        m_i = m_new

    # Store output
    tl.store(Out + (off_b * H * L * D + off_h * L * D + offs_m[:, None] * D + offs_d[None, :]),
             acc / (l_i[:, None] + 1e-8),
             mask=mask_m[:, None])
    