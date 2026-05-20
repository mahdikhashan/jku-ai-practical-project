import math
import torch
import triton
import triton.language as tl
import os


@triton.autotune(
    configs=[
        triton.Config({"BLOCK_M": 32}, num_warps=4),
        triton.Config({"BLOCK_M": 64}, num_warps=4),
        triton.Config({"BLOCK_M": 64}, num_warps=8),
        triton.Config({"BLOCK_M": 128}, num_warps=8),
    ],
    key=["N_CTX"],
)
@triton.jit
def swa_strided_kernel(
    Q, K, V, Out,
    stride_qb, stride_qh, stride_qm, stride_qk,
    stride_kb, stride_kh, stride_km, stride_kk,
    stride_vb, stride_vh, stride_vm, stride_vk,
    stride_ob, stride_oh, stride_om, stride_ok,
    Z, H, N_CTX,
    sm_scale,
    BWD_WINDOW: tl.constexpr,
    FWD_WINDOW: tl.constexpr,
    BLOCK_M: tl.constexpr,
    WIN_SIZE: tl.constexpr,
    BLOCK_DMODEL: tl.constexpr,
):
    start_m = tl.program_id(0)
    off_hz = tl.program_id(1)
    off_b = off_hz // H
    off_h = off_hz % H

    q_offset = off_b * stride_qb + off_h * stride_qh
    k_offset = off_b * stride_kb + off_h * stride_kh
    v_offset = off_b * stride_vb + off_h * stride_vh
    o_offset = off_b * stride_ob + off_h * stride_oh

    offs_m = start_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_d = tl.arange(0, BLOCK_DMODEL)
    offs_w = tl.arange(0, WIN_SIZE) - BWD_WINDOW

    q_ptrs = Q + q_offset + offs_m[:, None] * stride_qm + offs_d[None, :] * stride_qk
    q_mask = offs_m[:, None] < N_CTX
    q = tl.load(q_ptrs, mask=q_mask, other=0.0)

    kv_pos = offs_m[:, None] + offs_w[None, :]
    in_seq = (kv_pos >= 0) & (kv_pos < N_CTX)
    kv_pos_safe = tl.where(in_seq, kv_pos, 0)

    k_ptrs = (K + k_offset
              + kv_pos_safe[:, :, None] * stride_km
              + offs_d[None, None, :] * stride_kk)
    k_mask = in_seq[:, :, None]
    k_win = tl.load(k_ptrs, mask=k_mask, other=0.0)

    v_ptrs = (V + v_offset
              + kv_pos_safe[:, :, None] * stride_vm
              + offs_d[None, None, :] * stride_vk)
    v_win = tl.load(v_ptrs, mask=k_mask, other=0.0)

    qk = tl.sum(q[:, None, :] * k_win, axis=2)
    qk = qk * sm_scale
    qk = tl.where(in_seq, qk, -float("inf"))

    m_i = tl.max(qk, axis=1)
    m_safe = tl.where(m_i == -float("inf"), 0.0, m_i)
    p = tl.exp(qk - m_safe[:, None])
    p = tl.where(in_seq, p, 0.0)
    l_i = tl.sum(p, axis=1)

    acc = tl.sum(p[:, :, None] * v_win, axis=1)
    acc = acc / l_i[:, None]

    o_ptrs = Out + o_offset + offs_m[:, None] * stride_om + offs_d[None, :] * stride_ok
    tl.store(o_ptrs, acc.to(Out.dtype.element_ty), mask=q_mask)


def swa_strided(q, k, v, bwd_window, fwd_window):
    B, H, N, D = q.shape
    out = torch.empty_like(q)
    sm_scale = 1.0 / math.sqrt(D)
    win_size = 1 + bwd_window + fwd_window
    WIN_SIZE_POW2 = triton.next_power_of_2(win_size)
    BLOCK_DMODEL = triton.next_power_of_2(D)

    grid = lambda META: (triton.cdiv(N, META["BLOCK_M"]), B * H)

    swa_strided_triton_kernel[grid](
        q, k, v, out,
        *q.stride(), *k.stride(), *v.stride(), *out.stride(),
        B, H, N,
        sm_scale,
        BWD_WINDOW=bwd_window,
        FWD_WINDOW=fwd_window,
        WIN_SIZE=WIN_SIZE_POW2,
        BLOCK_DMODEL=BLOCK_DMODEL,
    )
    return out
