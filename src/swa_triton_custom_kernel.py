"""
Sliding window attention in Triton -- tiled (FlashAttention-style) kernels.

Two variants are provided, one per precision:
  * swa_tiled_triton_fp32 -- single-precision throughout (storage + compute).
  * swa_tiled_triton_fp16 -- half-precision storage, FP32 accumulators.

Each query tile of BLOCK_M rows iterates over key/value tiles of BLOCK_N
columns, visiting only those tiles that overlap the window
[i - bwd, i + fwd]. The softmax is computed with the standard FlashAttention
online algorithm (running max + denominator + rescale).

Autotuned parameters:
  - BLOCK_M, BLOCK_N : tile shape on the (query, key) axes.
  - num_warps        : warps per program instance; trades parallelism for
                       register pressure.
  - num_stages       : software-pipelining depth on KV-tile loads; hides
                       global-memory latency by issuing the next load while
                       the current matmul is still in flight.
"""

import math
import torch
import triton
import triton.language as tl


# ===========================================================================
# Autotune config spaces
# ===========================================================================

def _configs_fp32():
    """FP32 has larger per-element footprint, so tiles cap at 128 x 64."""
    return [
        triton.Config({"BLOCK_M": 32,  "BLOCK_N": 32},  num_warps=4, num_stages=2),
        triton.Config({"BLOCK_M": 32,  "BLOCK_N": 32},  num_warps=4, num_stages=3),
        triton.Config({"BLOCK_M": 64,  "BLOCK_N": 32},  num_warps=4, num_stages=2),
        triton.Config({"BLOCK_M": 64,  "BLOCK_N": 32},  num_warps=4, num_stages=3),
        triton.Config({"BLOCK_M": 64,  "BLOCK_N": 64},  num_warps=4, num_stages=2),
        triton.Config({"BLOCK_M": 64,  "BLOCK_N": 64},  num_warps=8, num_stages=2),
        triton.Config({"BLOCK_M": 64,  "BLOCK_N": 64},  num_warps=4, num_stages=3),
        triton.Config({"BLOCK_M": 128, "BLOCK_N": 32},  num_warps=8, num_stages=2),
        triton.Config({"BLOCK_M": 128, "BLOCK_N": 64},  num_warps=8, num_stages=2),
        triton.Config({"BLOCK_M": 128, "BLOCK_N": 64},  num_warps=8, num_stages=3),
    ]


def _configs_fp16():
    """FP16 halves register footprint, allowing 128 x 128 and beyond."""
    return [
        triton.Config({"BLOCK_M": 64,  "BLOCK_N": 32},  num_warps=4, num_stages=2),
        triton.Config({"BLOCK_M": 64,  "BLOCK_N": 64},  num_warps=4, num_stages=2),
        triton.Config({"BLOCK_M": 64,  "BLOCK_N": 64},  num_warps=4, num_stages=3),
        triton.Config({"BLOCK_M": 64,  "BLOCK_N": 128}, num_warps=4, num_stages=2),
        triton.Config({"BLOCK_M": 64,  "BLOCK_N": 128}, num_warps=8, num_stages=2),
        triton.Config({"BLOCK_M": 128, "BLOCK_N": 32},  num_warps=4, num_stages=3),
        triton.Config({"BLOCK_M": 128, "BLOCK_N": 64},  num_warps=4, num_stages=2),
        triton.Config({"BLOCK_M": 128, "BLOCK_N": 64},  num_warps=8, num_stages=2),
        triton.Config({"BLOCK_M": 128, "BLOCK_N": 64},  num_warps=8, num_stages=3),
        triton.Config({"BLOCK_M": 128, "BLOCK_N": 128}, num_warps=8, num_stages=2),
        triton.Config({"BLOCK_M": 128, "BLOCK_N": 128}, num_warps=8, num_stages=3),
    ]


_AUTOTUNE_KEY = ["N_CTX", "BLOCK_DMODEL", "BWD_WINDOW", "FWD_WINDOW"]


# ===========================================================================
# FP32 kernel
# ===========================================================================

@triton.autotune(configs=_configs_fp32(), key=_AUTOTUNE_KEY)
@triton.jit
def swa_tiled_kernel_fp32(
    Q, K, V, Out,
    stride_qb, stride_qh, stride_qm, stride_qk,
    stride_kb, stride_kh, stride_kn, stride_kk,
    stride_vb, stride_vh, stride_vn, stride_vk,
    stride_ob, stride_oh, stride_om, stride_ok,
    H, N_CTX, sm_scale,
    BWD_WINDOW: tl.constexpr,
    FWD_WINDOW: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_DMODEL: tl.constexpr,
):
    # ----- Program identification ------------------------------------------
    start_m = tl.program_id(0)
    off_hz  = tl.program_id(1)
    off_b   = off_hz // H
    off_h   = off_hz %  H

    q_offset = off_b * stride_qb + off_h * stride_qh
    k_offset = off_b * stride_kb + off_h * stride_kh
    v_offset = off_b * stride_vb + off_h * stride_vh
    o_offset = off_b * stride_ob + off_h * stride_oh

    offs_m = start_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_d = tl.arange(0, BLOCK_DMODEL)

    # ----- Load Q tile (FP32) ----------------------------------------------
    q_ptrs = Q + q_offset + offs_m[:, None] * stride_qm + offs_d[None, :] * stride_qk
    q_mask = offs_m[:, None] < N_CTX
    q = tl.load(q_ptrs, mask=q_mask, other=0.0)

    # ----- KV-tile range that overlaps the window for this query tile ------
    q_start = start_m * BLOCK_M
    q_end   = q_start + BLOCK_M - 1
    kv_lo   = tl.maximum(q_start - BWD_WINDOW, 0)
    kv_hi   = tl.minimum(q_end   + FWD_WINDOW, N_CTX - 1)
    n_start = (kv_lo // BLOCK_N) * BLOCK_N
    n_end   = kv_hi + 1

    # ----- Running softmax state -------------------------------------------
    m_i = tl.full((BLOCK_M,), -float("inf"), dtype=tl.float32)
    l_i = tl.zeros((BLOCK_M,), dtype=tl.float32)
    acc = tl.zeros((BLOCK_M, BLOCK_DMODEL), dtype=tl.float32)

    # ----- Iterate over relevant KV tiles ----------------------------------
    for n_block_start in range(n_start, n_end, BLOCK_N):
        offs_n = n_block_start + tl.arange(0, BLOCK_N)

        diff      = offs_n[None, :] - offs_m[:, None]
        in_window = (diff >= -BWD_WINDOW) & (diff <= FWD_WINDOW)
        in_seq_n  = offs_n < N_CTX
        valid     = in_window & in_seq_n[None, :]

        k_ptrs = K + k_offset + offs_n[:, None] * stride_kn + offs_d[None, :] * stride_kk
        v_ptrs = V + v_offset + offs_n[:, None] * stride_vn + offs_d[None, :] * stride_vk
        kv_mask_n = in_seq_n[:, None]
        k = tl.load(k_ptrs, mask=kv_mask_n, other=0.0)
        v = tl.load(v_ptrs, mask=kv_mask_n, other=0.0)

        qk = tl.dot(q, tl.trans(k))
        qk = qk * sm_scale
        qk = tl.where(valid, qk, -float("inf"))

        m_ij    = tl.max(qk, axis=1)
        m_new   = tl.maximum(m_i, m_ij)
        m_safe  = tl.where(m_new == -float("inf"), 0.0, m_new)
        alpha   = tl.exp(m_i - m_safe)
        p       = tl.exp(qk - m_safe[:, None])
        p       = tl.where(valid, p, 0.0)

        l_i = l_i * alpha + tl.sum(p, axis=1)
        acc = acc * alpha[:, None] + tl.dot(p, v)
        m_i = m_new

    # ----- Final normalization ---------------------------------------------
    l_safe = tl.where(l_i == 0.0, 1.0, l_i)
    acc    = acc / l_safe[:, None]

    o_ptrs = Out + o_offset + offs_m[:, None] * stride_om + offs_d[None, :] * stride_ok
    tl.store(o_ptrs, acc, mask=q_mask)


# ===========================================================================
# FP16 kernel
# ===========================================================================

@triton.autotune(configs=_configs_fp16(), key=_AUTOTUNE_KEY)
@triton.jit
def swa_tiled_kernel_fp16(
    Q, K, V, Out,
    stride_qb, stride_qh, stride_qm, stride_qk,
    stride_kb, stride_kh, stride_kn, stride_kk,
    stride_vb, stride_vh, stride_vn, stride_vk,
    stride_ob, stride_oh, stride_om, stride_ok,
    H, N_CTX, sm_scale,
    BWD_WINDOW: tl.constexpr,
    FWD_WINDOW: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_DMODEL: tl.constexpr,
):
    start_m = tl.program_id(0)
    off_hz  = tl.program_id(1)
    off_b   = off_hz // H
    off_h   = off_hz %  H

    q_offset = off_b * stride_qb + off_h * stride_qh
    k_offset = off_b * stride_kb + off_h * stride_kh
    v_offset = off_b * stride_vb + off_h * stride_vh
    o_offset = off_b * stride_ob + off_h * stride_oh

    offs_m = start_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_d = tl.arange(0, BLOCK_DMODEL)

    q_ptrs = Q + q_offset + offs_m[:, None] * stride_qm + offs_d[None, :] * stride_qk
    q_mask = offs_m[:, None] < N_CTX
    q = tl.load(q_ptrs, mask=q_mask, other=0.0)  # FP16

    q_start = start_m * BLOCK_M
    q_end   = q_start + BLOCK_M - 1
    kv_lo   = tl.maximum(q_start - BWD_WINDOW, 0)
    kv_hi   = tl.minimum(q_end   + FWD_WINDOW, N_CTX - 1)
    n_start = (kv_lo // BLOCK_N) * BLOCK_N
    n_end   = kv_hi + 1

    m_i = tl.full((BLOCK_M,), -float("inf"), dtype=tl.float32)
    l_i = tl.zeros((BLOCK_M,), dtype=tl.float32)
    acc = tl.zeros((BLOCK_M, BLOCK_DMODEL), dtype=tl.float32)

    for n_block_start in range(n_start, n_end, BLOCK_N):
        offs_n = n_block_start + tl.arange(0, BLOCK_N)

        diff      = offs_n[None, :] - offs_m[:, None]
        in_window = (diff >= -BWD_WINDOW) & (diff <= FWD_WINDOW)
        in_seq_n  = offs_n < N_CTX
        valid     = in_window & in_seq_n[None, :]

        k_ptrs = K + k_offset + offs_n[:, None] * stride_kn + offs_d[None, :] * stride_kk
        v_ptrs = V + v_offset + offs_n[:, None] * stride_vn + offs_d[None, :] * stride_vk
        kv_mask_n = in_seq_n[:, None]
        k = tl.load(k_ptrs, mask=kv_mask_n, other=0.0)  # FP16
        v = tl.load(v_ptrs, mask=kv_mask_n, other=0.0)  # FP16

        # FP16 x FP16 with FP32 accumulator on tensor cores.
        qk = tl.dot(q, tl.trans(k), out_dtype=tl.float32)
        qk = qk * sm_scale
        qk = tl.where(valid, qk, -float("inf"))

        m_ij    = tl.max(qk, axis=1)
        m_new   = tl.maximum(m_i, m_ij)
        m_safe  = tl.where(m_new == -float("inf"), 0.0, m_new)
        alpha   = tl.exp(m_i - m_safe)
        p       = tl.exp(qk - m_safe[:, None])
        p       = tl.where(valid, p, 0.0)

        l_i = l_i * alpha + tl.sum(p, axis=1)
        # Cast p to FP16 so PV matmul also runs on tensor cores; acc stays FP32.
        acc = acc * alpha[:, None] + tl.dot(p.to(tl.float16), v, out_dtype=tl.float32)
        m_i = m_new

    l_safe = tl.where(l_i == 0.0, 1.0, l_i)
    acc    = acc / l_safe[:, None]

    o_ptrs = Out + o_offset + offs_m[:, None] * stride_om + offs_d[None, :] * stride_ok
    tl.store(o_ptrs, acc.to(Out.dtype.element_ty), mask=q_mask)


# ===========================================================================
# Wrappers
# ===========================================================================

def swa_tiled_triton_fp32(q, k, v, bwd_window, fwd_window):
    """FP32 sliding window attention. Q, K, V: [B, H, N, D] CUDA tensors."""
    assert q.dtype == k.dtype == v.dtype == torch.float32, \
        f"swa_tiled_triton_fp32 requires FP32 inputs; got {q.dtype}"
    assert q.shape == k.shape == v.shape
    assert q.is_cuda and k.is_cuda and v.is_cuda

    B, H, N, D = q.shape
    out      = torch.empty_like(q)
    sm_scale = 1.0 / math.sqrt(D)
    BLOCK_D  = triton.next_power_of_2(D)

    grid = lambda META: (triton.cdiv(N, META["BLOCK_M"]), B * H)

    swa_tiled_kernel_fp32[grid](
        q, k, v, out,
        *q.stride(), *k.stride(), *v.stride(), *out.stride(),
        H, N, sm_scale,
        BWD_WINDOW=bwd_window,
        FWD_WINDOW=fwd_window,
        BLOCK_DMODEL=BLOCK_D,
    )
    return out


def swa_tiled_triton_fp16(q, k, v, bwd_window, fwd_window):
    """FP16 sliding window attention with FP32 accumulators."""
    assert q.dtype == k.dtype == v.dtype == torch.float16, \
        f"swa_tiled_triton_fp16 requires FP16 inputs; got {q.dtype}"
    assert q.shape == k.shape == v.shape
    assert q.is_cuda and k.is_cuda and v.is_cuda

    B, H, N, D = q.shape
    out      = torch.empty_like(q)
    sm_scale = 1.0 / math.sqrt(D)
    BLOCK_D  = triton.next_power_of_2(D)

    grid = lambda META: (triton.cdiv(N, META["BLOCK_M"]), B * H)

    swa_tiled_kernel_fp16[grid](
        q, k, v, out,
        *q.stride(), *k.stride(), *v.stride(), *out.stride(),
        H, N, sm_scale,
        BWD_WINDOW=bwd_window,
        FWD_WINDOW=fwd_window,
        BLOCK_DMODEL=BLOCK_D,
    )
    return out


def swa_tiled_triton(q, k, v, bwd_window, fwd_window):
    """Dispatch by dtype."""
    if q.dtype == torch.float32:
        return swa_tiled_triton_fp32(q, k, v, bwd_window, fwd_window)
    if q.dtype == torch.float16:
        return swa_tiled_triton_fp16(q, k, v, bwd_window, fwd_window)
    raise ValueError(f"Unsupported dtype {q.dtype}; expected FP32 or FP16.")
