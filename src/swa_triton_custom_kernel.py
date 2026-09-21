import math

import torch
import triton
import triton.language as tl

TILED_CONFIGS = [
    triton.Config({"BLOCK_M": m, "BLOCK_N": n}, num_warps=w, num_stages=s)
    for m, n, w, s in [
        (32, 32, 4, 2), (64, 32, 4, 3), (64, 64, 4, 2), (64, 64, 4, 3),
        (64, 128, 4, 2), (128, 32, 4, 3), (128, 64, 8, 2), (128, 64, 8, 3),
        (128, 128, 8, 2),
    ]
]

STRIDED_CONFIGS = [
    triton.Config({"BLOCK_M": m, "BLOCK_W": w}, num_warps=nw)
    for m, w, nw in [(8, 16, 4), (16, 8, 4), (16, 16, 4), (16, 16, 8), (32, 8, 8)]
]


@triton.jit
def _online_softmax_step(m_i, l_i, s):
    m_new = tl.maximum(m_i, tl.max(s, axis=1))
    m_safe = tl.where(m_new == -float("inf"), 0.0, m_new)
    alpha = tl.exp(m_i - m_safe)
    p = tl.exp(s - m_safe[:, None])
    l_i = l_i * alpha + tl.sum(p, axis=1)
    return m_new, l_i, alpha, p


@triton.autotune(configs=TILED_CONFIGS, key=["N", "BWD", "FWD"])
@triton.jit
def swa_tiled_kernel(Q, K, V, Out, N, sm_scale,
                     BWD: tl.constexpr, FWD: tl.constexpr, D: tl.constexpr,
                     BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr):
    pid_m = tl.program_id(0)
    base = tl.program_id(1) * N * D
    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_d = tl.arange(0, D)

    q = tl.load(Q + base + offs_m[:, None] * D + offs_d[None, :],
                mask=offs_m[:, None] < N, other=0.0)

    m_i = tl.full([BLOCK_M], -float("inf"), tl.float32)
    l_i = tl.zeros([BLOCK_M], tl.float32)
    acc = tl.zeros([BLOCK_M, D], tl.float32)

    lo = tl.maximum(pid_m * BLOCK_M - BWD, 0) // BLOCK_N * BLOCK_N
    hi = tl.minimum(pid_m * BLOCK_M + BLOCK_M - 1 + FWD, N - 1) + 1
    for start_n in range(lo, hi, BLOCK_N):
        offs_n = start_n + tl.arange(0, BLOCK_N)
        diff = offs_n[None, :] - offs_m[:, None]
        valid = (diff >= -BWD) & (diff <= FWD) & (offs_n[None, :] < N)

        kv = base + offs_n[:, None] * D + offs_d[None, :]
        k = tl.load(K + kv, mask=offs_n[:, None] < N, other=0.0)
        v = tl.load(V + kv, mask=offs_n[:, None] < N, other=0.0)

        s = tl.dot(q, tl.trans(k), out_dtype=tl.float32) * sm_scale
        s = tl.where(valid, s, -float("inf"))
        m_i, l_i, alpha, p = _online_softmax_step(m_i, l_i, s)
        acc = acc * alpha[:, None] + tl.dot(p.to(v.dtype), v, out_dtype=tl.float32)

    acc = acc / tl.where(l_i == 0.0, 1.0, l_i)[:, None]
    tl.store(Out + base + offs_m[:, None] * D + offs_d[None, :],
             acc.to(Out.dtype.element_ty), mask=offs_m[:, None] < N)


@triton.autotune(configs=STRIDED_CONFIGS, key=["N", "BWD", "W"])
@triton.jit
def swa_strided_kernel(Q, K, V, Out, N, sm_scale,
                       BWD: tl.constexpr, W: tl.constexpr, D: tl.constexpr,
                       BLOCK_M: tl.constexpr, BLOCK_W: tl.constexpr):
    base = tl.program_id(1) * N * D
    offs_m = tl.program_id(0) * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_d = tl.arange(0, D)

    q = tl.load(Q + base + offs_m[:, None] * D + offs_d[None, :],
                mask=offs_m[:, None] < N, other=0.0).to(tl.float32)

    m_i = tl.full([BLOCK_M], -float("inf"), tl.float32)
    l_i = tl.zeros([BLOCK_M], tl.float32)
    acc = tl.zeros([BLOCK_M, D], tl.float32)

    for w0 in range(0, W, BLOCK_W):
        offs_w = w0 + tl.arange(0, BLOCK_W)
        pos = offs_m[:, None] - BWD + offs_w[None, :]
        valid = (offs_w[None, :] < W) & (pos >= 0) & (pos < N)

        kv = base + pos[:, :, None] * D + offs_d[None, None, :]
        k = tl.load(K + kv, mask=valid[:, :, None], other=0.0).to(tl.float32)
        v = tl.load(V + kv, mask=valid[:, :, None], other=0.0).to(tl.float32)

        s = tl.sum(q[:, None, :] * k, axis=2) * sm_scale
        s = tl.where(valid, s, -float("inf"))
        m_i, l_i, alpha, p = _online_softmax_step(m_i, l_i, s)
        acc = acc * alpha[:, None] + tl.sum(p[:, :, None] * v, axis=1)

    acc = acc / tl.where(l_i == 0.0, 1.0, l_i)[:, None]
    tl.store(Out + base + offs_m[:, None] * D + offs_d[None, :],
             acc.to(Out.dtype.element_ty), mask=offs_m[:, None] < N)


def _prepare(q, k, v):
    assert q.shape == k.shape == v.shape and q.dtype == k.dtype == v.dtype
    assert q.dtype in (torch.float32, torch.float16)
    D = q.shape[-1]
    assert D >= 16 and D & (D - 1) == 0, "head dim must be a power of two >= 16"
    return q.contiguous(), k.contiguous(), v.contiguous()


def swa_tiled_triton(q, k, v, bwd, fwd):
    q, k, v = _prepare(q, k, v)
    B, H, N, D = q.shape
    out = torch.empty_like(q)
    grid = lambda meta: (triton.cdiv(N, meta["BLOCK_M"]), B * H)
    swa_tiled_kernel[grid](q, k, v, out, N, 1.0 / math.sqrt(D), BWD=bwd, FWD=fwd, D=D)
    return out


def swa_strided_triton(q, k, v, bwd, fwd):
    q, k, v = _prepare(q, k, v)
    B, H, N, D = q.shape
    out = torch.empty_like(q)
    grid = lambda meta: (triton.cdiv(N, meta["BLOCK_M"]), B * H)
    swa_strided_kernel[grid](q, k, v, out, N, 1.0 / math.sqrt(D), BWD=bwd, W=bwd + fwd + 1, D=D)
    return out