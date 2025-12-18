import math
import torch
import torch.nn as nn
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


@triton.jit
def gla_linear_kernel(
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


class LizardAttention(nn.Module):
    def __init__(self, d_model, n_heads, window_size=64, alpha=1.0, m=4):
        super().__init__()
        self.d_model, self.n_heads = d_model, n_heads
        self.d_head = d_model // n_heads
        self.window_size, self.alpha, self.m = window_size, alpha, m
        self.meta_tokens = nn.Parameter(torch.randn(1, n_heads, m, self.d_head))
        self.W_gamma = nn.Parameter(torch.randn(1, n_heads, 1, self.d_head))

    def forward(self, q, k, v, x=None, alpha=None, use_triton=False):
        alpha = alpha if alpha is not None else self.alpha
        x = x if x is not None else q
        if use_triton:
            g_out, a_out = self.fwd_gla_triton(q, k, v, x), self.fwd_awa_triton(q, k, v)
        else:
            g_out, a_out = self.fwd_gla(q, k, v, x), self.fwd_awa(q, k, v)
        return g_out + alpha * a_out

    def fwd_awa_triton(self, q, k, v):
        B, H, L, D = q.shape
        out = torch.empty_like(q)
        awa_kernel[(B, H, L)](
            q,
            k,
            v,
            self.meta_tokens,
            out,
            *q.stride(),
            *k.stride(),
            *v.stride(),
            *self.meta_tokens.stride(),
            *out.stride(),
            B,
            H,
            L,
            D,
            self.m,
            self.window_size,
            BLOCK_D=triton.next_power_of_2(D),
        )
        return out

    def fwd_gla_triton(self, q, k, v, x):
        B, H, L, D = q.shape
        gamma = torch.sigmoid((self.W_gamma * x).sum(-1, keepdim=True)).squeeze(-1)
        out = torch.empty_like(q)
        gla_linear_kernel[(B, H)](
            q,
            k,
            v,
            gamma,
            out,
            *q.stride(),
            *k.stride(),
            *v.stride(),
            *gamma.stride(),
            *out.stride(),
            B,
            H,
            L,
            D,
            BLOCK_D=triton.next_power_of_2(D),
        )
        return out

    def fwd_awa(self, q, k, v):
        B, H, L, D = q.shape
        mt = self.meta_tokens.expand(B, -1, -1, -1)
        out = torch.zeros_like(q)
        for i in range(L):
            s, e = max(0, i - self.window_size + 1), min(L, i + self.window_size)
            qi = q[:, :, i : i + 1, :]
            kw, vw = k[:, :, s:e, :], v[:, :, s:e, :]
            sk, st = torch.matmul(qi, kw.transpose(-2, -1)) / math.sqrt(
                D
            ), torch.matmul(qi, mt.transpose(-2, -1)) / math.sqrt(D)
            mv = torch.max(torch.cat([sk, st], dim=-1), dim=-1, keepdim=True)[0]
            ek, et = torch.exp(sk - mv), torch.exp(st - mv)
            out[:, :, i : i + 1, :] = torch.matmul(ek, vw) / (
                ek.sum(-1, True) + et.sum(-1, True) + 1e-8
            )
        return out

    def fwd_gla(self, q, k, v, x):
        B, H, L, D = q.shape
        gamma = torch.sigmoid((self.W_gamma * x).sum(-1, keepdim=True))
        out = torch.zeros_like(q)
        for h in range(H):
            sn, sd = 0.0, 0.0
            for i in range(L):
                g = gamma[:, h, i : i + 1, :]
                qi, ki, vi = (
                    q[:, h, i : i + 1, :],
                    k[:, h, i : i + 1, :],
                    v[:, h, i : i + 1, :],
                )
                qk = qi @ ki.transpose(-1, -2)
                sn = sn * g + qk @ vi
                sd = sd * g + qk
                out[:, h, i : i + 1, :] = sn / (sd + 1e-8)
        return out


def get_cuda_time(func, *args, **kwargs):
    # Warmup
    for _ in range(5):
        func(*args, **kwargs)
    torch.cuda.synchronize()

    start_event = torch.cuda.Event(enable_timing=True)
    end_event = torch.cuda.Event(enable_timing=True)

    start_event.record()
    func(*args, **kwargs)
    end_event.record()

    torch.cuda.synchronize()
    return start_event.elapsed_time(end_event)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--seq_len", type=int, default=1024)
    args = parser.parse_args()

    device, dt = "cuda", torch.float32
    model = LizardAttention(512, 2).to(device).to(dt)
    q, k, v = [
        torch.randn(1, 2, args.seq_len, 256, device=device, dtype=dt) for _ in range(3)
    ]

    with torch.no_grad():
        # Correctness
        o_py = model(q, k, v, use_triton=False)
        o_tr = model(q, k, v, use_triton=True)
        print(f"Max diff: {(o_py - o_tr).abs().max().item():.6e}")

        # Timing
        t_py = get_cuda_time(model, q, k, v, use_triton=False)
        t_tr = get_cuda_time(model, q, k, v, use_triton=True)

        print(f"\nCUDA Performance (L={args.seq_len}):")
        print(f"PyTorch Time: {t_py:.3f} ms")
        print(f"Triton Time:  {t_tr:.3f} ms")
        print(f"Speedup:      {t_py/t_tr:.2f}x")
