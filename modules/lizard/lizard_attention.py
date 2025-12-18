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
    b = tl.program_id(0)
    h = tl.program_id(1)
    i = tl.program_id(2)

    d_range = tl.arange(0, BLOCK_D)
    d_mask = d_range < D

    start = tl.maximum(0, i - W + 1)
    end = tl.minimum(L, i + W)

    q_ptr = Q + b * stride_qb + h * stride_qh + i * stride_ql + d_range * stride_qd
    q = tl.load(q_ptr, mask=d_mask, other=0.0)

    sqrt_d = tl.sqrt(D.to(tl.float32))
    m_i = -float("inf")

    for m_idx in range(M):
        meta_ptr = Meta + h * stride_mh + m_idx * stride_mm + d_range * stride_md
        meta = tl.load(meta_ptr, mask=d_mask, other=0.0)
        m_i = tl.maximum(m_i, tl.sum(q * meta) / sqrt_d)

    for t in range(start, end):
        k_ptr = K + b * stride_kb + h * stride_kh + t * stride_kl + d_range * stride_kd
        k = tl.load(k_ptr, mask=d_mask, other=0.0)
        m_i = tl.maximum(m_i, tl.sum(q * k) / sqrt_d)

    num = tl.zeros([BLOCK_D], dtype=tl.float32)
    den = 0.0

    for m_idx in range(M):
        meta_ptr = Meta + h * stride_mh + m_idx * stride_mm + d_range * stride_md
        meta = tl.load(meta_ptr, mask=d_mask, other=0.0)
        s = tl.sum(q * meta) / sqrt_d
        exp_s = tl.exp(s - m_i)
        den += exp_s

    for t in range(start, end):
        k_ptr = K + b * stride_kb + h * stride_kh + t * stride_kl + d_range * stride_kd
        v_ptr = V + b * stride_vb + h * stride_vh + t * stride_vl + d_range * stride_vd
        k = tl.load(k_ptr, mask=d_mask, other=0.0)
        v = tl.load(v_ptr, mask=d_mask, other=0.0)
        s = tl.sum(q * k) / sqrt_d
        exp_s = tl.exp(s - m_i)
        num += exp_s * v
        den += exp_s

    out = num / (den + 1e-8)
    tl.store(
        Out + b * stride_ob + h * stride_oh + i * stride_ol + d_range * stride_od,
        out,
        mask=d_mask,
    )


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
    b, h, i = tl.program_id(0), tl.program_id(1), tl.program_id(2)
    d_range = tl.arange(0, BLOCK_D)
    d_mask = d_range < D

    q = tl.load(
        Q + b * stride_qb + h * stride_qh + i * stride_ql + d_range * stride_qd,
        mask=d_mask,
        other=0.0,
    )

    num = tl.zeros([BLOCK_D], dtype=tl.float32)
    den = 0.0

    for t in range(L):
        cg = 1.0
        if t < i:
            for j in range(t + 1, i + 1):
                cg *= tl.load(Gamma + b * stride_gb + h * stride_gh + j * stride_gl)
        elif t > i:
            for j in range(i + 1, t + 1):
                cg *= tl.load(Gamma + b * stride_gb + h * stride_gh + j * stride_gl)

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

        qk = tl.sum(q * k)
        num += cg * qk * v
        den += cg * qk

    out = num / (den + 1e-8)
    tl.store(
        Out + b * stride_ob + h * stride_oh + i * stride_ol + d_range * stride_od,
        out,
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
        BD = triton.next_power_of_2(D)
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
            BLOCK_D=BD,
        )
        return out

    def fwd_gla_triton(self, q, k, v, x):
        B, H, L, D = q.shape
        gamma = torch.sigmoid((self.W_gamma * x).sum(-1, keepdim=True)).squeeze(-1)
        out = torch.empty_like(q)
        BD = triton.next_power_of_2(D)
        gla_kernel[(B, H, L)](
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
            BLOCK_D=BD,
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
        for i in range(L):
            qi, n, d = q[:, :, i : i + 1, :], 0.0, 0.0
            for t in range(L):
                if t < i:
                    cg = torch.prod(gamma[:, :, t + 1 : i + 1, :], dim=2, keepdim=True)
                elif t > i:
                    cg = torch.prod(gamma[:, :, i + 1 : t + 1, :], dim=2, keepdim=True)
                else:
                    cg = torch.ones(B, H, 1, 1, device=q.device)
                kt, vt = k[:, :, t : t + 1, :], v[:, :, t : t + 1, :]
                qk = (qi * kt).sum(-1, keepdim=True)
                n += cg * qk * vt
                d += cg * qk
            out[:, :, i : i + 1, :] = n / (d + 1e-8)
        return out


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--seq_len", type=int, default=128)
    parser.add_argument("--hidden_size", type=int, default=512)
    parser.add_argument("--num_heads", type=int, default=2)
    parser.add_argument("--dtype", type=str, default="float32")
    parser.add_argument("--device", type=str, default="cuda:0")
    args = parser.parse_args()

    dt = torch.float32
    model = LizardAttention(args.hidden_size, args.num_heads).to(args.device).to(dt)
    dh = args.hidden_size // args.num_heads
    q, k, v = [
        torch.randn(
            args.batch_size,
            args.num_heads,
            args.seq_len,
            dh,
            device=args.device,
            dtype=dt,
        )
        for _ in range(3)
    ]

    with torch.no_grad():
        o_py = model(q, k, v, use_triton=False)
        o_tr = model(q, k, v, use_triton=True)
        print(f"Max diff (FP32): {(o_py - o_tr).abs().max().item():.6e}")
