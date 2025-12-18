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

    start = tl.maximum(0, i - W + 1)
    end = tl.minimum(L, i + W)

    d_range = tl.arange(0, BLOCK_D)
    d_mask = d_range < D

    q_ptr = Q + b * stride_qb + h * stride_qh + i * stride_ql + d_range * stride_qd
    q = tl.load(q_ptr, mask=d_mask, other=0.0).to(tl.float32)

    sqrt_d = tl.sqrt(D.to(tl.float32))
    m_i = -float("inf")

    for m_idx in range(M):
        meta_ptr = Meta + h * stride_mh + m_idx * stride_mm + d_range * stride_md
        meta = tl.load(meta_ptr, mask=d_mask, other=0.0).to(tl.float32)
        score = tl.sum(q * meta) / sqrt_d
        m_i = tl.maximum(m_i, score)

    for t in range(start, end):
        k_ptr = K + b * stride_kb + h * stride_kh + t * stride_kl + d_range * stride_kd
        k = tl.load(k_ptr, mask=d_mask, other=0.0).to(tl.float32)
        score = tl.sum(q * k) / sqrt_d
        m_i = tl.maximum(m_i, score)

    num = tl.zeros([BLOCK_D], dtype=tl.float32)
    denom = 0.0

    for m_idx in range(M):
        meta_ptr = Meta + h * stride_mh + m_idx * stride_mm + d_range * stride_md
        meta = tl.load(meta_ptr, mask=d_mask, other=0.0).to(tl.float32)
        score = tl.sum(q * meta) / sqrt_d
        exp_score = tl.exp(score - m_i)
        denom += exp_score

    for t in range(start, end):
        k_ptr = K + b * stride_kb + h * stride_kh + t * stride_kl + d_range * stride_kd
        v_ptr = V + b * stride_vb + h * stride_vh + t * stride_vl + d_range * stride_vd
        k = tl.load(k_ptr, mask=d_mask, other=0.0).to(tl.float32)
        v = tl.load(v_ptr, mask=d_mask, other=0.0).to(tl.float32)

        score = tl.sum(q * k) / sqrt_d
        exp_score = tl.exp(score - m_i)
        num += exp_score * v
        denom += exp_score

    out = num / (denom + 1e-8)
    out_ptr = Out + b * stride_ob + h * stride_oh + i * stride_ol + d_range * stride_od
    tl.store(out_ptr, out.to(Out.dtype.element_ty), mask=d_mask)


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
    b = tl.program_id(0)
    h = tl.program_id(1)
    i = tl.program_id(2)

    d_range = tl.arange(0, BLOCK_D)
    d_mask = d_range < D
    q_ptr = Q + b * stride_qb + h * stride_qh + i * stride_ql + d_range * stride_qd
    q = tl.load(q_ptr, mask=d_mask, other=0.0).to(tl.float32)

    num = tl.zeros([BLOCK_D], dtype=tl.float32)
    denom = 0.0

    for t in range(L):
        log_cum_gamma = 0.0
        if t < i:
            for j in range(t + 1, i + 1):
                g = tl.load(Gamma + b * stride_gb + h * stride_gh + j * stride_gl).to(
                    tl.float32
                )
                log_cum_gamma += tl.log(tl.maximum(g, 1e-12))
        elif t > i:
            for j in range(i + 1, t + 1):
                g = tl.load(Gamma + b * stride_gb + h * stride_gh + j * stride_gl).to(
                    tl.float32
                )
                log_cum_gamma += tl.log(tl.maximum(g, 1e-12))

        cum_gamma = tl.exp(log_cum_gamma)

        k = tl.load(
            K + b * stride_kb + h * stride_kh + t * stride_kl + d_range * stride_kd,
            mask=d_mask,
        ).to(tl.float32)
        v = tl.load(
            V + b * stride_vb + h * stride_vh + t * stride_vl + d_range * stride_vd,
            mask=d_mask,
        ).to(tl.float32)

        qk = tl.sum(q * k)
        num += cum_gamma * qk * v
        denom += cum_gamma * qk

    out = num / (denom + 1e-8)
    out_ptr = Out + b * stride_ob + h * stride_oh + i * stride_ol + d_range * stride_od
    tl.store(out_ptr, out.to(Out.dtype.element_ty), mask=d_mask)


class LizardAttention(nn.Module):
    def __init__(self, d_model, n_heads, window_size=64, alpha=1.0, m=4):
        super().__init__()
        self.d_model = d_model
        self.n_heads = n_heads
        self.d_head = d_model // n_heads
        self.window_size = window_size
        self.alpha = alpha
        self.m = m
        self.meta_tokens = nn.Parameter(torch.randn(1, n_heads, m, self.d_head))
        self.W_gamma = nn.Parameter(torch.randn(1, n_heads, 1, self.d_head))

    def forward(self, q, k, v, x=None, alpha=None, use_triton=False):
        alpha = alpha if alpha is not None else self.alpha
        x = x if x is not None else q
        if use_triton:
            gla_out = self.fwd_gla_triton(q, k, v, x)
            awa_out = self.fwd_awa_triton(q, k, v)
        else:
            gla_out = self.fwd_gla(q, k, v, x)
            awa_out = self.fwd_awa(q, k, v)
        return gla_out + alpha * awa_out

    def fwd_awa_triton(self, q, k, v):
        B, H, L, D = q.shape
        out = torch.empty_like(q)
        BLOCK_D = triton.next_power_of_2(D)
        grid = (B, H, L)
        awa_kernel[grid](
            q,
            k,
            v,
            self.meta_tokens,
            out,
            q.stride(0),
            q.stride(1),
            q.stride(2),
            q.stride(3),
            k.stride(0),
            k.stride(1),
            k.stride(2),
            k.stride(3),
            v.stride(0),
            v.stride(1),
            v.stride(2),
            v.stride(3),
            self.meta_tokens.stride(0),
            self.meta_tokens.stride(1),
            self.meta_tokens.stride(2),
            self.meta_tokens.stride(3),
            out.stride(0),
            out.stride(1),
            out.stride(2),
            out.stride(3),
            B,
            H,
            L,
            D,
            self.m,
            self.window_size,
            BLOCK_D=BLOCK_D,
        )
        return out

    def fwd_gla_triton(self, q, k, v, x):
        B, H, L, D = q.shape
        gamma = torch.sigmoid((self.W_gamma * x).sum(dim=-1, keepdim=True)).squeeze(-1)
        out = torch.empty_like(q)
        BLOCK_D = triton.next_power_of_2(D)
        grid = (B, H, L)
        gla_kernel[grid](
            q,
            k,
            v,
            gamma,
            out,
            q.stride(0),
            q.stride(1),
            q.stride(2),
            q.stride(3),
            k.stride(0),
            k.stride(1),
            k.stride(2),
            k.stride(3),
            v.stride(0),
            v.stride(1),
            v.stride(2),
            v.stride(3),
            gamma.stride(0),
            gamma.stride(1),
            gamma.stride(2),
            out.stride(0),
            out.stride(1),
            out.stride(2),
            out.stride(3),
            B,
            H,
            L,
            D,
            BLOCK_D=BLOCK_D,
        )
        return out

    def fwd_awa(self, q, k, v):
        B, H, L, D = q.shape
        meta_tokens = self.meta_tokens.expand(B, -1, -1, -1)
        out = torch.zeros_like(q)
        for i in range(L):
            start, end = max(0, i - self.window_size + 1), min(L, i + self.window_size)
            q_i = q[:, :, i : i + 1, :].to(torch.float32)
            k_win, v_win = k[:, :, start:end, :].to(torch.float32), v[
                :, :, start:end, :
            ].to(torch.float32)
            m_win = meta_tokens.to(torch.float32)
            s_k = torch.matmul(q_i, k_win.transpose(-2, -1)) / math.sqrt(D)
            s_t = torch.matmul(q_i, m_win.transpose(-2, -1)) / math.sqrt(D)
            mv = torch.max(torch.cat([s_k, s_t], dim=-1), dim=-1, keepdim=True)[0]
            ek, et = torch.exp(s_k - mv), torch.exp(s_t - mv)
            out[:, :, i : i + 1, :] = (
                torch.matmul(ek, v_win) / (ek.sum(-1, True) + et.sum(-1, True) + 1e-8)
            ).to(q.dtype)
        return out

    def fwd_gla(self, q, k, v, x):
        B, H, L, D = q.shape
        gamma = torch.sigmoid((self.W_gamma * x).sum(dim=-1, keepdim=True))
        out = torch.zeros_like(q)
        for i in range(L):
            qi, n, d = q[:, :, i : i + 1, :].to(torch.float32), 0.0, 0.0
            for t in range(L):
                if t < i:
                    cg = torch.prod(gamma[:, :, t + 1 : i + 1, :], dim=2, keepdim=True)
                elif t > i:
                    cg = torch.prod(gamma[:, :, i + 1 : t + 1, :], dim=2, keepdim=True)
                else:
                    cg = torch.ones(B, H, 1, 1, device=q.device)
                kt, vt = k[:, :, t : t + 1, :].to(torch.float32), v[
                    :, :, t : t + 1, :
                ].to(torch.float32)
                qk = (qi * kt).sum(-1, keepdim=True)
                n += cg.to(torch.float32) * qk * vt
                d += cg.to(torch.float32) * qk
            out[:, :, i : i + 1, :] = (n / (d + 1e-8)).to(q.dtype)
        return out


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--seq_len", type=int, default=128)
    parser.add_argument("--hidden_size", type=int, default=512)
    parser.add_argument("--num_heads", type=int, default=2)
    parser.add_argument("--dtype", type=str, default="bfloat16")
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--use_triton", action="store_true")
    args = parser.parse_args()

    dt = {
        "float32": torch.float32,
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
    }[args.dtype]
    model = LizardAttention(args.hidden_size, args.num_heads).to(args.device).to(dt)
    dh = args.hidden_size // args.num_heads
    q = (
        torch.randn(args.batch_size, args.num_heads, args.seq_len, dh)
        .to(args.device)
        .to(dt)
    )
    k, v = torch.randn_like(q), torch.randn_like(q)

    with torch.no_grad():
        o_py, o_tr = model(q, k, v, use_triton=False), model(q, k, v, use_triton=True)
        print(f"Max diff: {(o_py - o_tr).abs().max().item():.6f}")
