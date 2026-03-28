import torch  # type: ignore
import torch.nn as nn  # type: ignore

from modules.lizard import AbstractLizardAttentionBlock

import triton

from kernels.awa.triton.fwd_kernel import awa_kernel
from kernels.gla.triton.fwd_kernel import gla_kernel


class LizardAttentionBlock(AbstractLizardAttentionBlock):
    def __init__(self, d_model, n_heads, window_size=64, alpha=1.0, m=4):
        super().__init__()
        self.d_model, self.n_heads = d_model, n_heads
        self.d_head = d_model // n_heads
        self.window_size, self.alpha, self.m = window_size, alpha, m

        self.meta_tokens = nn.Parameter(torch.randn(1, n_heads, m, self.d_head))
        self.W_gamma = nn.Parameter(torch.randn(1, n_heads, 1, self.d_head))

    def forward(self, q, k, v):
        g_out = self.fwd_gla_triton(q, k, v)
        a_out = self.fwd_awa_triton(q, k, v)
        return g_out + self.alpha * a_out

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

    def fwd_gla_triton(self, q, k, v):
        B, H, L, D = q.shape
        x = q
        gamma = torch.sigmoid((self.W_gamma * x).sum(-1, keepdim=True)).squeeze(-1)

        out = torch.empty_like(q)
        gla_kernel[(B, H)](
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
