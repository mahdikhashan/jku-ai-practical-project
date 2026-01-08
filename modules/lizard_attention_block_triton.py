import torch  # type: ignore
import torch.nn as nn  # type: ignore
import triton

from kernels.awa.triton.fwd_kernel import awa_kernel
from kernels.gla.triton.fwd_kernel import parallel_gla_kernel
from modules.lizard import AbstractLizardAttentionBlock


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
        B, L, H, D_QK = q.shape
        _, _, _, D_V = v.shape

        output = torch.empty_like(v)

        BLOCK_M = 64
        BLOCK_N = 64

        grid = (triton.cdiv(L, BLOCK_M), B * H)

        parallel_gla_kernel[grid](
            q, k, v, output,
            q.stride(0), q.stride(1), q.stride(2), q.stride(3),
            k.stride(0), k.stride(1), k.stride(2), k.stride(3),
            v.stride(0), v.stride(1), v.stride(2), v.stride(3),
            output.stride(0), output.stride(1), output.stride(2), output.stride(3),
            B, L, H,
            BLOCK_M=BLOCK_M,
            BLOCK_N=BLOCK_N,
            BLOCK_D_QK=D_QK,
            BLOCK_D_V=D_V,
        )

        return output
