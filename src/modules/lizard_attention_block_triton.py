import torch  # type: ignore
import torch.nn as nn  # type: ignore
import triton

from kernels.awa.triton.fwd_kernel import awa_kernel
from kernels.gla.triton.fwd_kernel import parallel_gla_kernel
from modules.lizard import AbstractLizardAttentionBlock


class LizardAttentionBlock(AbstractLizardAttentionBlock):
    def __init__(self, d_model, n_heads, window_size=64, alpha=1.0, m=4):
        super().__init__()
        self.d_model = d_model
        self.n_heads = n_heads
        self.d_head = d_model // n_heads
        self.window_size = window_size
        self.alpha = alpha
        self.m = m

        # learnable Anchor Keys
        self.anchor_proj = nn.Linear(self.d_head, m, bias=False)

    def forward(self, q, k, v):
        g_out = self.fwd_gla_triton(q, k, v)
        a_out = self.fwd_awa_triton(q, k, v)
        return g_out + self.alpha * a_out

    def fwd_awa_triton(self, q, k, v):
        q, k, v = q.contiguous(), k.contiguous(), v.contiguous()
        
        B, H, L, D = q.shape
        out = torch.empty_like(q)
        
        meta_weight = self.anchor_proj.weight 
        stride_m, stride_d = meta_weight.stride()

        grid = (B, H, L)
        
        awa_kernel[grid](
            q, k, v, meta_weight, out,            
            *q.stride(),
            *k.stride(),
            *v.stride(),
            0,          # stride_mb: Batch -> Broadcast (0)
            0,          # stride_mh: Head -> Broadcast (0)
            stride_m,   # stride_mm: Meta Token Dim
            stride_d,   # stride_md: Embedding Dim
            *out.stride(),
            # Dimensions
            B, H, L, D, 
            self.m,             # M (Num Meta Tokens)
            self.window_size,   # W (Causal Window Size)
            # Meta-Parameters
            BLOCK_D=triton.next_power_of_2(D),
        )
        return out

    def fwd_gla_triton(self, q, k, v):
        q, k, v = q.contiguous(), k.contiguous(), v.contiguous()
        
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
            D_QK=D_QK,
            D_V=D_V,
            ALLOW_TF32=True
        )

        return output
