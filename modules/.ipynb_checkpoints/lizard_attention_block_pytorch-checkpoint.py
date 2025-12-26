import torch
import torch.nn as nn

from modules.lizard import AbstractLizardAttentionBlock

from modules.awa.awa_block import AnchorWindowAttention
from modules.gla.gla_block import GatedLinearAttention


class LizardAttentionBlockPyTorch(AbstractLizardAttentionBlock):
    def __init__(self, d_model, n_heads, window_size=64, alpha=1.0, m=4):
        super().__init__()
        self.alpha = alpha
        self.d_head = d_model // n_heads

        w_gamma_init = torch.randn(1, n_heads, 1, self.d_head)
        self.gla_module = GatedLinearAttention(d_model, n_heads, w_gamma_init)

        self.awa_module = AnchorWindowAttention(
            embed_dim=self.d_head, window_radius=window_size, num_meta_tokens=m
        )

    def forward(self, q, k, v):
        g_out = self.gla_module(q, k, v)
        a_out = self.awa_module(q, k, v)

        return g_out + self.alpha * a_out


LizardAttentionBlock = LizardAttentionBlockPyTorch
