import torch
import torch.nn as nn


class GatedLinearAttention(nn.Module):
    def __init__(self, d_model, num_heads, gate_weights_init):
        super().__init__()

    def forward(self, q, k, v):
        pass
