import torch
import torch.nn as nn
import torch.nn.functional as F


class GatedLinearAttention(nn.Module):
    def __init__(self, d_model, num_heads, computation_dtype=torch.float32):
        super().__init__()
        self.d_model = d_model
        self.num_heads = num_heads
        self.head_dim = d_model // num_heads
        self.computation_dtype = computation_dtype
        self.q_proj = nn.Linear(d_model, d_model, bias=False)
        self.k_proj = nn.Linear(d_model, d_model, bias=False)
        self.v_proj = nn.Linear(d_model, d_model, bias=False)
        self.g_proj = nn.Linear(d_model, num_heads, bias=True)
        self.out_proj = nn.Linear(d_model, d_model, bias=False)

    def forward(self, x):
        B, L, D = x.shape
        H = self.num_heads
        E = self.head_dim

        input_dtype = x.dtype

        q = self.q_proj(x).view(B, L, H, E).to(self.computation_dtype)
        k = self.k_proj(x).view(B, L, H, E).to(self.computation_dtype)
        v = self.v_proj(x).view(B, L, H, E).to(self.computation_dtype)

        g_logits = self.g_proj(x).view(B, L, H, 1).to(self.computation_dtype)
        log_g = F.logsigmoid(g_logits)

        lambda_t = log_g.cumsum(dim=1)

        q_hat = torch.cat([
            torch.exp(q + lambda_t),
            torch.exp(-q + lambda_t)
        ], dim=-1)

        k_hat = torch.cat([
            torch.exp(k - lambda_t),
            torch.exp(-k - lambda_t)
        ], dim=-1)

        attn_matrix = torch.einsum('b l h d, b s h d -> b h l s', q_hat, k_hat)

        mask = torch.tril(torch.ones(L, L, device=x.device, dtype=torch.bool))
        attn_matrix = attn_matrix.masked_fill(~mask, 0.0)

        output = torch.einsum('b h l s, b s h d -> b l h d', attn_matrix, v)

        output = output.reshape(B, L, D).to(input_dtype)

        return self.out_proj(output)
