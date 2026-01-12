import torch
import torch.nn as nn
import torch.nn.functional as F

class AnchorWindowAttention(nn.Module):
    def __init__(self, embed_dim, window_size, meta_token, dtype=None):
        super().__init__()
        self.window_size = window_size
        self.dtype = dtype
        
        self.anchor_proj = nn.Linear(embed_dim, meta_token, bias=False)
        
        if self.dtype is not None:
            self.anchor_proj.to(dtype=self.dtype)

    def forward(self, q, k, v):
        q, k, v = q.to(dtype=self.dtype), k.to(dtype=self.dtype), v.to(dtype=self.dtype)

        batch_size, num_heads, seq_len, embed_dim = q.shape
        dim_tensor = torch.tensor(embed_dim, device=q.device, dtype=q.dtype)
        scale_factor = torch.sqrt(dim_tensor)

        # [Batch, Heads, SeqLen, Num_Anchors]
        global_scores = self.anchor_proj(q) / scale_factor

        pad_left = self.window_size - 1
        pad_right = 0
        
        k_padded = F.pad(k, (0, 0, pad_left, pad_right))
        v_padded = F.pad(v, (0, 0, pad_left, pad_right))

        # Unfold
        k_windows = k_padded.unfold(dimension=2, size=self.window_size, step=1)
        v_windows = v_padded.unfold(dimension=2, size=self.window_size, step=1)

        local_scores = (
            torch.matmul(q.unsqueeze(3), k_windows).squeeze(3) 
            / scale_factor
        )

        # Masking
        ones_mask = torch.ones((1, 1, seq_len, 1), device=q.device)
        mask_padded = F.pad(ones_mask, (0, 0, pad_left, pad_right))
        valid_token_mask = mask_padded.unfold(2, self.window_size, 1).squeeze(-2)
        local_scores = local_scores.masked_fill(valid_token_mask == 0, float("-inf"))

        # Normalization
        all_scores = torch.cat([local_scores, global_scores], dim=-1)
        max_score_val = all_scores.max(dim=-1, keepdim=True)[0]

        exp_local = torch.exp(local_scores - max_score_val)
        exp_global = torch.exp(global_scores - max_score_val)

        # Denominator
        normalization_term = (
            exp_local.sum(dim=-1, keepdim=True)
            + exp_global.sum(dim=-1, keepdim=True)
            + 1e-8
        )

        # Numerator
        numerator = torch.matmul(
            exp_local.unsqueeze(3), 
            v_windows.transpose(-1, -2)
        ).squeeze(3)

        output = numerator / normalization_term
        return output
