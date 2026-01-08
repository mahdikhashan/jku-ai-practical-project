import torch
import torch.nn as nn
import torch.nn.functional as F


class AnchorWindowAttention(nn.Module):
    def __init__(self, embed_dim, window_radius, num_meta_tokens):
        super().__init__()
        self.window_radius = window_radius
        self.meta_tokens = nn.Parameter(torch.randn(1, 1, num_meta_tokens, embed_dim))

    def forward(self, queries, keys, values):
        batch_size, num_heads, seq_len, embed_dim = queries.shape

        dim_tensor = torch.tensor(embed_dim, device=queries.device, dtype=queries.dtype)
        scale_factor = torch.sqrt(dim_tensor)

        meta_tokens_expanded = self.meta_tokens.expand(batch_size, num_heads, -1, -1)
        global_scores = (
            torch.matmul(queries, meta_tokens_expanded.transpose(-1, -2)) / scale_factor
        )

        window_diameter = 2 * self.window_radius - 1
        padding_amount = self.window_radius - 1

        keys_padded = F.pad(keys, (0, 0, padding_amount, padding_amount))
        values_padded = F.pad(values, (0, 0, padding_amount, padding_amount))

        keys_in_windows = keys_padded.unfold(dimension=2, size=window_diameter, step=1)
        values_in_windows = values_padded.unfold(
            dimension=2, size=window_diameter, step=1
        )

        local_scores = (
            torch.matmul(queries.unsqueeze(3), keys_in_windows).squeeze(3)
            / scale_factor
        )

        ones_mask = torch.ones((1, 1, seq_len, 1), device=queries.device)
        mask_padded = F.pad(ones_mask, (0, 0, padding_amount, padding_amount))

        valid_token_mask = mask_padded.unfold(2, window_diameter, 1).squeeze(
            -2
        )

        local_scores = local_scores.masked_fill(valid_token_mask == 0, float("-inf"))

        all_scores = torch.cat([local_scores, global_scores], dim=-1)
        max_score_val = all_scores.max(dim=-1, keepdim=True)[0]

        exp_local = torch.exp(local_scores - max_score_val)
        exp_global = torch.exp(global_scores - max_score_val)

        normalization_term = (
            exp_local.sum(dim=-1, keepdim=True)
            + exp_global.sum(dim=-1, keepdim=True)
            + 1e-8
        )

        numerator = torch.matmul(
            exp_local.unsqueeze(3), values_in_windows.transpose(-1, -2)
        ).squeeze(3)

        output = numerator / normalization_term

        return output
