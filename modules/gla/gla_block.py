# import torch  # type: ignore
# import torch.nn as nn  # type: ignore


# class GatedLinearAttention(nn.Module):
#     def __init__(self, d_model, num_heads, gate_weights_init):
#         super().__init__()
#         self.gate_weights = nn.Parameter(gate_weights_init)

#     def forward(self, queries, keys, values):
#         B, H, L, D = queries.shape

#         gate_logits = (self.gate_weights * queries).sum(-1, keepdim=True)
#         gate_values = torch.sigmoid(gate_logits)

#         activation_score = (queries * keys).sum(dim=-1, keepdim=True)
#         weighted_values = activation_score * values

#         log_gate = gate_values.log()
#         log_gate_cumsum = log_gate.cumsum(dim=-2)

#         decay_factor = log_gate_cumsum.exp()
#         decay_correction = (-log_gate_cumsum).exp()

#         numerator_accum = (weighted_values * decay_correction).cumsum(dim=-2)
#         numerator = numerator_accum * decay_factor

#         denominator_accum = (activation_score * decay_correction).cumsum(dim=-2)
#         denominator = denominator_accum * decay_factor

#         output = numerator / (denominator + 1e-8)

#         return output


import torch
import torch.nn as nn


class GatedLinearAttention(nn.Module):
    def __init__(self, d_model, num_heads, gate_weights_init):
        super().__init__()
        self.gate_weights = nn.Parameter(gate_weights_init)

    def forward(self, queries, keys, values):
        B, H, L, D = queries.shape

        # 1. Compute Gate
        gate_logits = (self.gate_weights * queries).sum(-1, keepdim=True)
        gate_values = torch.sigmoid(gate_logits)

        # 2. Compute Update
        activation_score = (queries * keys).sum(dim=-1, keepdim=True)
        weighted_values = activation_score * values

        # --- FIX STARTS HERE ---
        # We perform the Scan (Cumsum) in float32 for:
        # 1. Compatibility (fixes "cumsum not implemented for Half" on CPU)
        # 2. Stability (prevents overflow/underflow in log-space)

        # Cast to float32
        log_gate = gate_values.log().to(torch.float32)
        log_gate_cumsum = log_gate.cumsum(dim=-2)

        decay_factor = log_gate_cumsum.exp()
        decay_correction = (-log_gate_cumsum).exp()

        # Weighted values must also be float32 for the accumulation
        weighted_values_f32 = weighted_values.to(torch.float32)
        activation_score_f32 = activation_score.to(torch.float32)

        numerator_accum = (weighted_values_f32 * decay_correction).cumsum(dim=-2)
        numerator = numerator_accum * decay_factor

        denominator_accum = (activation_score_f32 * decay_correction).cumsum(dim=-2)
        denominator = denominator_accum * decay_factor

        output = numerator / (denominator + 1e-8)

        # Cast back to original dtype (e.g., float16) before returning
        return output.to(queries.dtype)
        # --- FIX ENDS HERE ---
