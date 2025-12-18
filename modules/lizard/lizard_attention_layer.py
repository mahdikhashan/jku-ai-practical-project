import math

import torch  # type: ignore #
import torch.nn as nn  # type: ignore
import torch.nn.functional as F  # type: ignore

from lizard import LizardModule


class LizardAttention(LizardModule):
    def __init__(self, d_model, n_heads, window_size=64, alpha=1.0, m=4):
        self.d_model = d_model
        self.n_heads = n_heads
        self.d_head = d_model // n_heads
        self.window_size = window_size
        self.alpha = alpha
        self.m = m
        self.meta_tokens = nn.Parameter(torch.randn(1, n_heads, m, self.d_head))
        self.W_gamma = nn.Parameter(torch.randn(1, n_heads, 1, self.d_head))
        self.phi_q = nn.Identity()
        self.phi_k = nn.Identity()

    # @benchmark(warmup_iterations=0, benchmark_iterations=1, save_results=True)
    def forward(self, q, k, v, x=None, alpha=None, triton_kernel=False):
        if alpha is None:
            alpha = self.alpha

        if x is None:
            x = q  # Use queries as input for gating if not provided

        if not triton_kernel:
            gla_out = self.gla_fwd(q, k, v, x)
            awa_out = self.awa_fwd(q, k, v)
            result = gla_out + alpha * awa_out

            # # Free memory after computation
            # del gla_out, awa_out
            # if torch.cuda.is_available():
            #     torch.cuda.empty_cache()
            # gc.collect()

            return result

        raise NotImplementedError("custom kernel is not implemented yet!")

    def awa_fwd(
        self,
        q,
        k,
        v,
    ):
        batch, heads, seq_len, d_head = q.shape

        # Expand meta-tokens for batch: (batch, heads, m, d_head)
        meta_tokens = self.meta_tokens.expand(batch, -1, -1, -1)

        out = torch.zeros_like(q)

        for i in range(seq_len):
            # Define window boundaries: [i-w+1, i] for causal
            start = max(0, i - self.window_size + 1)
            end = min(seq_len, i + self.window_size)

            # Extract current query and window
            q_i = q[:, :, i : i + 1, :]  # (batch, heads, 1, d_head)
            k_window = k[:, :, start:end, :]  # (batch, heads, window_len, d_head)
            v_window = v[:, :, start:end, :]  # (batch, heads, window_len, d_head)

            # Compute attention scores with keys in window: exp(q_i^T k_t / √d)
            scores_k = torch.matmul(q_i, k_window.transpose(-2, -1)) / math.sqrt(d_head)
            exp_scores_k = torch.exp(scores_k)  # (batch, heads, 1, window_len)

            # Compute attention scores with meta-tokens: exp(q_i^T t_j / √d)
            scores_t = torch.matmul(q_i, meta_tokens.transpose(-2, -1)) / math.sqrt(
                d_head
            )
            exp_scores_t = torch.exp(scores_t)  # (batch, heads, 1, m)

            # Denominator: Σ[j=0 to m-1] exp(q_i^T t_j / √d) + Σ[t=i-w+1 to i] exp(q_i^T k_t / √d)
            sum_meta = exp_scores_t.sum(dim=-1, keepdim=True)  # (batch, heads, 1, 1)
            sum_keys = exp_scores_k.sum(dim=-1, keepdim=True)  # (batch, heads, 1, 1)
            denominator = sum_meta + sum_keys  # (batch, heads, 1, 1)

            # Numerator: Σ[t=i-w+1 to i] exp(q_i^T k_t / √d) v_t
            numerator = torch.matmul(
                exp_scores_k, v_window
            )  # (batch, heads, 1, d_head)

            # Final output: ŷ_i = numerator / denominator
            out[:, :, i : i + 1, :] = numerator / (
                denominator + 1e-8
            )  # Add epsilon for numerical stability

        return out

    def gla_fwd(self, q, k, v, x):
        batch, heads, seq_len, d_head = q.shape

        # feature maps, identity
        # todo(mahdi): hedgehog
        q_feat = self.phi_q(q)  # (batch, heads, seq_len, d_head)
        k_feat = self.phi_k(k)  # (batch, heads, seq_len, d_head)

        # gating factors
        # W_gamma: (1, heads, 1, d_head), x: (batch, heads, seq_len, d_head)
        gamma = torch.sigmoid(
            (self.W_gamma * x).sum(dim=-1, keepdim=True)
        )  # (batch, heads, seq_len, 1)

        out = torch.zeros_like(q)

        # each position attends to all positions (non-causal)
        for i in range(seq_len):
            numerator = torch.zeros(
                batch, heads, 1, d_head, device=q.device, dtype=q.dtype
            )
            denominator = torch.zeros(
                batch, heads, 1, 1, device=q.device, dtype=q.dtype
            )

            q_i = q_feat[:, :, i : i + 1, :]

            for t in range(seq_len):
                # Compute gating product based on relative positions
                if t < i:
                    # Product from t+1 to i
                    cum_gamma = torch.prod(
                        gamma[:, :, t + 1 : i + 1, :], dim=2, keepdim=True
                    )
                elif t > i:
                    # Product from i+1 to t
                    cum_gamma = torch.prod(
                        gamma[:, :, i + 1 : t + 1, :], dim=2, keepdim=True
                    )
                else:
                    cum_gamma = torch.ones(
                        batch, heads, 1, 1, device=q.device, dtype=q.dtype
                    )

                k_t = k_feat[:, :, t : t + 1, :]
                v_t = v[:, :, t : t + 1, :]

                kv_t = k_t.transpose(-2, -1) @ v_t
                numerator += cum_gamma.squeeze(-1).unsqueeze(-1) * (q_i @ kv_t).squeeze(
                    2
                ).unsqueeze(2)

                qk_t = q_i @ k_t.transpose(-2, -1)
                denominator += cum_gamma * qk_t

            out[:, :, i : i + 1, :] = numerator / (denominator + 1e-8)

        return out


if __name__ == "__main__":
    if not torch.cuda.is_available():
        raise Exception(
            """
            to run this experiement, you need a cuda enabled device!
            failed!
            """
        )

    import argparse

    parser = argparse.ArgumentParser(description="Configuration for Model Parameters")
    parser.add_argument(
        "--batch_size", type=int, default=1, help="Batch size (default: 1)"
    )
    parser.add_argument(
        "--seq_len", type=int, default=128, help="Sequence length (default: 128)"
    )
    parser.add_argument(
        "--hidden_size", type=int, default=512, help="Hidden size (default: 512)"
    )
    parser.add_argument(
        "--num_heads",
        type=int,
        default=2,
        help="Number of attention heads (default: 2)",
    )
    parser.add_argument(
        "--dtype",
        type=str,
        default="float16",
        choices=["float32", "float16", "bfloat16"],
        help="Data type (default: bfloat16)",
    )
    parser.add_argument(
        "--device", type=str, default="cuda:0", help="Device to use (default: 'cuda:0')"
    )
    parser.add_argument("--experiment_name", type=str, default="Manual_Run")
    parser.add_argument(
        "--alpha", type=float, default=1.0, help="awa attention ratio (default: 1.0)"
    )
    parser.add_argument(
        "--m", type=int, default=4, help="Number of meta-tokens (default: 4)"
    )
    parser.add_argument(
        "--window_size", type=int, default=64, help="Window size (default: 64)"
    )

    args = parser.parse_args()

    batch_size = int(args.batch_size)
    sequence_length = int(args.seq_len)
    hidden_size = int(args.hidden_size)
    num_heads = int(args.num_heads)
    device = args.device
    dtype = args.dtype
    window_size = args.window_size
    m = args.m
    alpha = args.alpha

    print(f"Batch Size:  {batch_size}")
    print(f"Sequence Length:     {sequence_length}")
    print(f"Hidden Size: {hidden_size}")
    print(f"Num Heads:   {num_heads}")
    print(f"Dtype:       {dtype}")
    print(f"Device:      {device}")

    dtype_map = {
        "float32": torch.float32,
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
    }
    _dtype = dtype_map[dtype]

    model = (
        LizardAttention(
            d_model=hidden_size,
            n_heads=num_heads,
            window_size=window_size,
            alpha=alpha,
            m=m,
        )
        .to(device)
        .to(_dtype)
    )

    # inputs
    d_head = hidden_size // num_heads
    q = (
        torch.randn(batch_size, num_heads, sequence_length, d_head)
        .to(device)
        .to(_dtype)
    )
    k = (
        torch.randn(batch_size, num_heads, sequence_length, d_head)
        .to(device)
        .to(_dtype)
    )
    v = (
        torch.randn(batch_size, num_heads, sequence_length, d_head)
        .to(device)
        .to(_dtype)
    )

    output = model(q, k, v)
    print(f"shape: {output.shape}")
