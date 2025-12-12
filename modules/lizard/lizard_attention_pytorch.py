import torch  # type: ignore
import torch.nn as nn  # type: ignore
import torch.nn.functional as F  # type: ignore
import math


class LizardAttention(nn.Module):
    def __init__(self, d_model, n_heads, window_size=64, chunk_size=64, alpha=1, m=4):
        super().__init__()
        self.d_model = d_model
        self.n_heads = n_heads
        self.d_head = d_model // n_heads
        self.window_size = window_size
        self.chunk_size = chunk_size
        self.alpha = alpha
        self.m = m  # number of meta-tokens

        # Meta-tokens for AWA (learnable)
        self.meta_tokens = nn.Parameter(torch.randn(1, n_heads, m, self.d_head))

    def forward(
        self,
        q,
        k,
        v,
        g=None,
        alpha=None,
        triton_kernel=False,
        causal=True,
        use_softmax=True,
        chunked=True,
    ):
        if alpha is None:
            alpha = self.alpha

        if not triton_kernel:
            gla_out = self.gla_fwd(
                q, k, v, causal=causal, use_softmax=use_softmax, chunked=chunked
            )
            awa_out = self.awa_fwd(q, k, v, causal=causal)
            return gla_out + alpha * awa_out

        raise NotImplementedError("custom kernel is not implemented yet!")

    def awa_fwd(self, q, k, v, causal=True):
        """
        Anchor Window Attention with meta-tokens.
        Formula: ŷ_i = Σ[t=i-w+1 to i] exp(q_i^T k_t / √d) v_t / (Σ[j=0 to m-1] t_j + Σ[t=i-w+1 to i] exp(q_i^T k_t / √d))

        Args:
            q, k, v: (batch, heads, seq_len, d_head)
            causal: if True, apply causal masking

        Returns:
            out: (batch, heads, seq_len, d_head)
        """
        batch, heads, seq_len, d_head = q.shape

        # Expand meta-tokens for batch: (batch, heads, m, d_head)
        meta_tokens = self.meta_tokens.expand(batch, -1, -1, -1)

        # Output tensor
        out = torch.zeros_like(q)

        for i in range(seq_len):
            # Define window boundaries: [i-w+1, i] for causal
            start = max(0, i - self.window_size + 1)
            end = i + 1 if causal else min(seq_len, i + self.window_size)

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

    def gla_fwd(self, q, k, v, causal=True, use_softmax=False, chunked=True):
        """
        Gated Linear Attention.

        Args:
            q, k, v: (batch, heads, seq_len, d_head)
            causal: if True, apply causal masking
            use_softmax: if True, use standard attention; if False, use linear attention
            chunked: if True, process in chunks; if False, process entire sequence

        Returns:
            out: (batch, heads, seq_len, d_head)
        """
        if chunked:
            return self._gla_chunked(q, k, v, causal, use_softmax)
        else:
            return self._gla_full(q, k, v, causal, use_softmax)

    def _gla_full(self, q, k, v, causal, use_softmax):
        """Full sequence GLA without chunking."""
        batch, heads, seq_len, d_head = q.shape

        if use_softmax:
            # Standard attention with optional causal mask
            scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(d_head)

            if causal:
                causal_mask = torch.triu(
                    torch.ones(seq_len, seq_len, device=q.device), diagonal=1
                ).bool()
                scores = scores.masked_fill(causal_mask, float("-inf"))

            attn_weights = F.softmax(scores, dim=-1)
            out = torch.matmul(attn_weights, v)
        else:
            # Linear attention: O(N*d^2) instead of O(N^2*d)
            if causal:
                # Causal linear attention using recurrence
                out = torch.zeros_like(q)
                state = torch.zeros(batch, heads, d_head, d_head, device=q.device)

                for t in range(seq_len):
                    k_t = k[:, :, t : t + 1, :]  # (batch, heads, 1, d_head)
                    v_t = v[:, :, t : t + 1, :]  # (batch, heads, 1, d_head)
                    q_t = q[:, :, t : t + 1, :]  # (batch, heads, 1, d_head)

                    # Update state: state = state + k_t^T @ v_t
                    state = (
                        state + k_t.transpose(-2, -1) @ v_t
                    )  # (batch, heads, d_head, d_head)

                    # Compute output: o_t = q_t @ state
                    out[:, :, t : t + 1, :] = q_t @ state  # (batch, heads, 1, d_head)
            else:
                # Non-causal linear attention
                # KV = K^T @ V, out = Q @ KV
                kv = k.transpose(-2, -1) @ v  # (batch, heads, d_head, d_head)
                out = q @ kv  # (batch, heads, seq_len, d_head)

        return out

    def _gla_chunked(self, q, k, v, causal, use_softmax):
        """Chunked GLA processing."""
        batch, heads, seq_len, d_head = q.shape
        num_chunks = (seq_len + self.chunk_size - 1) // self.chunk_size

        out = torch.zeros_like(q)
        state = torch.zeros(batch, heads, d_head, d_head, device=q.device)

        for chunk_idx in range(num_chunks):
            start = chunk_idx * self.chunk_size
            end = min(start + self.chunk_size, seq_len)
            chunk_len = end - start

            q_chunk = q[:, :, start:end, :]
            k_chunk = k[:, :, start:end, :]
            v_chunk = v[:, :, start:end, :]

            if use_softmax:
                # Standard attention within chunk
                scores = torch.matmul(q_chunk, k_chunk.transpose(-2, -1)) / math.sqrt(
                    d_head
                )

                if causal:
                    # Causal mask within chunk
                    causal_mask = torch.triu(
                        torch.ones(chunk_len, chunk_len, device=q.device), diagonal=1
                    ).bool()
                    scores = scores.masked_fill(causal_mask, float("-inf"))

                # Attention to previous state (if causal and not first chunk)
                if causal and chunk_idx > 0:
                    # Query attends to accumulated state from previous chunks
                    state_contrib = q_chunk @ state  # (batch, heads, chunk_len, d_head)
                else:
                    state_contrib = 0

                attn_weights = F.softmax(scores, dim=-1)
                chunk_out = torch.matmul(attn_weights, v_chunk) + state_contrib
            else:
                # Linear attention within chunk
                if causal:
                    chunk_out = torch.zeros_like(q_chunk)
                    for t in range(chunk_len):
                        k_t = k_chunk[:, :, t : t + 1, :]
                        v_t = v_chunk[:, :, t : t + 1, :]
                        q_t = q_chunk[:, :, t : t + 1, :]

                        # Output from current state
                        chunk_out[:, :, t : t + 1, :] = q_t @ state

                        # Update state
                        state = state + k_t.transpose(-2, -1) @ v_t
                else:
                    # Non-causal: process entire chunk and update state
                    kv_chunk = k_chunk.transpose(-2, -1) @ v_chunk
                    chunk_out = q_chunk @ (state + kv_chunk)
                    state = state + kv_chunk

            out[:, :, start:end, :] = chunk_out

            # Update state for next chunk (for causal linear attention)
            if causal and not use_softmax:
                # State already updated in the loop above
                pass
            elif not causal and not use_softmax:
                # State already updated above
                pass

        return out


# Example usage
if __name__ == "__main__":
    batch_size = 2
    n_heads = 8
    seq_len = 128
    d_model = 512

    # Create model
    model = LizardAttention(
        d_model=d_model,
        n_heads=n_heads,
        window_size=64,
        chunk_size=64,
        alpha=1.0,  # type: ignore
        m=4,
    )

    # Create dummy inputs
    d_head = d_model // n_heads
    q = torch.randn(batch_size, n_heads, seq_len, d_head)
    k = torch.randn(batch_size, n_heads, seq_len, d_head)
    v = torch.randn(batch_size, n_heads, seq_len, d_head)

    # Test different configurations
    print("Testing causal + softmax + chunked:")
    out1 = model(q, k, v, causal=True, use_softmax=True, chunked=True)
    print(f"Output shape: {out1.shape}")

    print("\nTesting causal + linear + chunked:")
    out2 = model(q, k, v, causal=True, use_softmax=False, chunked=True)
    print(f"Output shape: {out2.shape}")

    print("\nTesting non-causal + linear + full:")
    out3 = model(q, k, v, causal=False, use_softmax=False, chunked=False)
    print(f"Output shape: {out3.shape}")
