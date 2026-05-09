import math
import torch


def swa_naive(q, k, v, window_sizes: tuple[int, int] = (15, 16)):
    """
    Naive implementation of sliding window attention.

    Parameters
    ----------
    q : (..., L, Dqk) torch.Tensor
        Queries to compute attention with.
    k : (..., L, Dqk) torch.Tensor
        Keys to compute attention with.
    v : (..., L, Dv) torch.Tensor
        Values to compute attention with.
    window_sizes : tuple[int, int]
        The number of time-steps to look back- and forward, in that order.
        Causal masking is enabled when the second value is zero.

    Returns
    -------
    h : (..., L, Dv) torch.Tensor
        The computed output sequence.
    """
    bwd_win_size, fwd_win_size = window_sizes

    row_idx = torch.arange(q.shape[-2], device=q.device).unsqueeze(-1)
    col_idx = torch.arange(k.shape[-2], device=k.device).unsqueeze(-2)
    fwd_inv_mask = row_idx < col_idx - fwd_win_size
    bwd_inv_mask = row_idx > col_idx + bwd_win_size

    qk = q @ k.transpose(-1, -2)
    qk = qk / math.sqrt(q.shape[-1])
    qk_masked = torch.masked_fill(qk, fwd_inv_mask | bwd_inv_mask, -float("inf"))
    a = torch.softmax(qk_masked, dim=-1)
    return a @ v
