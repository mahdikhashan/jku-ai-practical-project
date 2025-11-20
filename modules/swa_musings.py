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
    qk_masked = torch.masked_fill(qk, fwd_inv_mask | bwd_inv_mask, -float("inf"))
    a = torch.softmax(qk_masked, dim=-1)
    return a @ v


def swa_strided(q, k, v, window_sizes: tuple[int, int] = (15, 16)):
    """
    Sliding window attention implemented using striding tricks.
    """
    bwd_win_size, fwd_win_size = window_sizes
    win_size = 1 + bwd_win_size + fwd_win_size
    assert win_size < k.shape[-2]

    k_to_pad = [k.ravel()]
    v_to_pad = [v.ravel()]
    if bwd_win_size > 0:
        k_to_pad.insert(0, torch.zeros(bwd_win_size * k.shape[-1], device=k.device, dtype=k.dtype))
        v_to_pad.insert(0, torch.zeros(bwd_win_size * v.shape[-1], device=v.device, dtype=v.dtype))
    if fwd_win_size > 0:
        k_to_pad.append(torch.zeros(fwd_win_size * k.shape[-1], device=k.device, dtype=k.dtype))
        v_to_pad.append(torch.zeros(fwd_win_size * v.shape[-1], device=v.device, dtype=v.dtype))

    k_strided = torch.as_strided(
        torch.cat(k_to_pad, dim=0),
        size=(*k.shape[:-1], win_size, k.shape[-1]),
        stride=(*k.stride()[:-1], k.stride(-2), k.stride(-1))
    )
    v_strided = torch.as_strided(
        torch.cat(v_to_pad, dim=0),
        size=(*v.shape[:-1], win_size, v.shape[-1]),
        stride=(*v.stride()[:-1], v.stride(-2), v.stride(-1))
    )

    row_idx = torch.arange(q.shape[-2], device=q.device).unsqueeze(-1)
    col_idx = torch.arange(-bwd_win_size, fwd_win_size + 1, device=q.device).unsqueeze(-2)
    fwd_inv_mask = row_idx < -col_idx
    bwd_inv_mask = q.shape[-2] - 1 - row_idx < col_idx

    qk = torch.sum(q.unsqueeze(-2) * k_strided, dim=-1)
    qk_masked = torch.masked_fill(qk, fwd_inv_mask | bwd_inv_mask, -float("inf"))
    a = torch.softmax(qk_masked, dim=-1)
    return torch.sum(a.unsqueeze(-1) * v_strided, dim=-2)
