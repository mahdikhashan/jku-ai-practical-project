import math
import torch
import pytest
from src.modules.swa.swa_naive import swa_naive


def make_qkv(L, D, batch=1, heads=1, seed=0):
    torch.manual_seed(seed)
    shape = (batch, heads, L, D)
    return torch.randn(shape), torch.randn(shape), torch.randn(shape)


def test_output_shape():
    q, k, v = make_qkv(L=16, D=8)
    out = swa_naive(q, k, v)
    assert out.shape == q.shape


def test_output_shape_different_dv():
    L, Dqk, Dv = 16, 8, 4
    torch.manual_seed(0)
    q = torch.randn(1, 1, L, Dqk)
    k = torch.randn(1, 1, L, Dqk)
    v = torch.randn(1, 1, L, Dv)
    out = swa_naive(q, k, v)
    assert out.shape == (1, 1, L, Dv)


def test_causal_masking():
    """With fwd_win_size=0 each token attends only to past tokens within bwd window."""
    L, D = 8, 4
    q, k, v = make_qkv(L=L, D=D)
    bwd = 2
    out = swa_naive(q, k, v, window_sizes=(bwd, 0))
    assert out.shape == (1, 1, L, D)
    assert not torch.isnan(out).any()


def test_window_limits_attention():
    """Tokens outside the window must receive zero attention weight."""
    L, D = 8, 4
    q, k, v = make_qkv(L=L, D=D)
    bwd, fwd = 1, 1

    row_idx = torch.arange(L).unsqueeze(-1)
    col_idx = torch.arange(L).unsqueeze(-2)
    expected_mask = (row_idx >= col_idx - fwd) & (row_idx <= col_idx + bwd)  # True = in window

    qk = (q @ k.transpose(-1, -2)) / math.sqrt(D)
    qk_masked = qk.masked_fill(~expected_mask, -float("inf"))
    a = torch.softmax(qk_masked, dim=-1)

    # Positions outside window have -inf before softmax → 0 after softmax
    outside = ~expected_mask
    assert (a[0, 0][outside] == 0.0).all()


def test_no_nan_or_inf_in_output():
    q, k, v = make_qkv(L=32, D=16)
    out = swa_naive(q, k, v)
    assert not torch.isnan(out).any()
    assert not torch.isinf(out).any()


def test_batch_and_heads_preserved():
    q, k, v = make_qkv(L=12, D=8, batch=2, heads=4)
    out = swa_naive(q, k, v)
    assert out.shape == (2, 4, 12, 8)


def test_full_window_matches_standard_attention():
    """A window large enough to cover the whole sequence should match standard attention."""
    L, D = 8, 4
    q, k, v = make_qkv(L=L, D=D)

    out_swa = swa_naive(q, k, v, window_sizes=(L, L))

    qk = (q @ k.transpose(-1, -2)) / math.sqrt(D)
    out_std = torch.softmax(qk, dim=-1) @ v

    torch.testing.assert_close(out_swa, out_std)
