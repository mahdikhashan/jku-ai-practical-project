import torch
torch.manual_seed(0)

from torch.nn.attention.flex_attention import flex_attention, create_block_mask


flex_attention_compiled = torch.compile(flex_attention)


def swa_flex(q, k, v, window_sizes: tuple[int, int] = (15, 16)):
    bwd_win, fwd_win = window_sizes
    B, H, N, D = q.shape

    # mask_mod signature: (b, h, q_idx, kv_idx) -> bool
    # True  = attend,  False = masked (-inf before softmax)
    def sliding_window(b, h, q_idx, kv_idx):
        return (q_idx - kv_idx <= bwd_win) & (kv_idx - q_idx <= fwd_win)

    block_mask = create_block_mask(
        sliding_window, B=B, H=H, Q_LEN=N, KV_LEN=N, device=q.device,
    )

    return flex_attention_compiled(q, k, v, block_mask=block_mask)
