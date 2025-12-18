import torch.nn as nn  # type: ignore


class LizardModule(nn.Module):
    def __init__(self, d_model, n_heads, window_size=64, chunk_size=64, alpha=1):
        super().__init__()
        self.d_model = d_model
        self.n_heads = n_heads
        self.d_head = d_model // n_heads
        self.window_size = window_size
        self.chunk_size = chunk_size
        self.alpha = alpha

    def forward(self, q, k, v, x, alpha=1):
        raise NotImplementedError("awa forward: not implemented!")

    def awa_fwd(self, q, k, v):
        raise NotImplementedError("awa forward: not implemented!")

    def awa_fwd_triton_kernel(self, q, k, v):
        raise NotImplementedError("not implemented!")

    def gla_fwd(self, q, k, v, x):
        raise NotImplementedError("not implemented!")

    def gla_fwd_triton_kernel(self, q, k, v, g):
        raise NotImplementedError("not implemented!")
