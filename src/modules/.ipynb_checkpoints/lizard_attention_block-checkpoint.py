import math
import torch
import torch.nn as nn
import triton
import triton.language as tl


from dataclasses import dataclass


@dataclass
class LizardAttentionBlockConfig:
    embedding_dim: int
    """Embedding dimension of the model."""
    # num_heads: int
    # """Number of heads."""
    # num_blocks: int
    # """Number of blocks."""
    # vocab_size: int
    # """Vocabulary size."""
    # use_bias: bool = False
    # """Whether to use bias in linear layers."""
    # norm_eps: float = 1e-6
    # """Epsilon value for numerical stability in the normalization layers."""
    # norm_reduction_force_float32: bool = True
    # """Whether to force float32 reductions in the normalization layers."""
    # add_out_norm: bool = True
    # """Whether to add a normalization layer after the block stack."""

    # # mlstm backend
    # chunkwise_kernel: ChunkwiseKernelType = "chunkwise--triton_limit_chunk"
    # """Kernel to use for chunkwise parallel processing of the sequence.
    # Also supports fully parallel (i.e. quadratic) backends for comparison.
    # E.g. 'parallel--native_autograd'.
    # """
    # sequence_kernel: SequenceKernelType = "native_sequence__triton"
    # """The sequence kernel to use for processing sequneces step-by-step.
    # Used only for parts of the prefill sequence in inference mode.
    # """
    # mode: BackendModeType = "train"
    # """The mode of operation for the backend. Determines how the `forward` method behaves.
    # Available modes are 'train', 'train_with_padding', 'inference'.
    # 'inference' works with arbitrary sequence lengths, and does not support training. 
    # It calls a sequence of different kernels to process the sequence.
    # 'train_with_padding' pads the input to multiples of `chunk_size`.
    # """
    # chunk_size: int = 64
    # """The chunk size of the chunkwise kernel.
    # If `mode` is 'train_with_padding', the inputs are padded to multiples of this size.
    # """
    # return_last_states: bool = False
    # """Whether to return the last states of the sequence in training mode.
    # Inference mode always returns the last states.
    # """
    # autocast_kernel_dtype: DtypeType = "bfloat16"
    # """The dtype to use for autocast behavior in the kernel.
    # If autocast is enabled all inputs are cast to this dtype before the kernel is called.
    # """
    # eps: float = 1e-6
    # """Epsilon value for numerical stability in the kernel."""
    # inference_state_dtype: DtypeType = "float32"
    # """The dtype to use for the state tensors in inference mode."""
    # # feedforward
    # ffn_proj_factor: float = 2.6667
    # """The factor to determine the dimension of the intermediate projection in the feedforward layer."""
    # ffn_round_up_to_multiple_of: int = 64
    # """Round the intermediate projection dimension to the next multiple of this value."""

    # weight_mode: WeightModeType = "single"
    # """The weight mode to use for the mLSTM layer.
    # Mode 'single' uses separate weights for the query, key, value, and gates.
    # Mode 'fused' uses a single weight matrix for the query, key, value, and gates.
    # 'fused' is benefitial in inference settings.
    # """


class LizardAttentionBlock(nn.Module):
    def __init__(self, d_model, n_heads, window_size=64, alpha=1.0, m=4):
        super().__init__()
        self.d_model, self.n_heads = d_model, n_heads
        self.d_head = d_model // n_heads
        self.window_size, self.alpha, self.m = window_size, alpha, m
        self.meta_tokens = nn.Parameter(torch.randn(1, n_heads, m, self.d_head))
        self.W_gamma = nn.Parameter(torch.randn(1, n_heads, 1, self.d_head))

    def forward(self, q, k, v, x=None, alpha=None):
        alpha = alpha if alpha is not None else self.alpha
        x = x if x is not None else q
        if use_triton:
            g_out, a_out = self.fwd_gla_triton(q, k, v, x), self.fwd_awa_triton(q, k, v)
        else:
            g_out, a_out = self.fwd_gla(q, k, v, x), self.fwd_awa(q, k, v)
        return g_out + alpha * a_out

    def fwd_awa_triton(self, q, k, v):
        B, H, L, D = q.shape
        out = torch.empty_like(q)
        awa_kernel[(B, H, L)](
            q,
            k,
            v,
            self.meta_tokens,
            out,
            *q.stride(),
            *k.stride(),
            *v.stride(),
            *self.meta_tokens.stride(),
            *out.stride(),
            B,
            H,
            L,
            D,
            self.m,
            self.window_size,
            BLOCK_D=triton.next_power_of_2(D),
        )
        return out

    def fwd_gla_triton(self, q, k, v, x):
        B, H, L, D = q.shape
        gamma = torch.sigmoid((self.W_gamma * x).sum(-1, keepdim=True)).squeeze(-1)
        out = torch.empty_like(q)
        gla_linear_kernel[(B, H)](
            q,
            k,
            v,
            gamma,
            out,
            *q.stride(),
            *k.stride(),
            *v.stride(),
            *gamma.stride(),
            *out.stride(),
            B,
            H,
            L,
            D,
            BLOCK_D=triton.next_power_of_2(D),
        )
        return out
