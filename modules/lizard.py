from abc import ABC, abstractmethod

from dataclasses import dataclass

from typing import Literal
import torch  # type: ignore
import torch.nn as nn  # type: ignore


class AbstractLizardAttentionBlock(nn.Module, ABC):
    def __init__(self, **kwargs):
        super().__init__()

    @abstractmethod
    def forward(self, q, k, v):
        pass


@dataclass
class DTypeConfig:
    name: Literal["float16", "bfloat16", "float32"] = "bfloat16"

    def to_torch(self) -> torch.dtype:
        return {
            "float16": torch.float16,
            "bfloat16": torch.bfloat16,
            "float32": torch.float32,
        }[self.name]


@dataclass
class LizardAttentionBlockConfig:
    batch_size: int
    seq_len: int
    hidden_size: int
    num_heads: int
    dtype: DTypeConfig
    device: str


def create_lizard_attention_config(
    *,
    batch_size: int = 1,
    seq_len: int = 128,
    hidden_size: int = 128,
    num_heads: int = 1,
    dtype: Literal["float16", "bfloat16", "float32"] = "bfloat16",
    device: str = "cuda:0",
) -> LizardAttentionBlockConfig:
    return LizardAttentionBlockConfig(
        batch_size=batch_size,
        seq_len=seq_len,
        hidden_size=hidden_size,
        num_heads=num_heads,
        dtype=DTypeConfig(name=dtype),
        device=device,
    )
