import torch
from fla.layers import GatedLinearAttention


batch_size = 1
seq_len = 128
hidden_size = 512
num_heads = 2
# bfloat16
dtype = torch.bfloat16
device = "cuda:0"

gla = GatedLinearAttention(
    hidden_size=hidden_size,
    num_heads=num_heads,
    mode='chunk'
).to(device=device, dtype=dtype)

x = torch.randn(batch_size, seq_len, hidden_size, device=device, dtype=dtype)

try:
    y = gla(x)
    print("Success! Output shape:", len(y))
except Exception as e:
    print("Failed:", e)

print(y)
