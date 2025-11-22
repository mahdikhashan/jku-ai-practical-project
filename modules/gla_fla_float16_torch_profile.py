import torch
from torch.profiler import profile, record_function, ProfilerActivity
from fla.layers import GatedLinearAttention

print(f"Device: {torch.cuda.get_device_name(0)}")

batch_size = 16
seq_len = 2048
hidden_size = 512
num_heads = 32
dtype = torch.float16
device = "cuda:0"

gla = GatedLinearAttention(
    hidden_size=hidden_size,
    num_heads=num_heads,
    mode='chunk'
).to(device=device, dtype=dtype)

x = torch.randn(batch_size, seq_len, hidden_size, device=device, dtype=dtype)

for _ in range(10):
    _ = gla(x)
torch.cuda.synchronize()

with profile(
    activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
    record_shapes=True,
    profile_memory=True,
    with_stack=True
) as prof:
    with record_function("gla_forward"):
        for _ in range(100):
            y = gla(x)
        torch.cuda.synchronize()

print(prof.key_averages().table(sort_by="cuda_time_total", row_limit=20))

prof.export_chrome_trace("gla_fla_float_torch_profile.json")
