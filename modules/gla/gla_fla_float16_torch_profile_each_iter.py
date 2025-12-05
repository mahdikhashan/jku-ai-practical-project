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

# Warmup
for _ in range(10):
    _ = gla(x)
torch.cuda.synchronize()

iterations = 100

with profile(
    activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
    record_shapes=True,
    profile_memory=True,
) as prof:
    for _ in range(iterations):
        with record_function("gla_forward"):
            y = gla(x)
    torch.cuda.synchronize()

print(prof.key_averages().table(sort_by="cuda_time_total", row_limit=20))

for event in prof.key_averages():
    if event.key == "gla_forward":
        avg_cuda_time_ms = event.cuda_time / 1000
        avg_cpu_time_ms = event.cpu_time / 1000
        print("-" * 50)
        print(f"Iterations: {event.count}")
        print(f"Average CUDA time per run: {avg_cuda_time_ms:.4f} ms")
        print(f"Average CPU time per run: {avg_cpu_time_ms:.4f} ms")
        break

prof.export_chrome_trace("gla_fla_float_16_torch_profile_each_iter.json")
