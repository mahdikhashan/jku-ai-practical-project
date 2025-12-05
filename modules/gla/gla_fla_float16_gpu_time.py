import torch
from fla.layers import GatedLinearAttention

print(f"Device: {torch.cuda.get_device_name(0)}")

batch_size = 1
seq_len = 128
hidden_size = 512
num_heads = 2
dtype = torch.float16
device = "cuda:0"

gla = GatedLinearAttention(
    hidden_size=hidden_size,
    num_heads=num_heads,
    mode='chunk'
).to(device=device, dtype=dtype)

x = torch.randn(batch_size, seq_len, hidden_size, device=device, dtype=dtype)

print("Warming up GPU...")
for _ in range(10):
    _ = gla(x)
torch.cuda.synchronize()

print("Benchmarking...")
start_event = torch.cuda.Event(enable_timing=True)
end_event = torch.cuda.Event(enable_timing=True)

iterations = 100

start_event.record()
for _ in range(iterations):
    y = gla(x)
end_event.record()

torch.cuda.synchronize()

elapsed_time_ms = start_event.elapsed_time(end_event)
avg_time_ms = elapsed_time_ms / iterations

print(f"Success! Output shape: {len(y)}")
print("-" * 30)
print(f"Total time for {iterations} runs: {elapsed_time_ms:.2f} ms")
print(f"Average time per run: {avg_time_ms:.4f} ms")
print("-" * 30)

# print(y)