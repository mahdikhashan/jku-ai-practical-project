import torch
from torch.profiler import profile, record_function, ProfilerActivity
from local_attention import LocalAttention

print(f"Device: {torch.cuda.get_device_name(0)}")

batch_size = 16
seq_len = 2048
hidden_size = 512
num_heads = 8
head_dim = 16
dtype = torch.float32
device = "cuda:0"

attn = LocalAttention(
    dim=head_dim,           # dimension of each head
    window_size=512,        # window size
    causal=True,            # auto-regressive
    look_backward=1,        # each window looks at the window before
    look_forward=0,         # for causal attention
    dropout=0.0,            # no dropout for benchmarking
    exact_windowsize=False
).to(device=device)

# (batch, heads, seq_len, head_dim)
q = torch.randn(batch_size, num_heads, seq_len, head_dim, device=device, dtype=dtype)
k = torch.randn(batch_size, num_heads, seq_len, head_dim, device=device, dtype=dtype)
v = torch.randn(batch_size, num_heads, seq_len, head_dim, device=device, dtype=dtype)

mask = torch.ones(batch_size, seq_len, device=device).bool()

# Warmup
print("Warming up...")
for _ in range(10):
    with torch.no_grad():
        _ = attn(q, k, v, mask=mask)
torch.cuda.synchronize()

iterations = 100

print(f"Profiling ({iterations} iterations)...")
with profile(
    activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
    record_shapes=True,
    profile_memory=True,
) as prof:
    for _ in range(iterations):
        with record_function("local_attn_forward"):
            with torch.no_grad():
                y = attn(q, k, v, mask=mask)
    torch.cuda.synchronize()

print(prof.key_averages().table(sort_by="cuda_time_total", row_limit=20))

for event in prof.key_averages():
    if event.key == "local_attn_forward":
        avg_device_time_ms = event.device_time / 1000
        avg_cpu_time_ms = event.cpu_time / 1000
        print("-" * 50)
        print(f"Iterations: {event.count}")
        print(f"Average GPU time per run: {avg_device_time_ms:.4f} ms")
        print(f"Average CPU time per run: {avg_cpu_time_ms:.4f} ms")
        print(f"Throughput: {1000 / avg_device_time_ms:.2f} runs/sec")
        break

print("-" * 50)
print(f"Peak CUDA memory: {torch.cuda.max_memory_allocated() / 1024**3:.2f} GB")

print("\nTop 5 CUDA kernels:")
cuda_events = [(e.key, e.device_time) for e in prof.key_averages() 
               if e.device_time > 0 and not e.key.startswith("local_attn_forward")]
for name, time_us in sorted(cuda_events, key=lambda x: -x[1])[:5]:
    print(f"  {name[:60]:60s} {time_us/1000:.4f} ms")

prof.export_chrome_trace("local_attention_profile.json")
