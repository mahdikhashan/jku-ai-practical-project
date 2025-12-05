import torch
from torch.profiler import profile, record_function, ProfilerActivity


def swa_naive(q, k, v, window_sizes: tuple[int, int] = (15, 16)):
    """Naive implementation of sliding window attention."""
    bwd_win_size, fwd_win_size = window_sizes

    row_idx = torch.arange(q.shape[-2], device=q.device).unsqueeze(-1)
    col_idx = torch.arange(k.shape[-2], device=k.device).unsqueeze(-2)
    fwd_inv_mask = row_idx < col_idx - fwd_win_size
    bwd_inv_mask = row_idx > col_idx + bwd_win_size

    qk = q @ k.transpose(-1, -2)
    qk_masked = torch.masked_fill(qk, fwd_inv_mask | bwd_inv_mask, -float("inf"))
    a = torch.softmax(qk_masked, dim=-1)
    return a @ v


def swa_strided(q, k, v, window_sizes: tuple[int, int] = (15, 16)):
    """Sliding window attention implemented using striding tricks."""
    bwd_win_size, fwd_win_size = window_sizes
    win_size = 1 + bwd_win_size + fwd_win_size
    assert win_size < k.shape[-2]

    k_to_pad = [k.ravel()]
    v_to_pad = [v.ravel()]
    if bwd_win_size > 0:
        k_to_pad.insert(0, torch.zeros(bwd_win_size * k.shape[-1], device=k.device, dtype=k.dtype))
        v_to_pad.insert(0, torch.zeros(bwd_win_size * v.shape[-1], device=v.device, dtype=v.dtype))
    if fwd_win_size > 0:
        k_to_pad.append(torch.zeros(fwd_win_size * k.shape[-1], device=k.device, dtype=k.dtype))
        v_to_pad.append(torch.zeros(fwd_win_size * v.shape[-1], device=v.device, dtype=v.dtype))

    k_strided = torch.as_strided(
        torch.cat(k_to_pad, dim=0),
        size=(*k.shape[:-1], win_size, k.shape[-1]),
        stride=(*k.stride()[:-1], k.stride(-2), k.stride(-1))
    )
    v_strided = torch.as_strided(
        torch.cat(v_to_pad, dim=0),
        size=(*v.shape[:-1], win_size, v.shape[-1]),
        stride=(*v.stride()[:-1], v.stride(-2), v.stride(-1))
    )

    row_idx = torch.arange(q.shape[-2], device=q.device).unsqueeze(-1)
    col_idx = torch.arange(-bwd_win_size, fwd_win_size + 1, device=q.device).unsqueeze(-2)
    fwd_inv_mask = row_idx < -col_idx
    bwd_inv_mask = q.shape[-2] - 1 - row_idx < col_idx

    qk = torch.sum(q.unsqueeze(-2) * k_strided, dim=-1)
    qk_masked = torch.masked_fill(qk, fwd_inv_mask | bwd_inv_mask, -float("inf"))
    a = torch.softmax(qk_masked, dim=-1)
    return torch.sum(a.unsqueeze(-1) * v_strided, dim=-2)


print(f"Device: {torch.cuda.get_device_name(0)}")

batch_size = 16
seq_len = 2048
num_heads = 32
head_dim = 16
window_sizes = (127, 128)  # backward, forward
dtype = torch.float16
device = "cuda:0"

# (batch, num_heads, seq_len, head_dim)
q = torch.randn(batch_size, num_heads, seq_len, head_dim, device=device, dtype=dtype)
k = torch.randn(batch_size, num_heads, seq_len, head_dim, device=device, dtype=dtype)
v = torch.randn(batch_size, num_heads, seq_len, head_dim, device=device, dtype=dtype)

swa_func = swa_strided
impl_name = "swa_strided"

# Warmup
for _ in range(10):
    _ = swa_func(q, k, v, window_sizes=window_sizes)
torch.cuda.synchronize()

iterations = 100

with profile(
    activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
    record_shapes=True,
    profile_memory=True,
) as prof:
    for _ in range(iterations):
        with record_function(f"{impl_name}_forward"):
            y = swa_func(q, k, v, window_sizes=window_sizes)
    torch.cuda.synchronize()

print(prof.key_averages().table(sort_by="cuda_time_total", row_limit=20))

for event in prof.key_averages():
    if event.key == f"{impl_name}_forward":
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
               if e.device_time > 0 and not e.key.startswith(impl_name)]
for name, time_us in sorted(cuda_events, key=lambda x: -x[1])[:5]:
    print(f"  {name[:60]:60s} {time_us/1000:.4f} ms")

prof.export_chrome_trace(f"{impl_name}_profile_benchmark.json")
