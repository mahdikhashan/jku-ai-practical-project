# author: mahdi khashan

import logging
from typing import Callable


LOGGER = logging.getLogger(__name__)

import torch

import triton
import triton.language as tl

import tqdm

# from helper import is_cuda, DEVICE


TORCH_HAS_FP8 = hasattr(torch, "float8_e5m2")

is_cuda = torch.cuda.is_available()
if is_cuda:
    DEVICE = torch.device("cuda:0") 
else:
    DEVICE = torch.device("cpu")

def is_cuda():
    return torch.cuda.is_available()

@triton.jit
def kernel(
    A, B, C, M, N, K
):
    m = tl.program_id(0)
    n = tl.program_id(1)
    
    if m >= M or n >= N:
        return
    
    sum = 0.0
    for k in range(K):
        a = tl.load(A + m*K + k)
        b = tl.load(B + k*N + n)
        sum += a * b

    tl.store(C + m*N + n, sum)

def matmul(A, B):
    M, K = A.shape
    K2, N = B.shape
    assert K == K2
    # todo(mahdi): check why its not working
    # assert A.dtype == dtype

    C = torch.empty((M, N), device=A.device, dtype=A.dtype)

    # based on youtube tutorial: https://www.youtube.com/watch?v=DdTsX6DQk24
    # check_tensors_gpu_ready(A)
    # check_tensors_gpu_ready(B)
    # check_tensors_gpu_ready(C)

    grid = (M, N)
    kernel[grid](
        A, B, C,
        M, N, K,
    )

    return C

def profile_kernel(kernel: Callable, size=1024):
    import cProfile, pstats
    profiler = cProfile.Profile()
    profiler.enable()

    M = N = K = size
    A = torch.randn((M, K), device="cuda", dtype=torch.float32)
    B = torch.randn((K, N), device="cuda", dtype=torch.float32)

    C = kernel(A, B)

    profiler.disable()
    stats = pstats.Stats(profiler).sort_stats('tottime')
    stats.print_stats()

    return torch.allclose(C, A @ B, atol=1e-4)

# assert profile_kernel(matmul_triton_naive, 32)
# import can be too slow
# assert profile_kernel(matmul_triton_naive, 64)
# assert profile_kernel(matmul_triton_naive, 1024)
# assert profile_kernel(matmul_triton_naive, 2048)
# assert profile_kernel(matmul_triton_naive, 4096)

if __name__ == "__main__":
    ref_lib = 'cuBLAS' if is_cuda() else 'rocBLAS'

    configs = []
    for fp8_inputs in [False]: #, True]:
        if fp8_inputs and (not TORCH_HAS_FP8 or not is_cuda()):
            continue
        configs.append(
            triton.testing.Benchmark(
                x_names=["M", "N", "K"],
                x_vals=[128 * i for i in range(2, 33)],
                line_arg="provider",
                line_vals=["triton"] if fp8_inputs else [ref_lib.lower(), "triton"],
                line_names=["Triton"] if fp8_inputs else [ref_lib, "Triton"],
                styles=[("green", "-"), ("blue", "-")],
                ylabel="TFLOPS",
                plot_name="matmul-performance-" +
                ("fp16" if not fp8_inputs else "fp8"),
                args={"fp8_inputs": fp8_inputs},
            ))

    @triton.testing.perf_report(configs)
    def benchmark(M, N, K, provider, fp8_inputs):
        a = torch.randn((M, K), device=DEVICE, dtype=torch.float16)
        b = torch.randn((K, N), device=DEVICE, dtype=torch.float16)        
        if TORCH_HAS_FP8 and fp8_inputs:
            a = a.to(torch.float8_e5m2)
            b = b.T
            b = b.to(torch.float8_e5m2)
            
        quantiles = [0.5, 0.2, 0.8]
        if provider == ref_lib.lower():
            ms, min_ms, max_ms = triton.testing.do_bench(lambda: torch.matmul(a, b), quantiles=quantiles)
        elif provider == 'triton':
            ms, min_ms, max_ms = triton.testing.do_bench(lambda: matmul(a, b), quantiles=quantiles)
        else:
            return None, None, None

        perf = lambda ms: 2 * M * N * K * 1e-12 / (ms * 1e-3)
        return perf(ms), perf(max_ms), perf(min_ms)

    try:
        original_benchmark_fn = benchmark.fn 
    except AttributeError:
        # Fallback for older versions that might use 'f'
        original_benchmark_fn = benchmark.f
    
    final_results = {}
    total_runs = sum(len(c.x_vals) * len(c.line_vals) for c in configs)

    with tqdm.tqdm(total=total_runs, desc="Benchmarking MatMul Performance") as pbar:
        for config in configs:
            results = {}
            for M in config.x_vals:
                N = K = M
                for provider in config.line_vals:
                    perf_value = original_benchmark_fn(M, N, K, provider, config.args['fp8_inputs'])
                    if provider not in results:
                        results[provider] = {}
                    results[provider][(M, N, K)] = perf_value
                    
                    pbar.update(1)
            
            final_results[config] = results

    triton.testing.print_table(final_results, configs)
