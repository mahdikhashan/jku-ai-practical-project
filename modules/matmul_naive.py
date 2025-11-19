# author: mahdi khashan

import logging
from typing import Callable


LOGGER = logging.getLogger(__name__)

import torch

import triton
import triton.language as tl

from .helper import is_cuda, DEVICE


TORCH_HAS_FP8 = hasattr(torch, "float8_e5m2")

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
    # copied from: https://triton-lang.org/main/getting-started/tutorials/03-matrix-multiplication.html#benchmark
    ref_lib = 'cuBLAS' if is_cuda() else 'rocBLAS'

    configs = []
    for fp8_inputs in [False, True]:
        if fp8_inputs and (not TORCH_HAS_FP8 or not is_cuda()):
            continue
        configs.append(
            triton.testing.Benchmark(
                x_names=["M", "N", "K"],  # Argument names to use as an x-axis for the plot
                x_vals=[128 * i for i in range(2, 33)],  # Different possible values for `x_name`
                line_arg="provider",  # Argument name whose value corresponds to a different line in the plot
                # Possible values for `line_arg`
                # Don't compare to cublas for fp8 cases as torch.matmul doesn't support fp8 at the moment.
                line_vals=["triton"] if fp8_inputs else [ref_lib.lower(), "triton"],  # Label name for the lines
                line_names=["Triton"] if fp8_inputs else [ref_lib, "Triton"],  # Line styles
                styles=[("green", "-"), ("blue", "-")],
                ylabel="TFLOPS",  # Label name for the y-axis
                plot_name="matmul-performance-" +
                ("fp16" if not fp8_inputs else "fp8"),  # Name for the plot, used also as a file name for saving the plot.
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
        if provider == 'triton':
            ms, min_ms, max_ms = triton.testing.do_bench(lambda: matmul(a, b), quantiles=quantiles)
        perf = lambda ms: 2 * M * N * K * 1e-12 / (ms * 1e-3)
        return perf(ms), perf(max_ms), perf(min_ms)

    benchmark.run(show_plots=False, print_data=True)
