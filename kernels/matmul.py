# author: mahdi khashan

import logging
from typing import Callable


LOGGER = logging.getLogger(__name__)


import triton
import triton.language as tl


def profile_kernel(kernel: Callable, size=1024):
    import cProfile, pstats
    profiler = cProfile.Profile()
    profiler.enable()

    import torch

    M = N = K = size
    A = torch.randn((M, K), device="cuda", dtype=torch.float32)
    B = torch.randn((K, N), device="cuda", dtype=torch.float32)

    C = kernel(A, B)
    
    profiler.disable()
    stats = pstats.Stats(profiler).sort_stats('tottime')
    stats.print_stats()

    return torch.allclose(C, A @ B, atol=1e-4)

def matmul_triton_naive(A, B, dtype="float32"):
    M, K = A.shape
    K2, N = B.shape
    assert K == K2
    # todo(mahdi): check why its not working
    # assert A.dtype == dtype

    import torch
    C = torch.empty((M, N), device=A.device, dtype=A.dtype)

    # based on youtube tutorial: https://www.youtube.com/watch?v=DdTsX6DQk24
    # check_tensors_gpu_ready(A)
    # check_tensors_gpu_ready(B)
    # check_tensors_gpu_ready(C)

    grid = (M, N)
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

    kernel[grid](
        A, B, C,
        M, N, K,
    )

    return C


assert profile_kernel(matmul_triton_naive, 32)
# import can be too slow
# assert profile_kernel(matmul_triton_naive, 64)
# assert profile_kernel(matmul_triton_naive, 1024)
# assert profile_kernel(matmul_triton_naive, 2048)
# assert profile_kernel(matmul_triton_naive, 4096)
