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
    A = torch.randn((M, K), device=DEVICE, dtype=torch.float32)
    B = torch.randn((K, N), device=DEVICE, dtype=torch.float32)

    C = kernel(A, B)

    profiler.disable()
    stats = pstats.Stats(profiler).sort_stats('tottime')
    stats.print_stats()

    return torch.allclose(C, A @ B, atol=1e-4)

if __name__ == "__main__":    
    M, N, K = 3, 3, 3 
    DTYPE = torch.float32 
    
    print(f"Running Triton MatMul for a {M}x{K} @ {K}x{N} = {M}x{N} matrix (3x3)...")

    A = torch.randn((M, K), device=DEVICE, dtype=DTYPE)
    B = torch.randn((K, N), device=DEVICE, dtype=DTYPE)
    
    try:
        C_triton = matmul(A, B)
        print("Triton MatMul completed.")
    except Exception as e:
        print(f"Error running Triton MatMul: {e}")
        exit() 

    C_ref = torch.matmul(A, B)

    if torch.allclose(C_triton, C_ref, atol=1e-4):
        print(f"Correctness Check Passed (Tolerance: {1e-4})")
        print("Triton Result (C_triton):")
        print(C_triton)
    else:
        print("Correctness Check FAILED!")
        print(f"Difference (Absolute max): {torch.max(torch.abs(C_triton - C_ref))}")
        print("Reference Result (C_ref):")
        print(C_ref)
