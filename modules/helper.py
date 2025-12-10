# copied from: https://github.com/gpu-mode/lectures/blob/main/lecture_014/triton_util.py,
# based on youtube tutorial: https://www.youtube.com/watch?v=DdTsX6DQk24

import os

import torch

import triton
import triton.language as tl


def test_pid_conds(conds, pid_0=[0], pid_1=[0], pid_2=[0]):
    """Test if condition on pids are fulfilled
    E.g.:
        '=0'  checks that pid_0 == 0
        ',>1' checks that pid_1 > 1
        '>1,=0' checks that pid_0 > 1 and pid_1 == 0
    """
    pids = pid_0[0], pid_1[0], pid_2[0]
    conds = conds.replace(" ", "").split(",")
    for i, (cond, pid) in enumerate(zip(conds, pids)):
        if cond == "":
            continue
        op, threshold = cond[0], int(cond[1:])
        if op not in ["<", ">", ">=", "<=", "=", "!="]:
            raise ValueError(
                f"Rules may only use these ops: '<','>','>=','<=','=', '!='. Invalid rule: '{cond}'."
            )
        op = "==" if op == "=" else op
        if not eval(f"{pid} {op} {threshold}"):
            return False
    return True


assert test_pid_conds("")
assert test_pid_conds(">0", [1], [1])
assert not test_pid_conds(">0", [0], [1])
assert test_pid_conds("=0,=1", [0], [1], [0])


def breakpoint_if(conds, pid_0=[0], pid_1=[0], pid_2=[0]):
    """Stop kernel, if any condition of pids is fulfilled"""
    from IPython.core.debugger import set_trace

    if test_pid_conds(conds, pid_0, pid_1, pid_2):
        set_trace()


def print_if(txt, conds, pid_0=[0], pid_1=[0], pid_2=[0]):
    """Print txt, if any condition of pids is fulfilled"""
    if test_pid_conds(conds, pid_0, pid_1, pid_2):
        print(txt)


def check_tensors_gpu_ready(*tensors):
    for t in tensors:
        assert t.is_contiguous(), "A tensor is not contiguous"
        if not os.environ.get("TRITON_INTERPRET") == "1":
            assert t.is_cuda, "A tensor is not on cuda"


def cdiv(a, b):
    return (a + b - 1) // b


assert cdiv(10, 2) == 5
assert cdiv(10, 3) == 4


@triton.jit
def get_1d_offest(size, n_prev_chunks):
    return n_prev_chunks * size + tl.arange(0, size)


@triton.jit
def get_2d_offset(offs_0, offs_1, stride_0, stride_1=1):
    return tl.expand_dims(offs_0, 1) * stride_0 + tl.expand_dims(offs_1, 0) * stride_1


@triton.jit
def get_1d_mask(offs, max):
    return offs < max


@triton.jit
def get_2d_mask(offs_0, offs_1, max_0, max_1):
    return (tl.expand_dims(offs_0, 1) < max_0) & (tl.expand_dims(offs_1, 0) < max_1)


def triton_get_active_torch_device():
    # copied from: https://triton-lang.org/main/getting-started/tutorials/03-matrix-multiplication.html#final-result
    return triton.runtime.driver.active.get_active_torch_device()


def is_cuda():
    return triton.runtime.driver.active.get_current_target().backend == "cuda"


def print_gpu_specs():
    print(f"Detected {torch.cuda.device_count()} device(s):")
    for i in range(torch.cuda.device_count()):
        p = torch.cuda.get_device_properties(i)
        print(
            f"[{i}] {p.name} | {p.total_memory / 1024**3:.1f} GB VRAM | {p.multi_processor_count} SMs"
        )


def get_pid():
    import os

    return os.getpid()


def get_git_commit():
    import subprocess

    subprocess.check_output(["git", "describe", "--always"])


def get_datetime_now():
    import datetime

    datetime.datetime.now().strftime("%Y-%m-%d %H:%M")


def is_contagious():
    # todo(mahdi): is contagious
    pass


def get_dtype():
    return (
        "bfloat16"
        if torch.cuda.is_available() and torch.cuda.is_bf16_supported()
        else "float16"
    )


from functools import wraps
from datetime import datetime
import json
import os


def benchmark(
    warmup_iterations=10, benchmark_iterations=100, log_file=None, save_results=True
):
    """
    Decorator to benchmark GPU performance with memory tracking and logging.

    Args:
        warmup_iterations: Number of warmup runs
        benchmark_iterations: Number of benchmark runs
        log_file: Optional path to log file (default: benchmark_results.json)
        save_results: Whether to save results to file
    """

    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            results = {
                "timestamp": datetime.now().isoformat(),
                "function": func.__name__,
            }

            # Print device info
            if torch.cuda.is_available():
                device_name = torch.cuda.get_device_name(0)
                print(f"Device: {device_name}")
                results["device"] = device_name
            else:
                print("CUDA not available")
                results["device"] = "CPU"
                return func(*args, **kwargs)

            # Reset memory stats
            torch.cuda.reset_peak_memory_stats()
            torch.cuda.empty_cache()

            # Warmup
            print("Warming up GPU...")
            for _ in range(warmup_iterations):
                _ = func(*args, **kwargs)
            torch.cuda.synchronize()

            # Reset memory stats after warmup
            torch.cuda.reset_peak_memory_stats()

            # Benchmark
            print("Benchmarking...")
            start_event = torch.cuda.Event(enable_timing=True)
            end_event = torch.cuda.Event(enable_timing=True)

            start_event.record()
            for _ in range(benchmark_iterations):
                result = func(*args, **kwargs)
            end_event.record()

            torch.cuda.synchronize()

            # Timing statistics
            elapsed_time_ms = start_event.elapsed_time(end_event)
            avg_time_ms = elapsed_time_ms / benchmark_iterations

            # Memory statistics
            peak_memory_mb = torch.cuda.max_memory_allocated() / 1024**2
            current_memory_mb = torch.cuda.memory_allocated() / 1024**2
            reserved_memory_mb = torch.cuda.memory_reserved() / 1024**2

            # Store results
            results.update(
                {
                    "output_shape": str(
                        result.shape if hasattr(result, "shape") else len(result)
                    ),
                    "warmup_iterations": warmup_iterations,
                    "benchmark_iterations": benchmark_iterations,
                    "total_time_ms": round(elapsed_time_ms, 2),
                    "avg_time_ms": round(avg_time_ms, 4),
                    "peak_memory_mb": round(peak_memory_mb, 2),
                    "current_memory_mb": round(current_memory_mb, 2),
                    "reserved_memory_mb": round(reserved_memory_mb, 2),
                }
            )

            # Print results table
            print("\n" + "=" * 60)
            print(f"{'BENCHMARK RESULTS':^60}")
            print("=" * 60)
            print(f"{'Metric':<30} {'Value':>28}")
            print("-" * 60)
            print(f"{'Function':<30} {func.__name__:>28}")
            print(f"{'Device':<30} {results['device']:>28}")
            print(f"{'Output Shape':<30} {results['output_shape']:>28}")
            print("-" * 60)
            print(f"{'Warmup Iterations':<30} {warmup_iterations:>28}")
            print(f"{'Benchmark Iterations':<30} {benchmark_iterations:>28}")
            print(f"{'Total Time (ms)':<30} {elapsed_time_ms:>27.2f}")
            print(f"{'Average Time (ms)':<30} {avg_time_ms:>27.4f}")
            print("-" * 60)
            print(f"{'Peak Memory (MB)':<30} {peak_memory_mb:>27.2f}")
            print(f"{'Current Memory (MB)':<30} {current_memory_mb:>27.2f}")
            print(f"{'Reserved Memory (MB)':<30} {reserved_memory_mb:>27.2f}")
            print("=" * 60 + "\n")

            # Save to file if requested
            if save_results:
                log_path = log_file or "benchmark_results.json"

                # Load existing results if file exists
                if os.path.exists(log_path):
                    try:
                        with open(log_path, "r") as f:
                            all_results = json.load(f)
                    except json.JSONDecodeError:
                        all_results = []
                else:
                    all_results = []

                # Append new results
                all_results.append(results)

                # Save to file
                with open(log_path, "w") as f:
                    json.dump(all_results, f, indent=2)

                print(f"Results saved to: {log_path}")

            return result

        return wrapper

    return decorator
