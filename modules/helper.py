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

# gemini 3 pro generated code
import torch
from torch.profiler import profile, record_function, ProfilerActivity
from functools import wraps
from datetime import datetime
import json
import os
import inspect

def benchmark(
    warmup_iterations=10,
    benchmark_iterations=100,
    log_file=None,
    save_results=True,
    trace_filename=None,
):
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            # --- 1. Capture & Decompose Parameters ---
            sig = inspect.signature(func)
            bound_args = sig.bind(*args, **kwargs)
            bound_args.apply_defaults()

            params_to_log = {}
            for name, value in bound_args.arguments.items():
                
                # Handling Tensors: Decompose shape AND dtype
                if isinstance(value, torch.Tensor):
                    shape = value.shape
                    
                    # 1. Log generic info
                    params_to_log[name] = f"Tensor{list(shape)}"
                    params_to_log[f"{name}_dtype"] = str(value.dtype)  # <--- NEW: Log dtype separately
                    
                    # 2. Extract specific dimensions (Batch, Seq, Hidden)
                    if len(shape) == 3:
                        params_to_log[f"{name}_batch_size"] = shape[0]
                        params_to_log[f"{name}_seq_len"] = shape[1]
                        params_to_log[f"{name}_hidden_size"] = shape[2]
                    
                    # Fallback: Log every dimension index
                    for i, dim in enumerate(shape):
                        params_to_log[f"{name}_dim_{i}"] = dim

                # Handling Simple Types
                elif isinstance(value, (int, float, str, bool, type(None))):
                    params_to_log[name] = value
                
                # Handling Others
                else:
                    params_to_log[name] = str(value)

            exp_name = params_to_log.get("experiment_name", "N/A")

            # --- 2. Initialize Results ---
            results = {
                "timestamp": datetime.now().isoformat(),
                "function": func.__name__,
                **params_to_log 
            }

            if torch.cuda.is_available():
                device_name = torch.cuda.get_device_name(0)
                print(f"Device: {device_name}")
                results["device"] = device_name
            else:
                print("CUDA not available")
                results["device"] = "CPU"
                return func(*args, **kwargs)

            # --- 3. Warmup ---
            torch.cuda.reset_peak_memory_stats()
            torch.cuda.empty_cache()
            print(f"Warming up GPU ({warmup_iterations} iters)...")
            for _ in range(warmup_iterations):
                _ = func(*args, **kwargs)
            torch.cuda.synchronize()

            # --- 4. Wall Clock Benchmark ---
            torch.cuda.reset_peak_memory_stats()
            print(f"Benchmarking Wall Clock ({benchmark_iterations} iters)...")

            start_event = torch.cuda.Event(enable_timing=True)
            end_event = torch.cuda.Event(enable_timing=True)

            start_event.record()
            for _ in range(benchmark_iterations):
                result = func(*args, **kwargs)
            end_event.record()
            torch.cuda.synchronize()

            total_wall_time_ms = start_event.elapsed_time(end_event)
            avg_wall_time_ms = total_wall_time_ms / benchmark_iterations

            peak_memory_mb = torch.cuda.max_memory_allocated() / 1024**2
            current_memory_mb = torch.cuda.memory_allocated() / 1024**2
            reserved_memory_mb = torch.cuda.memory_reserved() / 1024**2

            # --- 5. Profiler ---
            print(f"Running Profiler ({benchmark_iterations} iters)...")
            final_trace_name = trace_filename or f"{func.__name__}_trace.json"

            avg_cuda_time_ms = 0.0
            avg_cpu_time_ms = 0.0

            with profile(
                activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
                record_shapes=True,
                profile_memory=True,
            ) as prof:
                for _ in range(benchmark_iterations):
                    with record_function(func.__name__):
                        _ = func(*args, **kwargs)
                torch.cuda.synchronize()

            prof.export_chrome_trace(final_trace_name)
            
            key_avgs = prof.key_averages()
            print(key_avgs.table(sort_by="cuda_time_total", row_limit=10))

            found_event = False
            for event in key_avgs:
                if event.key == func.__name__:
                    c_total = getattr(event, "cpu_time_total", getattr(event, "cpu_time", 0.0))
                    g_total = getattr(event, "cuda_time_total", getattr(event, "cuda_time", 0.0))
                    avg_cuda_time_ms = (g_total / 1000.0) / benchmark_iterations
                    avg_cpu_time_ms = (c_total / 1000.0) / benchmark_iterations
                    found_event = True
                    break

            if not found_event:
                print(f"Warning: Could not find key '{func.__name__}' in profiler results.")

            # --- 6. Save Results ---
            output_shape = str(result.shape if hasattr(result, "shape") else len(result))
            
            results.update({
                "output_shape": output_shape,
                "warmup_iterations": warmup_iterations,
                "benchmark_iterations": benchmark_iterations,
                "total_wall_time_ms": round(total_wall_time_ms, 2),
                "avg_wall_time_ms": round(avg_wall_time_ms, 4),
                "avg_cuda_time_ms": round(avg_cuda_time_ms, 4),
                "avg_cpu_time_ms": round(avg_cpu_time_ms, 4),
                "peak_memory_mb": round(peak_memory_mb, 2),
                "current_memory_mb": round(current_memory_mb, 2),
                "reserved_memory_mb": round(reserved_memory_mb, 2),
                "trace_file": final_trace_name,
            })

            # Pretty Print Summary
            print("\n" + "=" * 60)
            print(f"{'BENCHMARK RESULTS':^60}")
            print("=" * 60)
            print(f"{'Function':<30} {func.__name__:>28}")
            
            # Print important params
            for k, v in params_to_log.items():
                # Print dims, dtypes, and standard params
                if any(x in k for x in ["dim_", "_len", "_size", "dtype", "mode", "heads"]):
                     print(f"{k:<30} {str(v):>28}")
            
            print("-" * 60)
            print(f"{'Avg Wall Time (ms)':<30} {avg_wall_time_ms:>27.4f}")
            print(f"{'Avg CUDA Kernel Time (ms)':<30} {avg_cuda_time_ms:>27.4f}")
            print("=" * 60 + "\n")

            if save_results:
                log_path = log_file or "benchmark_results.json"
                if os.path.exists(log_path):
                    try:
                        with open(log_path, "r") as f:
                            all_results = json.load(f)
                    except json.JSONDecodeError:
                        all_results = []
                else:
                    all_results = []

                all_results.append(results)

                with open(log_path, "w") as f:
                    json.dump(all_results, f, indent=2)
                print(f"JSON results saved to: {log_path}")

            return result
        return wrapper
    return decorator
