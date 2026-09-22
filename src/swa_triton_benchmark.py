import csv
import sys
import time
import traceback
from functools import lru_cache
from pathlib import Path

import torch
import triton

from swa_torch_naive import swa_naive
from swa_torch_strided import swa_strided as swa_strided_pt
from swa_triton_custom_kernel import swa_strided_triton, swa_tiled_triton

SEQ_LENS = [1024, 2048, 4096, 8192, 16384]
WINDOWS = [(15, 16), (31, 32), (63, 64), (127, 128), (255, 256)]
DTYPES = [torch.float32, torch.float16]
B, H, D = 1, 8, 64
NAIVE_BUDGET_GIB = 18.0
WARMUP_MS, REP_MS = 25, 100
OUTPUT = Path("results/swa_bench.csv")
ERROR_LOG = Path("results/swa_bench_errors.log")

torch.set_float32_matmul_precision("highest")
torch._dynamo.config.recompile_limit = 128

naive_compiled = torch.compile(swa_naive, dynamic=False)
strided_compiled = torch.compile(swa_strided_pt, dynamic=False)

try:
    from torch.nn.attention.flex_attention import create_block_mask, flex_attention

    flex_attention_compiled = torch.compile(flex_attention, dynamic=False)
    create_block_mask_compiled = torch.compile(create_block_mask, dynamic=False)
    HAS_FLEX = True
except ImportError:
    HAS_FLEX = False


@lru_cache(maxsize=None)
def block_mask_for(n, bwd, fwd):
    def mask_mod(b, h, qi, ki):
        return (qi - ki <= bwd) & (ki - qi <= fwd)

    return create_block_mask_compiled(
        mask_mod, B=None, H=None, Q_LEN=n, KV_LEN=n, device="cuda",
    )


def swa_flex(q, k, v, bwd, fwd):
    mask = block_mask_for(q.shape[2], bwd, fwd)
    return flex_attention_compiled(q, k, v, block_mask=mask)


def with_precision(fn, precision):
    def wrapped(*args):
        prev = torch.get_float32_matmul_precision()
        torch.set_float32_matmul_precision(precision)
        try:
            return fn(*args)
        finally:
            torch.set_float32_matmul_precision(prev)
    return wrapped


VARIANTS = {
    "naive_pt": lambda q, k, v, bwd, fwd: swa_naive(q, k, v, (bwd, fwd)),
    "naive_pt_compiled": lambda q, k, v, bwd, fwd: naive_compiled(q, k, v, (bwd, fwd)),
    "strided_pt": lambda q, k, v, bwd, fwd: swa_strided_pt(q, k, v, (bwd, fwd)),
    "strided_pt_compiled": lambda q, k, v, bwd, fwd: strided_compiled(q, k, v, (bwd, fwd)),
    "triton_tiled": swa_tiled_triton,
    "triton_strided": swa_strided_triton,
}
if HAS_FLEX:
    VARIANTS["flex"] = swa_flex
    VARIANTS["flex_tf32"] = with_precision(swa_flex, "high")

QUADRATIC_VARIANTS = {"naive_pt", "naive_pt_compiled"}
FP32_ONLY_VARIANTS = {"flex_tf32"}
LOGGED_ERRORS = set()
EMPTY = {"latency_ms": float("nan"), "latency_p20": float("nan"), "latency_p80": float("nan"),
         "tflops": float("nan"), "peak_mib": float("nan"), "max_abs_err": float("nan")}


def naive_gib(n):
    return 3.0 * B * H * n * n * 4 / (1024 ** 3)


def flops(n, bwd, fwd):
    w = 1 + bwd + fwd
    return 4.0 * n * w * D * H * B


def peak_mem_mib(fn, *args):
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    fn(*args)
    torch.cuda.synchronize()
    return torch.cuda.max_memory_allocated() / (1024 ** 2)


def log_error(variant, dtype_name, exc):
    key = (variant, dtype_name)
    if key not in LOGGED_ERRORS:
        LOGGED_ERRORS.add(key)
        ERROR_LOG.parent.mkdir(parents=True, exist_ok=True)
        with open(ERROR_LOG, "a") as f:
            f.write(f"\n{'=' * 78}\nvariant={variant}  dtype={dtype_name}\n{'=' * 78}\n")
            f.write("".join(traceback.format_exception(type(exc), exc, exc.__traceback__)))
    return f"error:{type(exc).__name__}"


def bench_one(variant, fn, n, bwd, fwd, ref, row, q, k, v):
    try:
        out = fn(q, k, v, bwd, fwd)
        torch.cuda.synchronize()
        err = float("nan") if ref is None else (out.float() - ref).abs().max().item()
        med, p20, p80 = triton.testing.do_bench(
            lambda: fn(q, k, v, bwd, fwd), warmup=WARMUP_MS, rep=REP_MS,
            quantiles=[0.5, 0.2, 0.8])
        mem = peak_mem_mib(fn, q, k, v, bwd, fwd)
        return {**row, "latency_ms": med, "latency_p20": p20, "latency_p80": p80,
                "tflops": flops(n, bwd, fwd) / (med * 1e-3) / 1e12,
                "peak_mib": mem, "max_abs_err": err, "status": "ok"}
    except torch.cuda.OutOfMemoryError:
        torch.cuda.empty_cache()
        return {**row, **EMPTY, "status": "oom"}
    except Exception as e:
        torch.cuda.empty_cache()
        return {**row, **EMPTY, "status": log_error(variant, row["dtype"], e)}


def build_reference(q32, k32, v32, bwd, fwd, naive_fits):
    if naive_fits:
        try:
            ref = swa_naive(q32, k32, v32, (bwd, fwd))
            torch.cuda.synchronize()
            return ref, "naive_pt"
        except torch.cuda.OutOfMemoryError:
            torch.cuda.empty_cache()
    try:
        ref = swa_strided_pt(q32, k32, v32, (bwd, fwd))
        torch.cuda.synchronize()
        return ref, "strided_pt"
    except torch.cuda.OutOfMemoryError:
        torch.cuda.empty_cache()
        return None, "none"


def run_sweep():
    torch.manual_seed(0)
    rows = []

    for n in SEQ_LENS:
        for bwd, fwd in WINDOWS:
            naive_fits = naive_gib(n) < NAIVE_BUDGET_GIB
            q32 = torch.randn(B, H, n, D, device="cuda")
            k32 = torch.randn(B, H, n, D, device="cuda")
            v32 = torch.randn(B, H, n, D, device="cuda")
            ref, ref_source = build_reference(q32, k32, v32, bwd, fwd, naive_fits)

            for dtype in DTYPES:
                q, k, v = (q32, k32, v32) if dtype == torch.float32 else (
                    q32.to(dtype), k32.to(dtype), v32.to(dtype))
                dtype_name = str(dtype).rsplit(".", 1)[-1]

                for variant, fn in VARIANTS.items():
                    if variant in FP32_ONLY_VARIANTS and dtype != torch.float32:
                        continue
                    row = {"variant": variant, "dtype": dtype_name, "N": n,
                           "bwd": bwd, "fwd": fwd, "B": B, "H": H, "D": D,
                           "ref_source": "self" if variant == ref_source else ref_source}
                    if variant in QUADRATIC_VARIANTS and not naive_fits:
                        row = {**row, **EMPTY,
                               "status": f"skipped:budget({naive_gib(n):.1f}GiB)"}
                    else:
                        row = bench_one(variant, fn, n, bwd, fwd, ref, row, q, k, v)
                    rows.append(row)
                    print(row, flush=True)
                    torch.cuda.empty_cache()

                if dtype != torch.float32:
                    del q, k, v
                    torch.cuda.empty_cache()

            del q32, k32, v32, ref
            torch.cuda.empty_cache()

    return rows


def main():
    if not torch.cuda.is_available():
        sys.exit("CUDA not available.")

    props = torch.cuda.get_device_properties(0)
    print(f"{props.name}  CC {props.major}.{props.minor}  "
          f"torch {torch.__version__}  triton {triton.__version__}")
    if not HAS_FLEX:
        print("FlexAttention unavailable, skipping that variant.", file=sys.stderr)

    t0 = time.time()
    rows = run_sweep()
    print(f"done in {time.time() - t0:.1f}s, {len(rows)} rows, "
          f"ok={sum(1 for r in rows if r['status'] == 'ok')}")

    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    with open(OUTPUT, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    print(f"wrote {OUTPUT}")
    if LOGGED_ERRORS:
        print(f"tracebacks in {ERROR_LOG}: {sorted(LOGGED_ERRORS)}")


if __name__ == "__main__":
    main()