"""
Benchmark sliding window attention implementations.

Measures, per (variant, precision, N, window):
  * Latency       : mean wall-clock ms per forward pass.
  * Throughput    : TFLOPS, derived from latency using
                       FLOPs = 4 * N * W * D * H * B
                    where W = 1 + bwd + fwd is the effective window width.
                    The factor of 4 = 2 (QK^T) + 2 (PV); see Section 4.5 of
                    the report.
  * Peak memory   : torch.cuda.max_memory_allocated after a single forward.
  * Numerical err : max absolute deviation from the FP32 naive reference.

Variants benchmarked:
  - naive_pt       : PyTorch naive mask-based attention (O(N^2)).
  - strided_pt     : PyTorch strided gather (O(N*W) memory, no kernel fusion).
  - flex           : torch.nn.attention.flex_attention with block-sparse mask.
  - triton_tiled   : the tiled Triton kernels in swa_triton_tiled.py.

The script writes a CSV to results/swa_bench.csv and prints a summary table.

Usage:
    python benchmark_swa.py                    # default sweep
    python benchmark_swa.py --quick            # small sweep for sanity
    python benchmark_swa.py --variants naive_pt strided_pt triton_tiled
"""

from __future__ import annotations

import argparse
import csv
import math
import os
import sys
import time
import traceback
import warnings
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Callable

import torch
import triton

# Reference implementations
from swa_torch_naive import swa_naive
from swa_torch_strided import swa_strided as swa_strided_pt

# Triton kernels
from swa_triton_custom_kernel import swa_tiled_triton_fp32, swa_tiled_triton_fp16

# FlexAttention (optional)
try:
    from torch.nn.attention.flex_attention import flex_attention, create_block_mask
    _HAS_FLEX = True
    _flex_compiled = torch.compile(flex_attention)
except Exception as e:
    _HAS_FLEX = False
    _flex_import_error = repr(e)


# ===========================================================================
# Variant adapters
# ===========================================================================

def _swa_flex(q, k, v, bwd, fwd):
    B, H, N, _ = q.shape
    def sliding_window(b, h, q_idx, kv_idx):
        return (q_idx - kv_idx <= bwd) & (kv_idx - q_idx <= fwd)
    block_mask = create_block_mask(
        sliding_window, B=B, H=H, Q_LEN=N, KV_LEN=N, device=q.device,
    )
    return _flex_compiled(q, k, v, block_mask=block_mask)


VARIANTS: dict[str, Callable] = {
    "naive_pt":     lambda q, k, v, bwd, fwd: swa_naive(q, k, v, (bwd, fwd)),
    "strided_pt":   lambda q, k, v, bwd, fwd: swa_strided_pt(q, k, v, (bwd, fwd)),
    "triton_tiled_fp32": lambda q, k, v, bwd, fwd: swa_tiled_triton_fp32(q, k, v, bwd, fwd),
    "triton_tiled_fp16": lambda q, k, v, bwd, fwd: swa_tiled_triton_fp16(q, k, v, bwd, fwd),
}
if _HAS_FLEX:
    VARIANTS["flex"] = _swa_flex


# Variants that only support a specific dtype.
_DTYPE_RESTRICTION = {
    "triton_tiled_fp32": torch.float32,
    "triton_tiled_fp16": torch.float16,
}


# ===========================================================================
# Result record
# ===========================================================================

@dataclass
class BenchResult:
    variant:     str
    dtype:       str
    N:           int
    bwd:         int
    fwd:         int
    B:           int
    H:           int
    D:           int
    latency_ms:  float
    tflops:      float
    peak_mib:    float
    max_abs_err: float
    status:      str  # "ok", "oom", "skipped:<reason>", "error:<msg>"


# ===========================================================================
# Core benchmarking
# ===========================================================================

def _flops(N: int, bwd: int, fwd: int, B: int, H: int, D: int) -> float:
    W = 1 + bwd + fwd
    # 2 * N * W * D for QK^T, 2 * N * W * D for PV, times B * H.
    return 4.0 * N * W * D * H * B


def _measure(fn, *args, warmup: int = 5, rep: int = 25) -> float:
    """Latency in milliseconds, using triton.testing.do_bench."""
    return triton.testing.do_bench(lambda: fn(*args), warmup=warmup, rep=rep)


def _peak_memory(fn, *args) -> float:
    """Peak memory in MiB during one forward pass."""
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    fn(*args)
    torch.cuda.synchronize()
    return torch.cuda.max_memory_allocated() / (1024 ** 2)


def _max_abs_err(out: torch.Tensor, ref_fp32: torch.Tensor) -> float:
    return (out.float() - ref_fp32).abs().max().item()


# Module-level error log -- one full traceback per (variant, dtype) the first
# time it fails. Truncating these to 100 chars in the on-screen status was
# hiding the actual cause; the on-screen tag stays short but the full text
# goes to results/swa_bench_errors.log.
_LOGGED_ERRORS: set[tuple[str, str]] = set()
_ERROR_LOG_PATH = "results/swa_bench_errors.log"

def _log_error(variant: str, dtype_name: str, exc: BaseException) -> str:
    """Write full traceback once per (variant, dtype); return short tag for the row."""
    key = (variant, dtype_name)
    if key not in _LOGGED_ERRORS:
        _LOGGED_ERRORS.add(key)
        Path(_ERROR_LOG_PATH).parent.mkdir(parents=True, exist_ok=True)
        with open(_ERROR_LOG_PATH, "a") as f:
            f.write(f"\n{'=' * 78}\n")
            f.write(f"variant={variant}  dtype={dtype_name}\n")
            f.write(f"{'=' * 78}\n")
            f.write("".join(traceback.format_exception(type(exc), exc, exc.__traceback__)))
            f.write("\n")
    # Short tag for the on-screen status column.
    return f"error:{type(exc).__name__}"


def bench_one(
    variant: str,
    fn: Callable,
    dtype: torch.dtype,
    B: int, H: int, N: int, D: int,
    bwd: int, fwd: int,
    ref_fp32: torch.Tensor,
    q: torch.Tensor, k: torch.Tensor, v: torch.Tensor,
) -> BenchResult:
    dtype_name = str(dtype).split(".")[-1]
    base = dict(variant=variant, dtype=dtype_name, N=N, bwd=bwd, fwd=fwd,
                B=B, H=H, D=D)

    # Dtype restriction check
    required = _DTYPE_RESTRICTION.get(variant)
    if required is not None and required != dtype:
        return BenchResult(**base, latency_ms=float("nan"), tflops=float("nan"),
                           peak_mib=float("nan"), max_abs_err=float("nan"),
                           status=f"skipped:dtype_mismatch")

    try:
        out = fn(q, k, v, bwd, fwd)
        torch.cuda.synchronize()
        err = _max_abs_err(out, ref_fp32)

        latency_ms = _measure(fn, q, k, v, bwd, fwd)
        peak_mib   = _peak_memory(fn, q, k, v, bwd, fwd)
        tflops     = _flops(N, bwd, fwd, B, H, D) / (latency_ms * 1e-3) / 1e12

        return BenchResult(**base,
                           latency_ms=latency_ms,
                           tflops=tflops,
                           peak_mib=peak_mib,
                           max_abs_err=err,
                           status="ok")
    except torch.cuda.OutOfMemoryError:
        torch.cuda.empty_cache()
        return BenchResult(**base, latency_ms=float("nan"), tflops=float("nan"),
                           peak_mib=float("nan"), max_abs_err=float("nan"),
                           status="oom")
    except Exception as e:
        torch.cuda.empty_cache()
        tag = _log_error(variant, dtype_name, e)
        return BenchResult(**base, latency_ms=float("nan"), tflops=float("nan"),
                           peak_mib=float("nan"), max_abs_err=float("nan"),
                           status=tag)


# ===========================================================================
# Sweep
# ===========================================================================

def run_sweep(
    variants: list[str],
    dtypes: list[torch.dtype],
    seq_lens: list[int],
    windows: list[tuple[int, int]],
    B: int = 1, H: int = 8, D: int = 64,
    seed: int = 0,
) -> list[BenchResult]:
    torch.manual_seed(seed)
    device = "cuda"
    results: list[BenchResult] = []

    for N in seq_lens:
        for (bwd, fwd) in windows:
            # FP32 reference (computed once per (N, window) and reused).
            q_ref = torch.randn(B, H, N, D, device=device, dtype=torch.float32)
            k_ref = torch.randn(B, H, N, D, device=device, dtype=torch.float32)
            v_ref = torch.randn(B, H, N, D, device=device, dtype=torch.float32)
            try:
                ref_fp32 = swa_naive(q_ref, k_ref, v_ref, (bwd, fwd))
                torch.cuda.synchronize()
            except torch.cuda.OutOfMemoryError:
                # If even the reference OOMs, mark everything OOM for this N.
                torch.cuda.empty_cache()
                for v_name in variants:
                    for dt in dtypes:
                        results.append(BenchResult(
                            variant=v_name, dtype=str(dt).split(".")[-1],
                            N=N, bwd=bwd, fwd=fwd, B=B, H=H, D=D,
                            latency_ms=float("nan"), tflops=float("nan"),
                            peak_mib=float("nan"), max_abs_err=float("nan"),
                            status="oom:ref"))
                continue

            for dtype in dtypes:
                if dtype == torch.float32:
                    q, k, v = q_ref, k_ref, v_ref
                else:
                    q, k, v = q_ref.to(dtype), k_ref.to(dtype), v_ref.to(dtype)

                for v_name in variants:
                    if v_name not in VARIANTS:
                        continue
                    fn = VARIANTS[v_name]
                    res = bench_one(v_name, fn, dtype,
                                    B, H, N, D, bwd, fwd,
                                    ref_fp32, q, k, v)
                    results.append(res)
                    print(_fmt_row(res), flush=True)

            del q_ref, k_ref, v_ref, ref_fp32
            torch.cuda.empty_cache()

    return results


# ===========================================================================
# Output formatting
# ===========================================================================

_HEADER = (
    f"{'variant':<20s} {'dtype':<8s} {'N':>6s} "
    f"{'win':>9s} {'lat(ms)':>10s} {'TFLOPS':>8s} "
    f"{'mem(MiB)':>10s} {'max_err':>10s}  status"
)

def _fmt_row(r: BenchResult) -> str:
    win = f"({r.bwd},{r.fwd})"
    if r.status == "ok":
        return (f"{r.variant:<20s} {r.dtype:<8s} {r.N:>6d} "
                f"{win:>9s} {r.latency_ms:>10.3f} {r.tflops:>8.2f} "
                f"{r.peak_mib:>10.1f} {r.max_abs_err:>10.2e}  {r.status}")
    else:
        return (f"{r.variant:<20s} {r.dtype:<8s} {r.N:>6d} "
                f"{win:>9s} {'-':>10s} {'-':>8s} {'-':>10s} {'-':>10s}  {r.status}")


def write_csv(results: list[BenchResult], path: str) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(asdict(results[0]).keys()))
        writer.writeheader()
        for r in results:
            writer.writerow(asdict(r))


# ===========================================================================
# Entry point
# ===========================================================================

def parse_args():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--variants", nargs="+", default=None,
                    help=f"subset of: {list(VARIANTS.keys())}")
    ap.add_argument("--dtypes", nargs="+", default=["fp32", "fp16"],
                    choices=["fp32", "fp16"])
    ap.add_argument("--seq-lens", nargs="+", type=int,
                    default=[1024, 2048, 4096, 8192, 16384])
    ap.add_argument("--windows", nargs="+", default=["15,16", "127,128", "255,256"],
                    help="comma-separated bwd,fwd pairs")
    ap.add_argument("--batch", type=int, default=1)
    ap.add_argument("--heads", type=int, default=8)
    ap.add_argument("--head-dim", type=int, default=64)
    ap.add_argument("--output", default="results/swa_bench.csv")
    ap.add_argument("--quick", action="store_true",
                    help="small sweep for sanity checks")
    return ap.parse_args()


def main():
    args = parse_args()

    if not torch.cuda.is_available():
        print("CUDA not available; nothing to benchmark.", file=sys.stderr)
        sys.exit(1)

    if not _HAS_FLEX:
        print(f"[warn] FlexAttention not available: {_flex_import_error}",
              file=sys.stderr)

    if args.quick:
        seq_lens = [512, 1024, 2048]
        windows  = [(15, 16)]
    else:
        seq_lens = args.seq_lens
        windows  = [tuple(int(x) for x in w.split(",")) for w in args.windows]

    variants = args.variants or list(VARIANTS.keys())
    unknown = [v for v in variants if v not in VARIANTS]
    if unknown:
        print(f"[error] unknown variants: {unknown}", file=sys.stderr)
        print(f"        available: {list(VARIANTS.keys())}", file=sys.stderr)
        sys.exit(2)

    dtype_map = {"fp32": torch.float32, "fp16": torch.float16}
    dtypes = [dtype_map[d] for d in args.dtypes]

    # --- Setup info -------------------------------------------------------
    props = torch.cuda.get_device_properties(0)
    print(f"Device:    {props.name} ({props.total_memory / (1024**3):.1f} GiB)")
    print(f"CUDA cap:  {props.major}.{props.minor}")
    print(f"PyTorch:   {torch.__version__}")
    print(f"Triton:    {triton.__version__}")
    print(f"Variants:  {variants}")
    print(f"Dtypes:    {[str(d).split('.')[-1] for d in dtypes]}")
    print(f"Seq lens:  {seq_lens}")
    print(f"Windows:   {windows}")
    print(f"B={args.batch}, H={args.heads}, D={args.head_dim}")
    print()
    print(_HEADER)
    print("-" * len(_HEADER))

    t0 = time.time()
    results = run_sweep(
        variants=variants,
        dtypes=dtypes,
        seq_lens=seq_lens,
        windows=windows,
        B=args.batch, H=args.heads, D=args.head_dim,
    )
    elapsed = time.time() - t0

    print()
    print(f"Done in {elapsed:.1f}s. {len(results)} rows.")

    write_csv(results, args.output)
    print(f"Wrote {args.output}")
    if _LOGGED_ERRORS:
        print(f"Wrote full tracebacks for failed variants to {_ERROR_LOG_PATH}")
        print(f"   Failed: {sorted(_LOGGED_ERRORS)}")


if __name__ == "__main__":
    main()