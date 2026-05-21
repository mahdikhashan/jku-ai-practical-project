"""
Benchmark FlexAttention for sliding window attention -- standalone.

This script is split out from benchmark_swa.py because FlexAttention requires
torch.compile + Inductor + Triton compatibility that the main benchmark
machine may not have (e.g. CUDA toolkit version that the installed Triton
doesn't recognise yet). Running it separately means a broken FlexAttention
setup doesn't take down the rest of the sweep.

Measurements per (dtype, N, window):
  * Latency       : mean ms per forward, including the per-call block-mask
                    construction. This matches the main benchmark's
                    treatment of FlexAttention; cache the mask externally
                    if you want steady-state-only numbers.
  * Throughput    : TFLOPS from latency, using FLOPs = 4 * N * W * D * H * B.
  * Peak memory   : torch.cuda.max_memory_allocated for one forward pass.
  * Numerical err : max-abs deviation from an FP32 reference (naive_pt if
                    it fits the budget, strided_pt as fallback).

Failures handled:
  * Block-mask construction OOM     -> status="oom:block_mask"
  * Forward pass OOM                -> status="oom"
  * Inductor / Triton compile error -> status="error:<ExceptionClass>"
                                       (full traceback to stderr)

Output: results/swa_bench_flex.csv, schema-compatible with swa_bench.csv so
the two can be concatenated for plotting.
"""

from __future__ import annotations

import argparse
import csv
import sys
import time
import traceback
from pathlib import Path

import torch
import triton

from swa_torch_naive import swa_naive
from swa_torch_strided import swa_strided as swa_strided_pt


# ---------------------------------------------------------------------------
# FlexAttention setup
# ---------------------------------------------------------------------------

try:
    from torch.nn.attention.flex_attention import flex_attention, create_block_mask
    _flex_compiled = torch.compile(flex_attention)
    _HAS_FLEX = True
except Exception as e:
    _HAS_FLEX = False
    print(f"[fatal] FlexAttention import failed: {e!r}", file=sys.stderr)
    print("        Install a PyTorch + Triton + CUDA combo that supports "
          "torch.nn.attention.flex_attention.", file=sys.stderr)


def _swa_flex(q, k, v, bwd, fwd):
    B, H, N, _ = q.shape
    mask = create_block_mask(
        lambda b, h, qi, ki: (qi - ki <= bwd) & (ki - qi <= fwd),
        B=B, H=H, Q_LEN=N, KV_LEN=N, device=q.device,
    )
    return _flex_compiled(q, k, v, block_mask=mask)


# ---------------------------------------------------------------------------
# Helpers (duplicated from benchmark_swa.py for standalone use)
# ---------------------------------------------------------------------------

def _naive_size_gib(N: int, H: int, B: int) -> float:
    return 3.0 * B * H * N * N * 4 / (1024 ** 3)


def _peak_mem_mib(fn, *args) -> float:
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    fn(*args)
    torch.cuda.synchronize()
    return torch.cuda.max_memory_allocated() / (1024 ** 2)


def _build_reference(q32, k32, v32, bwd, fwd, naive_fits):
    if naive_fits:
        try:
            ref = swa_naive(q32, k32, v32, (bwd, fwd))
            torch.cuda.synchronize()
            return ref, "naive"
        except torch.cuda.OutOfMemoryError:
            torch.cuda.empty_cache()
    try:
        ref = swa_strided_pt(q32, k32, v32, (bwd, fwd))
        torch.cuda.synchronize()
        return ref, "strided"
    except torch.cuda.OutOfMemoryError:
        torch.cuda.empty_cache()
        return None, "none"


def _measure_flex(q, k, v, bwd, fwd, ref):
    """Run + benchmark FlexAttention. Returns a dict of metrics + status."""
    try:
        out = _swa_flex(q, k, v, bwd, fwd)
        torch.cuda.synchronize()
        err = float("nan") if ref is None else (out.float() - ref).abs().max().item()
        lat = triton.testing.do_bench(
            lambda: _swa_flex(q, k, v, bwd, fwd), warmup=5, rep=25,
        )
        mem = _peak_mem_mib(_swa_flex, q, k, v, bwd, fwd)
        return {"latency_ms": lat, "peak_mib": mem, "max_abs_err": err, "status": "ok"}
    except torch.cuda.OutOfMemoryError as e:
        torch.cuda.empty_cache()
        # Distinguish OOM during block_mask vs OOM during the kernel.
        # create_block_mask is the first allocation in _swa_flex, so if the
        # traceback's deepest frame mentions create_block_mask we tag it.
        tag = "oom:block_mask" if "create_block_mask" in traceback.format_exc() else "oom"
        return {"latency_ms": float("nan"), "peak_mib": float("nan"),
                "max_abs_err": float("nan"), "status": tag}
    except Exception as e:
        torch.cuda.empty_cache()
        traceback.print_exc(file=sys.stderr)
        return {"latency_ms": float("nan"), "peak_mib": float("nan"),
                "max_abs_err": float("nan"), "status": f"error:{type(e).__name__}"}


def _fmt(row: dict) -> str:
    if row["status"] == "ok":
        return (f"flex   {row['dtype']:<5s} N={row['N']:<6d} "
                f"w=({row['bwd']},{row['fwd']})  "
                f"lat={row['latency_ms']:7.2f}ms  "
                f"tflops={row['tflops']:5.2f}  "
                f"mem={row['peak_mib']:7.1f}MiB  "
                f"err={row['max_abs_err']:.2e}  "
                f"ref={row['ref_source']}")
    return (f"flex   {row['dtype']:<5s} N={row['N']:<6d} "
            f"w=({row['bwd']},{row['fwd']})  {row['status']}")


# ---------------------------------------------------------------------------
# Sweep
# ---------------------------------------------------------------------------

def run_sweep(seq_lens, windows, dtypes, B, H, D, naive_budget_gib):
    rows = []
    device = "cuda"

    for N in seq_lens:
        for bwd, fwd in windows:
            naive_size = _naive_size_gib(N, H, B)
            naive_fits = naive_size < naive_budget_gib
            if not naive_fits:
                print(f"[info] N={N} w=({bwd},{fwd}): naive over budget "
                      f"(~{naive_size:.1f} GiB); using strided as reference.")

            torch.manual_seed(0)
            q32 = torch.randn(B, H, N, D, device=device, dtype=torch.float32)
            k32 = torch.randn(B, H, N, D, device=device, dtype=torch.float32)
            v32 = torch.randn(B, H, N, D, device=device, dtype=torch.float32)

            ref, ref_src = _build_reference(q32, k32, v32, bwd, fwd, naive_fits)

            for dtype in dtypes:
                if dtype == torch.float32:
                    q, k, v = q32, k32, v32
                else:
                    q, k, v = q32.to(dtype), k32.to(dtype), v32.to(dtype)

                row = {
                    "variant": "flex", "dtype": str(dtype).rsplit(".", 1)[-1],
                    "N": N, "bwd": bwd, "fwd": fwd, "B": B, "H": H, "D": D,
                    "ref_source": ref_src,
                }
                res = _measure_flex(q, k, v, bwd, fwd, ref)
                if res["status"] == "ok":
                    W = 1 + bwd + fwd
                    res["tflops"] = (4 * N * W * D * H * B) / (res["latency_ms"] * 1e-3) / 1e12
                else:
                    res["tflops"] = float("nan")
                row.update(res)

                rows.append(row)
                print(_fmt(row), flush=True)
                torch.cuda.empty_cache()

                if dtype != torch.float32:
                    del q, k, v
                    torch.cuda.empty_cache()

            del q32, k32, v32
            if ref is not None:
                del ref
            torch.cuda.empty_cache()

    return rows


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main():
    if not _HAS_FLEX:
        sys.exit(1)

    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--seq-lens", nargs="+", type=int,
                    default=[1024, 2048, 4096, 8192, 16384])
    ap.add_argument("--windows", nargs="+",
                    default=["15,16", "127,128", "255,256"])
    ap.add_argument("--dtypes", nargs="+", default=["fp32", "fp16"],
                    choices=["fp32", "fp16"])
    ap.add_argument("--batch", type=int, default=1)
    ap.add_argument("--heads", type=int, default=8)
    ap.add_argument("--head-dim", type=int, default=64)
    ap.add_argument("--naive-budget-gib", type=float, default=18.0)
    ap.add_argument("--output", default="results/swa_bench_flex.csv")
    args = ap.parse_args()

    if not torch.cuda.is_available():
        sys.exit("CUDA not available.")

    dtype_map = {"fp32": torch.float32, "fp16": torch.float16}
    dtypes  = [dtype_map[d] for d in args.dtypes]
    windows = [tuple(int(x) for x in w.split(",")) for w in args.windows]

    print(f"device: {torch.cuda.get_device_name(0)}  "
          f"torch: {torch.__version__}  triton: {triton.__version__}")
    print(f"B={args.batch} H={args.heads} D={args.head_dim}  "
          f"budget={args.naive_budget_gib:.1f}GiB")
    print(f"[note] first call per (N, window) compiles via Inductor; "
          f"expect a few seconds of latency on the warmup pass.\n")

    t0 = time.time()
    rows = run_sweep(args.seq_lens, windows, dtypes,
                     args.batch, args.heads, args.head_dim,
                     args.naive_budget_gib)
    print(f"\nDone in {time.time() - t0:.1f}s; {len(rows)} rows.")

    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    print(f"Wrote {args.output}")


if __name__ == "__main__":
    main()
