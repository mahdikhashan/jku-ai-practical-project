import csv
import sys
import traceback
from contextlib import contextmanager
from functools import lru_cache
from pathlib import Path

import torch
import triton
from torch.nn.attention.flex_attention import create_block_mask, flex_attention

from swa_torch_naive import swa_naive
from swa_torch_strided import swa_strided as swa_strided_pt

SEQ_LENS = [1024, 2048, 4096, 8192, 16384]
WINDOWS = [(15, 16), (31, 32), (63, 64), (127, 128), (255, 256)]
DTYPES = [torch.float32, torch.float16]
FP32_PRECISIONS = ["highest", "high"]
B, H, D = 1, 8, 64
NAIVE_BUDGET_GIB = 18.0
OUTPUT = Path("results/swa_bench_flex.csv")

torch._dynamo.config.recompile_limit = 128

flex_attention_compiled = torch.compile(flex_attention, dynamic=False)
create_block_mask_compiled = torch.compile(create_block_mask, dynamic=False)


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


@contextmanager
def fp32_precision(precision):
    prev = torch.get_float32_matmul_precision()
    torch.set_float32_matmul_precision(precision)
    try:
        yield
    finally:
        torch.set_float32_matmul_precision(prev)


def naive_fits_budget(n):
    return 3.0 * B * H * n * n * 4 / (1024 ** 3) < NAIVE_BUDGET_GIB


def peak_mem_mib(fn, *args):
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    fn(*args)
    torch.cuda.synchronize()
    return torch.cuda.max_memory_allocated() / (1024 ** 2)


def build_reference(q32, k32, v32, bwd, fwd):
    if naive_fits_budget(q32.shape[2]):
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


def measure(q, k, v, bwd, fwd, ref, precision):
    try:
        with fp32_precision(precision):
            out = swa_flex(q, k, v, bwd, fwd)
            torch.cuda.synchronize()
            err = float("nan") if ref is None else (out.float() - ref).abs().max().item()
            lat = triton.testing.do_bench(lambda: swa_flex(q, k, v, bwd, fwd), warmup=5, rep=25)
            mem = peak_mem_mib(swa_flex, q, k, v, bwd, fwd)
        return {"latency_ms": lat, "peak_mib": mem, "max_abs_err": err, "status": "ok"}
    except torch.cuda.OutOfMemoryError:
        torch.cuda.empty_cache()
        return {"latency_ms": float("nan"), "peak_mib": float("nan"),
                "max_abs_err": float("nan"), "status": "oom"}
    except Exception as e:
        torch.cuda.empty_cache()
        traceback.print_exc(file=sys.stderr)
        return {"latency_ms": float("nan"), "peak_mib": float("nan"),
                "max_abs_err": float("nan"), "status": f"error:{type(e).__name__}"}


def run_sweep():
    rows = []
    for n in SEQ_LENS:
        for bwd, fwd in WINDOWS:
            torch.manual_seed(0)
            q32 = torch.randn(B, H, n, D, device="cuda")
            k32 = torch.randn(B, H, n, D, device="cuda")
            v32 = torch.randn(B, H, n, D, device="cuda")
            ref, ref_src = build_reference(q32, k32, v32, bwd, fwd)

            for dtype in DTYPES:
                q, k, v = (q32, k32, v32) if dtype == torch.float32 else (
                    q32.to(dtype), k32.to(dtype), v32.to(dtype))
                precisions = FP32_PRECISIONS if dtype == torch.float32 else ["n/a"]

                for precision in precisions:
                    row = {"variant": "flex", "dtype": str(dtype).rsplit(".", 1)[-1],
                           "precision": precision,
                           "N": n, "bwd": bwd, "fwd": fwd, "B": B, "H": H, "D": D,
                           "ref_source": ref_src}
                    res = measure(q, k, v, bwd, fwd, ref,
                                  "highest" if precision == "n/a" else precision)
                    w = 1 + bwd + fwd
                    res["tflops"] = ((4 * n * w * D * H * B) / (res["latency_ms"] * 1e-3) / 1e12
                                      if res["status"] == "ok" else float("nan"))
                    row.update(res)
                    rows.append(row)
                    print(row, flush=True)
                    torch.cuda.empty_cache()

            del q32, k32, v32, ref
            torch.cuda.empty_cache()

    return rows


def main():
    if not torch.cuda.is_available():
        sys.exit("CUDA not available.")

    rows = run_sweep()

    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    with open(OUTPUT, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    print(f"wrote {OUTPUT}")


if __name__ == "__main__":
    main()