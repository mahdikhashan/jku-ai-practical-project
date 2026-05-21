#!/usr/bin/env python3
"""
Generate meaningful benchmark plots for sliding window attention results.

Usage:
    python plot_swa_results.py --csv results/swa_bench.csv --outdir plots

This script generates:
1. TFLOPS vs sequence length
2. Latency vs sequence length
3. Memory usage vs sequence length
4. TFLOPS vs window size
5. Speedup over baseline
6. Accuracy/error plots
7. Scalability plots
8. Heatmaps
9. Combined summary figures

All plots are generated for:
- float16 / float32
- different window sizes
- all implementations
"""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# -----------------------------
# Utilities
# -----------------------------

VARIANT_NAMES = {
    "naive_pt": "Naive PyTorch",
    "strided_pt": "Strided PyTorch",
    "triton_tiled_fp32": "Triton FP32",
    "triton_tiled_fp16": "Triton FP16",
    "flex": "FlexAttention",
}

MARKERS = {
    "naive_pt": "o",
    "strided_pt": "s",
    "triton_tiled_fp32": "^",
    "triton_tiled_fp16": "D",
    "flex": "x",
}


def prepare_dataframe(csv_path):
    df = pd.read_csv(csv_path)

    # Keep only successful runs
    df_ok = df[df["status"] == "ok"].copy()

    # Window size = forward window
    df_ok["window"] = df_ok["fwd"]

    # Clean names
    df_ok["variant_pretty"] = df_ok["variant"].map(VARIANT_NAMES)

    return df_ok


def savefig(path):
    plt.tight_layout()
    plt.savefig(path, dpi=300, bbox_inches="tight")
    plt.close()


# -----------------------------
# Plot Functions
# -----------------------------

def plot_tflops_vs_n(df, outdir):
    """
    TFLOPS vs sequence length for each window size and dtype.
    """
    for dtype in sorted(df["dtype"].unique()):
        for window in sorted(df["window"].unique()):

            sub = df[(df["dtype"] == dtype) & (df["window"] == window)]

            plt.figure(figsize=(9, 6))

            for variant in sub["variant"].unique():
                vdf = sub[sub["variant"] == variant].sort_values("N")

                plt.plot(
                    vdf["N"],
                    vdf["tflops"],
                    marker=MARKERS.get(variant, "o"),
                    linewidth=2,
                    markersize=7,
                    label=VARIANT_NAMES.get(variant, variant),
                )

            plt.xscale("log", base=2)

            plt.xlabel("Sequence Length (N)")
            plt.ylabel("TFLOPS")
            plt.title(f"TFLOPS vs Sequence Length ({dtype}, window={window})")

            plt.grid(True, alpha=0.3)
            plt.legend()

            savefig(outdir / f"tflops_vs_n_{dtype}_w{window}.png")


def plot_latency_vs_n(df, outdir):
    """
    Latency scaling.
    """
    for dtype in sorted(df["dtype"].unique()):
        for window in sorted(df["window"].unique()):

            sub = df[(df["dtype"] == dtype) & (df["window"] == window)]

            plt.figure(figsize=(9, 6))

            for variant in sub["variant"].unique():
                vdf = sub[sub["variant"] == variant].sort_values("N")

                plt.plot(
                    vdf["N"],
                    vdf["latency_ms"],
                    marker=MARKERS.get(variant, "o"),
                    linewidth=2,
                    markersize=7,
                    label=VARIANT_NAMES.get(variant, variant),
                )

            plt.xscale("log", base=2)
            plt.yscale("log")

            plt.xlabel("Sequence Length (N)")
            plt.ylabel("Latency (ms)")
            plt.title(f"Latency Scaling ({dtype}, window={window})")

            plt.grid(True, which="both", alpha=0.3)
            plt.legend()

            savefig(outdir / f"latency_vs_n_{dtype}_w{window}.png")


def plot_memory_vs_n(df, outdir):
    """
    Peak memory usage scaling.
    """
    for dtype in sorted(df["dtype"].unique()):
        for window in sorted(df["window"].unique()):

            sub = df[(df["dtype"] == dtype) & (df["window"] == window)]

            plt.figure(figsize=(9, 6))

            for variant in sub["variant"].unique():
                vdf = sub[sub["variant"] == variant].sort_values("N")

                plt.plot(
                    vdf["N"],
                    vdf["peak_mib"],
                    marker=MARKERS.get(variant, "o"),
                    linewidth=2,
                    markersize=7,
                    label=VARIANT_NAMES.get(variant, variant),
                )

            plt.xscale("log", base=2)
            plt.yscale("log")

            plt.xlabel("Sequence Length (N)")
            plt.ylabel("Peak Memory (MiB)")
            plt.title(f"Memory Scaling ({dtype}, window={window})")

            plt.grid(True, which="both", alpha=0.3)
            plt.legend()

            savefig(outdir / f"memory_vs_n_{dtype}_w{window}.png")


def plot_tflops_vs_window(df, outdir):
    """
    TFLOPS vs window size for fixed sequence lengths.
    """
    for dtype in sorted(df["dtype"].unique()):
        for N in sorted(df["N"].unique()):

            sub = df[(df["dtype"] == dtype) & (df["N"] == N)]

            plt.figure(figsize=(9, 6))

            for variant in sub["variant"].unique():
                vdf = sub[sub["variant"] == variant].sort_values("window")

                plt.plot(
                    vdf["window"],
                    vdf["tflops"],
                    marker=MARKERS.get(variant, "o"),
                    linewidth=2,
                    markersize=7,
                    label=VARIANT_NAMES.get(variant, variant),
                )

            plt.xscale("log", base=2)

            plt.xlabel("Window Size")
            plt.ylabel("TFLOPS")
            plt.title(f"TFLOPS vs Window Size ({dtype}, N={N})")

            plt.grid(True, alpha=0.3)
            plt.legend()

            savefig(outdir / f"tflops_vs_window_{dtype}_N{N}.png")


def plot_speedup(df, outdir):
    """
    Speedup over naive/strided baseline.
    """
    speedup_rows = []

    for _, row in df.iterrows():

        if row["variant"] in ["naive_pt", "strided_pt"]:
            continue

        baseline_name = (
            "naive_pt"
            if row["N"] < 16384
            else "strided_pt"
        )

        baseline = df[
            (df["variant"] == baseline_name)
            & (df["dtype"] == row["dtype"])
            & (df["N"] == row["N"])
            & (df["window"] == row["window"])
        ]

        if len(baseline) == 0:
            continue

        baseline_latency = baseline.iloc[0]["latency_ms"]

        speedup_rows.append({
            "variant": row["variant"],
            "dtype": row["dtype"],
            "N": row["N"],
            "window": row["window"],
            "speedup": baseline_latency / row["latency_ms"],
        })

    sdf = pd.DataFrame(speedup_rows)

    for dtype in sorted(sdf["dtype"].unique()):
        for window in sorted(sdf["window"].unique()):

            sub = sdf[
                (sdf["dtype"] == dtype)
                & (sdf["window"] == window)
            ]

            plt.figure(figsize=(9, 6))

            for variant in sub["variant"].unique():
                vdf = sub[sub["variant"] == variant].sort_values("N")

                plt.plot(
                    vdf["N"],
                    vdf["speedup"],
                    marker=MARKERS.get(variant, "o"),
                    linewidth=2,
                    markersize=7,
                    label=VARIANT_NAMES.get(variant, variant),
                )

            plt.xscale("log", base=2)
            plt.yscale("log")

            plt.xlabel("Sequence Length (N)")
            plt.ylabel("Speedup")
            plt.title(f"Speedup over Baseline ({dtype}, window={window})")

            plt.grid(True, which="both", alpha=0.3)
            plt.legend()

            savefig(outdir / f"speedup_{dtype}_w{window}.png")


def plot_accuracy(df, outdir):
    """
    Numerical error comparison.
    """
    adf = df[df["max_abs_err"].notna()]

    for dtype in sorted(adf["dtype"].unique()):

        plt.figure(figsize=(10, 6))

        for variant in adf["variant"].unique():

            if variant == "naive_pt":
                continue

            vdf = (
                adf[
                    (adf["dtype"] == dtype)
                    & (adf["variant"] == variant)
                ]
                .sort_values("N")
            )

            plt.scatter(
                vdf["N"],
                vdf["max_abs_err"],
                s=80,
                label=VARIANT_NAMES.get(variant, variant),
            )

        plt.xscale("log", base=2)
        plt.yscale("log")

        plt.xlabel("Sequence Length (N)")
        plt.ylabel("Max Absolute Error")
        plt.title(f"Numerical Accuracy ({dtype})")

        plt.grid(True, which="both", alpha=0.3)
        plt.legend()

        savefig(outdir / f"accuracy_{dtype}.png")


def plot_heatmap(df, outdir):
    """
    Heatmap of TFLOPS for Triton kernels.
    """
    for dtype in sorted(df["dtype"].unique()):

        sub = df[
            (df["dtype"] == dtype)
            & (
                df["variant"].isin(
                    ["triton_tiled_fp32", "triton_tiled_fp16"]
                )
            )
        ]

        pivot = sub.pivot_table(
            index="N",
            columns="window",
            values="tflops",
            aggfunc="max",
        )

        plt.figure(figsize=(8, 6))

        plt.imshow(pivot, aspect="auto")

        plt.colorbar(label="TFLOPS")

        plt.xticks(
            range(len(pivot.columns)),
            pivot.columns,
        )

        plt.yticks(
            range(len(pivot.index)),
            pivot.index,
        )

        plt.xlabel("Window Size")
        plt.ylabel("Sequence Length (N)")
        plt.title(f"Triton TFLOPS Heatmap ({dtype})")

        savefig(outdir / f"heatmap_triton_{dtype}.png")


def plot_summary(df, outdir):
    """
    Single publication-style summary figure.
    """

    for dtype in sorted(df["dtype"].unique()):

        fig, axs = plt.subplots(1, 3, figsize=(18, 5))

        # --------------------------------
        # TFLOPS
        # --------------------------------

        ax = axs[0]

        sub = df[
            (df["dtype"] == dtype)
            & (df["window"] == 256)
        ]

        for variant in sub["variant"].unique():

            vdf = sub[sub["variant"] == variant].sort_values("N")

            ax.plot(
                vdf["N"],
                vdf["tflops"],
                marker=MARKERS.get(variant, "o"),
                linewidth=2,
                label=VARIANT_NAMES.get(variant, variant),
            )

        ax.set_xscale("log", base=2)
        ax.set_title("TFLOPS")
        ax.set_xlabel("N")
        ax.grid(True, alpha=0.3)

        # --------------------------------
        # Latency
        # --------------------------------

        ax = axs[1]

        for variant in sub["variant"].unique():

            vdf = sub[sub["variant"] == variant].sort_values("N")

            ax.plot(
                vdf["N"],
                vdf["latency_ms"],
                marker=MARKERS.get(variant, "o"),
                linewidth=2,
                label=VARIANT_NAMES.get(variant, variant),
            )

        ax.set_xscale("log", base=2)
        ax.set_yscale("log")
        ax.set_title("Latency")
        ax.set_xlabel("N")
        ax.grid(True, which="both", alpha=0.3)

        # --------------------------------
        # Memory
        # --------------------------------

        ax = axs[2]

        for variant in sub["variant"].unique():

            vdf = sub[sub["variant"] == variant].sort_values("N")

            ax.plot(
                vdf["N"],
                vdf["peak_mib"],
                marker=MARKERS.get(variant, "o"),
                linewidth=2,
                label=VARIANT_NAMES.get(variant, variant),
            )

        ax.set_xscale("log", base=2)
        ax.set_yscale("log")
        ax.set_title("Peak Memory")
        ax.set_xlabel("N")
        ax.grid(True, which="both", alpha=0.3)

        handles, labels = axs[0].get_legend_handles_labels()

        fig.legend(
            handles,
            labels,
            loc="upper center",
            ncol=5,
        )

        fig.suptitle(f"Sliding Window Attention Benchmark Summary ({dtype})")

        plt.tight_layout(rect=[0, 0, 1, 0.92])

        plt.savefig(
            outdir / f"summary_{dtype}.png",
            dpi=300,
            bbox_inches="tight",
        )

        plt.close()


# -----------------------------
# Main
# -----------------------------

def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--csv",
        required=True,
        help="Path to CSV benchmark results",
    )

    parser.add_argument(
        "--outdir",
        default="plots",
        help="Output directory",
    )

    args = parser.parse_args()

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    df = prepare_dataframe(args.csv)

    print("Generating plots...")

    plot_tflops_vs_n(df, outdir)
    plot_latency_vs_n(df, outdir)
    plot_memory_vs_n(df, outdir)

    plot_tflops_vs_window(df, outdir)

    plot_speedup(df, outdir)

    plot_accuracy(df, outdir)

    plot_heatmap(df, outdir)

    plot_summary(df, outdir)

    print(f"Plots saved to: {outdir.resolve()}")


if __name__ == "__main__":
    main()
