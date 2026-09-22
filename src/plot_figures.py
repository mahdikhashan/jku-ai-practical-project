from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import ListedColormap

CSV = Path("results/swa_bench.csv")
OUT = Path("figures")

LABELS = {
    "naive_pt": "Naive PyTorch",
    "naive_pt_compiled": "Naive PyTorch (compiled)",
    "strided_pt": "Strided PyTorch",
    "strided_pt_compiled": "Strided PyTorch (compiled)",
    "triton_strided": "Triton strided",
    "triton_tiled": "Triton tiled",
    "flex": "FlexAttention",
    "flex_tf32": "FlexAttention (TF32)",
}
MARKERS = "osD^vP"
LIGHT, DARK = "#c6dbef", "#08306b"

plt.rcParams.update({"font.size": 9})


def load(path):
    df = pd.read_csv(path)
    df = df[df.status == "ok"].copy()
    df["W"] = 1 + df.bwd + df.fwd
    return df


def blues(n):
    return plt.cm.Blues(np.linspace(0.45, 1.0, n))


def plot_lines(ax, df, x, variants, y="latency_ms", errors=True):
    for v, color, marker in zip(variants, blues(len(variants)), MARKERS):
        d = df[df.variant == v].sort_values(x)
        yerr = [d[y] - d.latency_p20, d.latency_p80 - d[y]] if errors else None
        ax.errorbar(d[x], d[y], yerr=yerr, color=color, marker=marker,
                    ms=4, capsize=2, label=LABELS[v])
    ticks = sorted(df[x].unique())
    ax.set_xscale("log", base=2)
    ax.set_yscale("log")
    ax.set_xticks(ticks, [str(t) for t in ticks])
    ax.grid(True, which="major", color=LIGHT, lw=0.5)
    ax.spines[["top", "right"]].set_visible(False)


def legend_right(ax):
    ax.legend(frameon=False, fontsize=7, loc="upper left", bbox_to_anchor=(1.01, 1))


def save(fig, name):
    fig.tight_layout()
    fig.savefig(OUT / f"{name}.pdf", bbox_inches="tight")
    plt.close(fig)


def fig1_strided_view(n=8, bwd=1, fwd=1):
    w = bwd + fwd + 1
    grid = np.zeros((n, n + w - 1))
    for i in range(n):
        grid[i, i:i + w] = 1
    fig, ax = plt.subplots(figsize=(4.5, 3.2))
    ax.imshow(grid, cmap=ListedColormap(["white", "#4292c6"]))
    ax.set_xticks(range(n + w - 1), ["pad"] * bwd + [f"$k_{j}$" for j in range(n)] + ["pad"] * fwd)
    ax.set_yticks(range(n), [f"$q_{i}$" for i in range(n)])
    ax.set_xticks(np.arange(-0.5, n + w - 1), minor=True)
    ax.set_yticks(np.arange(-0.5, n), minor=True)
    ax.grid(which="minor", color=LIGHT, lw=0.8)
    ax.tick_params(which="both", length=0)
    ax.set_xlabel("Padded key buffer")
    ax.set_ylabel("Query")
    save(fig, "fig1_strided_view")


def fig2_tiled_kernel(n=64, tile=8, bwd=6, fwd=6):
    i, j = np.indices((n, n))
    in_window = (j - i >= -bwd) & (j - i <= fwd)
    tiles = in_window.reshape(n // tile, tile, n // tile, tile).any(axis=(1, 3))
    visited = np.kron(tiles, np.ones((tile, tile), dtype=bool))
    grid = visited.astype(int) + in_window
    fig, ax = plt.subplots(figsize=(3.6, 3.6))
    ax.imshow(grid, cmap=ListedColormap(["white", LIGHT, "#2171b5"]))
    ticks = np.arange(-0.5, n, tile)
    ax.set_xticks(ticks, [])
    ax.set_yticks(ticks, [])
    ax.grid(color=DARK, lw=0.6)
    ax.tick_params(length=0)
    ax.set_xlabel("Key position")
    ax.set_ylabel("Query position")
    save(fig, "fig2_tiled_kernel")


def fig3_latency_vs_n(df, w=256):
    d = df[(df.W == w) & (df.dtype == "float16")]
    variants = ["naive_pt", "strided_pt", "triton_strided", "triton_tiled", "flex"]
    fig, ax = plt.subplots(figsize=(5.5, 3.4))
    plot_lines(ax, d, "N", variants)
    n = np.array(sorted(d.N.unique()), dtype=float)
    for v, power, style in [("triton_tiled", 1, ":"), ("naive_pt", 2, "--")]:
        y0 = d[d.variant == v].sort_values("N").latency_ms.iloc[0]
        ax.plot(n, y0 * (n / n[0]) ** power, style, color="#6baed6", lw=1,
                label=f"$O(N^{power})$" if power > 1 else "$O(N)$")
    ax.set_xlabel("Sequence length $N$")
    ax.set_ylabel("Latency (ms)")
    legend_right(ax)
    save(fig, "fig3_latency_vs_n")


def fig4_latency_vs_w(df, n=8192):
    d = df[df.N == n]
    panels = [("float32", "FP32", "flex_tf32"), ("float16", "FP16", "flex")]
    fig, axes = plt.subplots(1, 2, figsize=(8, 3.4))
    for ax, (dtype, title, flex) in zip(axes, panels):
        variants = ["naive_pt", "strided_pt", "triton_strided", "triton_tiled", flex]
        plot_lines(ax, d[d.dtype == dtype], "W", variants)
        ax.set_title(title)
        ax.set_xlabel("Window width $W$")
    axes[0].set_ylabel("Latency (ms)")
    fig.legend(*axes[1].get_legend_handles_labels(), loc="lower center",
               ncol=5, frameon=False, fontsize=8, bbox_to_anchor=(0.5, -0.08))
    save(fig, "fig4_latency_vs_w")


def fig5_memory_vs_n(df, w=512):
    d = df[(df.W == w) & (df.dtype == "float32")]
    variants = ["naive_pt", "naive_pt_compiled", "strided_pt",
                "strided_pt_compiled", "triton_tiled", "flex"]
    fig, ax = plt.subplots(figsize=(5.5, 3.4))
    plot_lines(ax, d, "N", variants, y="peak_mib", errors=False)
    ax.set_xlabel("Sequence length $N$")
    ax.set_ylabel("Peak memory (MiB)")
    legend_right(ax)
    save(fig, "fig5_memory_vs_n")


def main():
    OUT.mkdir(exist_ok=True)
    df = load(CSV)
    fig1_strided_view()
    fig2_tiled_kernel()
    fig3_latency_vs_n(df)
    fig4_latency_vs_w(df)
    fig5_memory_vs_n(df)
    print(f"saved figures to {OUT}/")


if __name__ == "__main__":
    main()
