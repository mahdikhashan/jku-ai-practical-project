import gradio as gr
import pandas as pd
import sys
import json
import os


def load_data_from_file(path: str):
    if not os.path.exists(path):
        raise FileNotFoundError(f"No such file: {path}")

    if path.endswith(".json"):
        with open(path, "r") as f:
            data = json.load(f)
        return pd.DataFrame(data)

    if path.endswith(".csv"):
        return pd.read_csv(path)

    raise ValueError("Unsupported file type. Use .json or .csv")


def get_leaderboard_data():
    if len(sys.argv) < 2:
        raise RuntimeError("Usage: python leaderboard.py <data_file.json|csv>")
    return load_data_from_file(sys.argv[1])


with gr.Blocks(title="Benchmark Leaderboard") as demo:
    df = get_leaderboard_data()
    gr.Markdown("## Benchmark Leaderboard")
    gr.Markdown("### Performance Scatter Plot")

    possible_x = [c for c in df.columns if "seq" in c.lower() or "context" in c.lower()]
    possible_y = [c for c in df.columns if "time" in c.lower()]

    x_col = possible_x[0] if possible_x else df.columns[0]
    y_col = possible_y[0] if possible_y else df.columns[1]

    gr.ScatterPlot(
        value=df,
        x=x_col,
        y=y_col,
        color="device" if "device" in df.columns else None,
        title=f"{y_col} vs {x_col}",
        tooltip=list(df.columns),
    )

    gr.DataFrame(value=df, headers=list(df.columns), interactive=False)


if __name__ == "__main__":
    demo.launch()
