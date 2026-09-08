"""Plotly-only charts and PIL overlay images (no matplotlib / seaborn plots)."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from PIL import Image

# Distinct method colours used on every chart.
METHOD_COLORS = {
    "frangi": "#1f77b4",
    "rf": "#ff7f0e",
    "svm": "#2ca02c",
    "unet": "#d62728",
}

METRIC_LABELS = {
    "dice": "Dice",
    "iou": "IoU",
    "sensitivity": "Sensitivity",
    "specificity": "Specificity",
    "precision": "Precision",
}


def overlay_rgb(
    image: np.ndarray,
    pred: np.ndarray,
    truth: np.ndarray,
    fov: np.ndarray | None = None,
) -> np.ndarray:
    """TP green, FP red, FN blue on the fundus photograph."""
    base = np.asarray(image, dtype=np.float32)
    if base.ndim == 2:
        base = np.repeat(base[..., None], 3, axis=2)
    if base.max() <= 1.0:
        base = base * 255.0
    out = base.copy()
    pred = np.asarray(pred, dtype=bool)
    truth = np.asarray(truth, dtype=bool)
    if fov is not None:
        pred = pred & fov
        truth = truth & fov
    tp = pred & truth
    fp = pred & ~truth
    fn = ~pred & truth
    alpha = 0.55
    out[tp] = (1 - alpha) * out[tp] + alpha * np.array([40.0, 220.0, 80.0])
    out[fp] = (1 - alpha) * out[fp] + alpha * np.array([230.0, 50.0, 50.0])
    out[fn] = (1 - alpha) * out[fn] + alpha * np.array([50.0, 90.0, 230.0])
    return np.clip(out, 0, 255).astype(np.uint8)


def save_png(path: Path, array: np.ndarray) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(array).save(path)
    return path


def _write_plotly(fig: go.Figure, html_path: Path, png_path: Path | None = None) -> list[Path]:
    html_path.parent.mkdir(parents=True, exist_ok=True)
    fig.write_html(html_path, include_plotlyjs="cdn")
    written = [html_path]
    if png_path is not None:
        png_path.parent.mkdir(parents=True, exist_ok=True)
        try:
            fig.write_image(png_path, scale=2)
            written.append(png_path)
        except Exception as exc:  # kaleido / chrome may be missing in headless CI
            print(f"[viz] PNG export skipped for {png_path.name}: {exc}")
    return written


def metrics_bar_chart(summary: pd.DataFrame, html_path: Path, png_path: Path) -> list[Path]:
    """Grouped bars: one cluster per metric, one colour per method."""
    metric_cols = [c for c in METRIC_LABELS if c in summary.columns]
    long = summary.melt(
        id_vars=["method"],
        value_vars=metric_cols,
        var_name="metric",
        value_name="value",
    )
    long["metric"] = long["metric"].map(METRIC_LABELS)
    fig = px.bar(
        long,
        x="metric",
        y="value",
        color="method",
        barmode="group",
        range_y=[0, 1],
        color_discrete_map=METHOD_COLORS,
        title="Segmentation metrics on the demo test split (FOV pixels only)",
        labels={"value": "Score", "metric": "", "method": "Method"},
    )
    fig.update_layout(template="plotly_white", legend_title_text="Method")
    return _write_plotly(fig, html_path, png_path)


def morphology_chart(morpho: pd.DataFrame, html_path: Path, png_path: Path) -> list[Path]:
    keys = [
        "vessel_density",
        "length_px",
        "mean_width_px",
        "n_branch_points",
        "mean_tortuosity",
    ]
    fig = make_subplots(
        rows=1,
        cols=len(keys),
        subplot_titles=["Density", "Length (px)", "Width (px)", "Branch pts", "Tortuosity"],
    )
    methods = list(morpho["method"].unique())
    for col, key in enumerate(keys, start=1):
        for method in methods:
            sub = morpho[morpho["method"] == method]
            fig.add_trace(
                go.Bar(
                    name=method,
                    x=[method],
                    y=[float(sub[key].mean())],
                    marker_color=METHOD_COLORS.get(method, "#444"),
                    showlegend=(col == 1),
                ),
                row=1,
                col=col,
            )
    fig.update_layout(
        template="plotly_white",
        title="Morphology derived from each predicted mask (test-set mean)",
        barmode="group",
    )
    return _write_plotly(fig, html_path, png_path)


def overlay_figure(
    image: np.ndarray,
    truth: np.ndarray,
    predictions: dict[str, np.ndarray],
    fov: np.ndarray,
    html_path: Path,
) -> Path:
    names = ["fundus", "ground truth"] + list(predictions.keys())
    n = len(names)
    fig = make_subplots(rows=1, cols=n, subplot_titles=names)
    gt_overlay = overlay_rgb(image, truth, truth, fov)
    fig.add_trace(go.Image(z=image), row=1, col=1)
    fig.add_trace(go.Image(z=gt_overlay), row=1, col=2)
    for i, (name, pred) in enumerate(predictions.items(), start=3):
        fig.add_trace(go.Image(z=overlay_rgb(image, pred, truth, fov)), row=1, col=i)
    fig.update_layout(
        template="plotly_white",
        title="Overlays — green TP, red FP, blue FN",
        height=380,
    )
    fig.update_xaxes(visible=False)
    fig.update_yaxes(visible=False)
    _write_plotly(fig, html_path, None)
    return html_path
