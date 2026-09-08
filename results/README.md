# Results

`pixi run demo` (or the individual tasks) write:

- `figures/` — Plotly HTML charts, optional Kaleido PNGs, RGB overlays
- `metrics/` — CSV/JSON for pixel scores and morphology
- `predictions/` — binary masks per method (gitignored, regenerated)
- `models/` — `rf.joblib`, `svm.joblib`, `unet.pt` (gitignored)

Charts are produced only by Plotly (`src/drive_seg/viz.py`). Overlay PNGs are composed with Pillow from the fundus pixels (they are images, not matplotlib figures).
