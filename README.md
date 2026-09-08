# DRIVE retinal vessel segmentation

A teaching pipeline for **retinal vessel segmentation** on the [DRIVE](https://drive.grand-challenge.org/) benchmark: classical image processing, pixel-wise machine learning, then a small **PyTorch U-Net**. The stack is **Pixi-only** (no `requirements.txt`), **PyTorch with NVIDIA CUDA 12.x** as the primary device, **Napari** for interactive overlays, and **Plotly** for every metric chart.

CPU execution is supported when no GPU is visible. CUDA is still the documented primary path.

There is **no TensorFlow** in this project.

---

## Clinical problem

Colour fundus photographs are the workhorse image of diabetic-retinopathy screening. The vascular tree in those photographs is not just anatomy to outline: **calibre, density, branching complexity and tortuosity** shift with hypertension, diabetes, glaucoma and ageing. Manual tracing of every pixel is slow and noisy, so automated masks are the usual first step before those morphological biomarkers are computed.

**DRIVE** (Digital Retinal Images for Vessel Extraction; Staal *et al.*, *IEEE TMI* 2004) is the canonical 40-image benchmark for this task. Twenty training and twenty test photographs, each with a circular field of view (FOV) and a first-observer vessel tracing, remain the standard comparison point for Frangi-style filters, random forests, and U-Nets.

What the numbers mean in clinic:

| Metric | Reads as | Failure mode |
| --- | --- | --- |
| **Dice / IoU** | Overlap with the observer tracing | Can look fine while thin capillaries are missing |
| **Sensitivity** | Fraction of true vessel pixels recovered | Low → missed ischaemia / calibre change |
| **Specificity** | Fraction of true background left clean | Low → choroidal streaks counted as vessels |
| **Precision** | Fraction of predicted vessel that is real | Low → morphology (length, branches) explodes |

A mask is not the end product. After segmentation we skeletonise and measure **density, centreline length, mean width, branch-point count and tortuosity**. Those quantities are closer to what a report would mention than Dice is. They are also unforgiving: a handful of spurious pixels barely moves Dice and wrecks length and branching.

---

## Pipeline

```
fundus RGB
  │  green channel + CLAHE (FOV-aware)
  ├─► Frangi vesselness + Otsu          (classical)
  ├─► pixel RF / linear SVM             (hand-crafted features)
  └─► U-Net, Dice + BCE, flip/rot/jitter (deep learning, CUDA)
           │
           ▼
     Dice / IoU / sens / spec / prec
           │
           ▼
     skeleton → density, length, width, branching, tortuosity
           │
           ▼
     Plotly charts + RGB overlays + Napari layers
```

1. **Classical** — Frangi *et al.* (MICCAI 1998) multi-scale Hessian vesselness on the inverted CLAHE green channel, Otsu on the FOV, small-object cleanup.
2. **ML** — each FOV pixel is a sample. Features: intensity, local mean/std, Sobel, Frangi, Hessian ridge, distance from centre. Random Forest and a **linear SVM** (a kernel SVM on every pixel is too slow to teach with).
3. **U-Net** — three-level encoder/decoder, Dice + weighted BCE inside the FOV, geometric and photometric augmentation. Trains on CUDA when `torch.cuda.is_available()`, otherwise CPU.

---

## Stack (hard constraints)

| Constraint | How it is met |
| --- | --- |
| Pixi only, `linux-64` + `win-64` | `pixi.toml` — no `requirements.txt` |
| PyTorch + NVIDIA CUDA 12.x | `pytorch-gpu` + `cuda-version = 12.*` on CUDA platforms |
| CPU fallback | named `linux-64-cpu` / `win-64-cpu` platforms |
| Napari | `napari` + `pyqt`; `pixi run napari` |
| Plotly only for charts | `drive_seg.viz` — no matplotlib/seaborn plotting |
| No TensorFlow | not a dependency; `pixi run check-stack` verifies |
| Do not redistribute DRIVE | download script + Grand Challenge link; synthetic demo data |

### Why not `pytorch-cuda` from the `nvidia` / `pytorch` channels?

The historical conda recipe was:

```text
pytorch + pytorch-cuda=12.4  -c pytorch -c nvidia
```

Pixi now treats that channel mix as **legacy**. Mixing the `pytorch`/`nvidia` channels with conda-forge **Napari / Qt / vispy** is a reliable way to get an unsolvable or silently broken environment. NVIDIA’s CUDA 12 libraries are published on **conda-forge**; the Pixi-recommended equivalent (and what this repo uses) is:

```toml
[target."*-cuda".dependencies]
pytorch-gpu = "*"
cuda-version = "12.*"
```

with CUDA-capable platforms declared first so they win when an NVIDIA driver is present:

```toml
platforms = [
  { name = "linux-64-cuda", platform = "linux-64", cuda = "12.0" },
  { name = "win-64-cuda", platform = "win-64", cuda = "12.0" },
  { name = "linux-64-cpu", platform = "linux-64" },
  { name = "win-64-cpu", platform = "win-64" },
]
```

That is CUDA 12.x via NVIDIA packages, expressed the way Pixi expects.

---

## Install

Install [Pixi](https://pixi.sh/) (Linux or Windows), clone this repo, then:

```bash
pixi install
pixi run check-cuda
pixi run check-stack
```

`check-cuda` prints `torch`’s version and `torch.cuda.is_available()`. On a machine with NVIDIA drivers this should be `True` and the CUDA 12.x build is used. Without a GPU, Pixi selects the CPU platform and the same tasks still run.

### CUDA (primary)

- NVIDIA GPU + recent driver (CUDA 12 capable).
- Linux x86_64 or Windows x86_64.
- After `pixi install`, `pixi info` should list a `__cuda=12…` virtual package and the selected platform name should end in `-cuda`.

Force a platform when you need to inspect the other solve:

```bash
pixi run --platform linux-64-cuda check-cuda
pixi run --platform linux-64-cpu check-cuda
```

(On Windows use `win-64-cuda` / `win-64-cpu`.)

### CPU fallback

If Pixi reports that the CUDA environment cannot be installed (`system does not match the requirements` / no `__cuda` virtual package), you are on the CPU solve. That is expected on laptops without NVIDIA hardware and in many CI images. Training the demo U-Net is slower but finishes.

### Napari

`pixi run napari` opens a viewer with fundus, FOV, ground truth and one labels layer per method. It needs a desktop session (Windows, or Linux with `DISPLAY` / Wayland). Headless machines write `results/figures/napari_overlay_*.png` instead — the demo does this automatically.

### Windows and Linux

The lockfile is meant to cover **both** `win-64` and `linux-64`. Use a developer prompt / PowerShell on Windows; Git Bash works if `pixi` is on `PATH`. Qt/Napari windows are native on both.

---

## Quickstart (demo, no DRIVE download)

```bash
pixi run demo
```

This will:

1. Write a synthetic DRIVE-layout dataset under `data/synthetic/` (if missing).
2. Run Frangi, Random Forest, linear SVM, and the U-Net.
3. Score Dice / IoU / sensitivity / specificity / precision.
4. Compute morphology.
5. Write Plotly HTML (and PNG when Kaleido can export) plus RGB overlays under `results/figures/`.

Then open `results/figures/metrics_comparison.html` in a browser, and on a desktop:

```bash
pixi run napari
```

### Official DRIVE

DRIVE is **not** in git. Register and download from Grand Challenge:

- Dataset: https://drive.grand-challenge.org/
- Download (account required): https://drive.grand-challenge.org/Download/

```bash
pixi run download-drive -- --archive /path/to/DRIVE.zip
```

The loaders **prefer** `data/drive/` when that tree exists. Details: [`data/README.md`](data/README.md).

---

## `pixi run` tasks

| Task | What it does |
| --- | --- |
| `pixi run check-cuda` | Print `torch` version and CUDA availability |
| `pixi run check-stack` | Import torch, napari, plotly; refuse TensorFlow as a project dep |
| `pixi run generate-data` | Procedural DRIVE-layout images |
| `pixi run download-drive` | Instructions + optional zip extract |
| `pixi run classical` | Frangi + Otsu |
| `pixi run ml` | Random Forest + linear SVM |
| `pixi run unet` | Train U-Net (Dice+BCE, augmentation) on CUDA or CPU |
| `pixi run evaluate` | Metrics tables + Plotly charts + overlays |
| `pixi run morphology` | Density, length, width, branching, tortuosity |
| `pixi run napari` | Interactive image + segmentation viewer |
| `pixi run demo` | All of the above on synthetic data |
| `pixi run lab` | JupyterLab on `notebooks/` |

---

## Notebooks

| Notebook | Contents |
| --- | --- |
| [`notebooks/01_data_and_classical.ipynb`](notebooks/01_data_and_classical.ipynb) | DRIVE / synthetic I/O, green+CLAHE, Frangi, metrics |
| [`notebooks/02_ml_baselines.ipynb`](notebooks/02_ml_baselines.ipynb) | Pixel features, RF vs SVM, class imbalance |
| [`notebooks/03_unet.ipynb`](notebooks/03_unet.ipynb) | Architecture, Dice+BCE, augmentation, CUDA device |
| [`notebooks/04_morphology_and_interpretation.ipynb`](notebooks/04_morphology_and_interpretation.ipynb) | Skeleton metrics, clinical reading, Napari cell |

Launch with `pixi run lab` so the kernel sees the Pixi environment.

---

## Interpreting a demo run

On **synthetic** data the ranking is a teaching device, not a DRIVE leaderboard. A typical `pixi run demo` looks like:

- **Frangi** recovers the wide trunks (precision ≈ 1) and drops capillaries (sensitivity ≈ 0.47). Overlays show blue false negatives along thin branches.
- **RF / linear SVM** nearly saturate Dice on this procedural set because the engineered features include Frangi plus clean intensity cues. That is the 2010s DRIVE story, not a claim about clinical photographs.
- **U-Net** (12 short CPU epochs, base width 16) is high-sensitivity / lower-precision: it paints extra pixels around vessels. More epochs and a GPU on real DRIVE reverse that gap.

Read morphology next to Dice. If U-Net Dice is high but branch-point count or length diverges from ground truth, you are looking at noisy capillaries or FOV-edge speckle — clinically the worse mask.

Overlays use a fixed legend: **green = true positive**, **red = false positive**, **blue = false negative**.

---

## Layout

```
pixi.toml                 # the only dependency file
src/drive_seg/            # library: data, classical, ml, unet, metrics, viz
scripts/                  # pixi tasks
notebooks/                # 01–04
data/                     # synthetic/ committed or generated; drive/ gitignored
results/figures/          # Plotly HTML/PNG + overlay PNGs
results/metrics/          # CSV/JSON tables
```

---

## Limitations (read these before quoting numbers)

- Synthetic images are a **computational stand-in**. They are not a licence to skip DRIVE.
- The U-Net is **small** (base width 16) so CPU demos finish. For a paper, widen it, train longer, and use the real 584×565 photographs with patch sampling.
- Metrics are **FOV-only**, matching the DRIVE convention of ignoring the black frame.
- Linear SVM ≠ kernel SVM. The comment in `src/drive_seg/ml.py` is deliberate.
- Napari needs a GUI; Plotly HTML does not.

## References

- J. Staal *et al.*, “Ridge-based vessel segmentation in color images of the retina,” *IEEE TMI*, 2004. Dataset: https://drive.grand-challenge.org/
- A. F. Frangi *et al.*, “Multiscale vessel enhancement filtering,” MICCAI 1998.
- O. Ronneberger *et al.*, “U-Net: Convolutional networks for biomedical image segmentation,” MICCAI 2015.
