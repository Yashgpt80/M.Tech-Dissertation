# XAI-Guided Facial Emotion Recognition on FER2013

M.Tech dissertation project (IIT Roorkee). Train strong FER2013 baselines, then improve them with **Grad-CAM-guided training** that pulls model saliency toward facial landmark regions (eyes, brows, mouth) depending on the emotion class.

## Quickstart (Windows + local CUDA GPU)

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install --upgrade pip
# Pick the CUDA wheel matching your driver (cu121 shown):
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
pip install -r requirements.txt
```

Verify CUDA:

```powershell
python -c "import torch; print(torch.__version__, torch.cuda.is_available(), torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU')"
```

## Data

Place `fer2013.csv` at the repo root (already present). The CSV has columns `emotion`, `pixels`, `Usage` where `Usage ∈ {Training, PublicTest, PrivateTest}`.

Emotions: `0=angry, 1=disgust, 2=fear, 3=happy, 4=sad, 5=surprise, 6=neutral`.

## Train a baseline

```powershell
python -m src.train_baseline --config configs/baseline_resnet50.yaml
python -m src.train_baseline --config configs/baseline_efficientnet_b0.yaml
```

## Precompute landmark masks (one-time)

```powershell
python -m src.data.landmark_masks --csv fer2013.csv --out runs/masks.npz
```

## Train XAI-guided model

```powershell
python -m src.train_xai_guided --config configs/xai_guided_resnet50.yaml
```

## Evaluate

```powershell
python -m src.evaluate --ckpt runs/<run_name>/best.pt --config configs/<...>.yaml
python -m src.xai_eval  --ckpt runs/<run_name>/best.pt --config configs/<...>.yaml
```

## End-to-end orchestration

`scripts/run_all.py` is fully resumable (skips runs whose `best.pt` exists):

```powershell
# 1) baselines (ResNet50 ~75 min, EffNet-B0 ~60 min, Swin-Tiny ~90 min)
python scripts/run_all.py --include baselines

# 2) XAI-guided default (lambda=0.5, landmark masks, Grad-CAM) for all 3 backbones
python scripts/run_all.py --include xai_default

# 3) Ablations on ResNet50 (lambda x mask x cam method)
python scripts/run_all.py --include lambda_sweep mask_sweep cam_sweep

# 4) Aggregate -> runs/_summary.csv + _summary.md
python scripts/aggregate_results.py
```

## XAI evaluation per checkpoint

```powershell
python -m src.evaluate     --config configs/baseline_resnet50.yaml --ckpt runs/resnet50_baseline/best.pt
python -m src.xai_eval     --config configs/abl/S4_resnet50_xai.yaml  --ckpt runs/S4_resnet50_xai/best.pt
python -m src.xai_visualize --config configs/abl/S4_resnet50_xai.yaml --ckpt runs/S4_resnet50_xai/best.pt
```

`xai_eval` computes faithfulness metrics (deletion AUC, insertion AUC, AOPC, pointing-game vs landmark masks).
`xai_visualize` renders a multi-XAI grid (Grad-CAM, Grad-CAM++, SHAP, LIME) per emotion.

## Notebooks

- `notebooks/01_eda.ipynb` — class distribution, per-class sample grid.
- `notebooks/02_results.ipynb` — reads `runs/_summary.csv`, plots ablation curves, generates LaTeX-ready tables.
- `notebooks/03_xai_viz.ipynb` — side-by-side baseline-vs-XAI Grad-CAM overlays.

## Notes / caveats

- This repo uses **canonical anatomical region masks** (per-emotion soft ellipses on a 48x48 face template) instead of MediaPipe landmarks: FER2013's 48x48 grayscale faces are too low-res for reliable landmark detection, and recent MediaPipe (>=0.10.30) dropped the legacy `mp.solutions` namespace anyway. The masks are produced by `python -m src.data.region_masks` (the original `landmark_masks.py` is a thin alias kept for back-compat).
- For training-time Grad-CAM regularization we implement a *differentiable* Grad-CAM (`src/xai/gradcam.py`) using `torch.autograd.grad(create_graph=True)` so a second backward pass through the CAM works. Standard `pytorch-grad-cam` returns numpy and breaks the graph; we only use it (via `grad-cam` package) at *evaluation* time.
- `torch>=2.5,<2.6` with `cu121` is the recommended combo. We hit a `CUDNN_BACKEND_TENSOR_DESCRIPTOR` mismatch with bleeding-edge `2.12+cu130`; if you see that error, set `YAAS_USE_CUDNN=0` to disable cuDNN at the cost of speed.

## Repo layout

See `plan` artifact at `~/.windsurf/plans/xai-guided-fer-dissertation-0b5be3.md`.
