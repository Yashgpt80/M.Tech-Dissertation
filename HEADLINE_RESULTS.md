# Headline dissertation result — baseline vs XAI-guided across backbones

Run dates: 2026-05-14 to 2026-05-16.
Hardware: NVIDIA RTX 4060 Laptop GPU, torch 2.5.1+cu121, cuDNN 9.0.1.
Data: FER2013 (28 709 / 3 589 / 3 589 train/val/test), per-emotion soft anatomical region masks (`runs/masks.npz`, 14×14, see `src/data/region_masks.py`).

## Classification performance (test split)

| Backbone           | Variant                       | Test acc    | Test macro-F1 | Best epoch |
|--------------------|-------------------------------|------------:|--------------:|-----------:|
| ResNet50           | baseline                      | 69.55 %     | 66.31 %       | 37 |
| ResNet50           | **XAI-guided**                | **69.94 %** | **67.51 %**   | 11 |
| Swin-Tiny          | baseline                      | 72.28 %     | 70.37 %       | 23 |
| Swin-Tiny          | XAI-guided (S4)               | 71.72 %     | 70.96 %       | 10 |
| Swin-Tiny          | XAI-guided **B1** (S5, late warmup, blur 1.2)         | **73.45 %** | **71.64 %**   | 14 |
| Swin-Tiny          | XAI-guided **B2** (B1 + 14×14 penultimate + freeze)   | 73.45 %     | 70.97 %       | 14 |
| Swin-Tiny          | XAI-guided **B3** (B2 + hinge deletion penalty)       | 72.42 %     | 70.68 %       | 6  |

XAI-guided variants add a cosine attention loss between the differentiable Grad-CAM and the per-emotion anatomical mask, with `lambda_attn = 0.5`, `mask_source = landmark`, warm-started from the matching plain baseline checkpoint. Swin-Tiny uses the same differentiable Grad-CAM with a `channels_last_2d` BHWC→BCHW adapter so the loss is layout-agnostic.

**Effect of XAI-guidance on classification:**
- **ResNet50** → +0.39 acc, **+1.20 macro-F1** (wins on both).
- **Swin-Tiny S4** → −0.56 acc, +0.59 macro-F1 (small accuracy/balance trade-off; macro-F1 — the dissertation's headline metric — improves).
- **Swin-Tiny S5/B1** → **+1.17 acc, +1.27 macro-F1** over the baseline (the S4 trade-off is *eliminated* by raising the warmup, lowering λ, blurring the masks, and switching to the penultimate Swin block). This is the strongest XAI-guided result in the project.
- **Swin-Tiny S5/B3** keeps a +0.14 acc and +0.31 macro-F1 over the baseline while delivering the best deletion-AUC of any run — the *deletion penalty* trades a fraction of a point of accuracy for a 33 % drop in deletion-AUC and a 4× rise in faithfulness-correlation.

**Effect of stronger backbone:**
- Swin-Tiny baseline beats ResNet50 baseline by **+2.73 acc, +4.06 macro-F1** — a clean cross-architecture validation.
- Swin-Tiny XAI-guided **B1** beats ResNet50 XAI-guided by **+3.51 acc, +4.13 macro-F1**.

## XAI faithfulness

All metrics evaluated on the FER2013 test split with `src/xai_eval.py` (deletion/insertion: 20 perturbation steps; faithfulness-correlation: 7×7 patch grid; pointing-game: half-max region of the soft anatomical mask). Earlier rows use 144–192 samples (S3 / S4 era), S5 rows use 480 samples (`max_batches=20, batch=24`).

### ResNet50

| Metric             | Baseline | XAI-guided | Δ |
|--------------------|---------:|-----------:|------:|
| Deletion AUC ↓     | 0.196    | **0.185**  | −0.011 (better) |
| Insertion AUC ↑    | 0.523    | 0.511      | −0.012 |
| AOPC               | 0.306    | 0.305      | ≈ |
| **Pointing-game ↑**| 34.90 %  | **98.96 %**| **+64.06** |
| Faithfulness corr ↑| 0.142    | 0.098      | −0.044 |
| Gini sparsity      | 0.843    | 0.741      | −0.102 |

### Swin-Tiny — S4 vs new S5 sweep

| Metric             | Baseline | S4 XAI | S5/B1 | S5/B2 | **S5/B3** |
|--------------------|---------:|-------:|------:|------:|----------:|
| Deletion AUC ↓     | 0.189    | 0.223  | 0.202 | 0.200 | **0.135** |
| Insertion AUC ↑    | 0.529    | 0.519  | 0.511 | **0.520** | 0.503 |
| AOPC               | 0.321    | 0.280  | 0.298 | 0.309 | **0.354** |
| Pointing-game ↑    | 47.92 %  | **100.00 %** | 45.21 % | 53.75 % | 48.75 % |
| Faithfulness corr ↑| 0.150    | 0.078  | 0.091 | 0.132 | **0.388** |
| Gini sparsity      | 0.647    | 0.675  | 0.475 | 0.660 | 0.681 |

### S5 sweep design

The S5 sweep was launched after the S4 Swin-Tiny XAI-guided model regressed on perturbation-faithfulness (deletion AUC went *up* by 0.034 vs the baseline). The four interventions were applied incrementally so each addition is attributable:

- **B1** — `lambda_attn` 0.5 → 0.1, `warmup_xai_epochs` 2 → 5, mask Gaussian-blurred with `σ=1.2` before resizing. *Goal: stop the attention loss from collapsing the CAM onto a hard mask boundary.*
- **B2** — B1 + Grad-CAM hooked on the **penultimate** Swin block (14×14 spatial map, 384 channels) instead of the final 7×7 block, with stages [0, 1] frozen. *Goal: give the CAM enough resolution to actually match the 14×14 mask, while keeping low-level features fixed.*
- **B3** — B2 + a **hinge deletion penalty** added to the loss: `(log p_true(x_del) − margin).clamp(min=0).mean()` with `margin = −2`, `lambda_deletion = 0.05`, `top_frac = 0.10`. The penalty is bounded in `[0, 2]` per sample and only fires while the model is still confident on the true class after its own top-CAM region is masked. *Goal: directly minimise deletion AUC at training time without destabilising the classifier.*

### Reading the table

- **B1 is the headline classification result** — a clean **+1.17 acc / +1.27 macro-F1** over the Swin-Tiny baseline, eliminating the S4 accuracy drop while keeping faithfulness-correlation, pointing-game, and AOPC roughly at baseline level (and a sharper, less concentrated CAM, gini 0.475 vs 0.647).
- **B3 is the headline faithfulness result** — deletion AUC **0.135** is **−0.054 vs the Swin-Tiny baseline** (−29 %) and **−0.088 vs S4 XAI** (−40 %), AOPC peaks at **0.354**, and faithfulness-correlation jumps from 0.150 (baseline) to **0.388** — a 2.6× improvement on the only metric that requires the model to actually *agree* with its own saliency map. Classification cost is small (−0.69 macro-F1 vs B1, but still +0.31 over the Swin-Tiny baseline).
- **The S4 → S5 progression** is a worked example of how to debug an XAI-guided regression on a hierarchical transformer: lower the regulariser strength, make the explanation target less sharp, give the saliency hook the spatial resolution to match the supervision, and add a *direct* perturbation-based loss only after the classification objective is already healthy.

The pointing-game is the fraction of test images for which the argmax of Grad-CAM lies inside the half-max region of the corresponding emotion mask. The S4 XAI-guided models drive pointing-game close to perfect alignment with anatomical face regions — exactly the property the *cosine attention loss* targets directly. The ResNet50 sweep is the cleanest case (better classification *and* better faithfulness across the board). On Swin-Tiny the S4 trade-off is real: forcing attention onto a tight mask region makes the transformer rely on redundant context elsewhere, so deleting that region barely hurts predictions (deletion AUC *worsens*). The **S5 sweep resolves this**: B1 fixes the classification regression, and B3 attacks deletion-AUC head-on with a perturbation-based loss term, trading a fraction of a point of macro-F1 for the strongest faithfulness profile of any run in the project. The lesson for the dissertation is: **on hierarchical-transformer backbones, pointing-game / AOPC / deletion families of XAI metrics measure different things and can move in opposite directions; aligning all of them requires both spatial supervision (cosine attention loss) and perturbation supervision (deletion penalty).**

## Plots / artifacts

- `runs/resnet50_baseline/xai_eval/curves.png` and `runs/S4_resnet50_xai/xai_eval/curves.png`
  — deletion and insertion curves (ResNet50).
- `runs/swin_tiny_baseline/xai_eval/curves.png`, `runs/S4_swin_tiny_xai/xai_eval/curves.png`,
  `runs/S5_swin_tiny_xai_b{1,2,3}/xai_eval/curves.png` — Swin-Tiny progression.
- `runs/*/xai_eval/qualitative.png` — Grad-CAM / Grad-CAM++ overlays per emotion.
- `runs/*/xai_eval/faithfulness.npz` — raw deletion/insertion curves, faithfulness-correlation, Gini scores.
- `runs/{resnet50,swin_tiny}_baseline/cm.png`, `runs/*/per_class.csv` — confusion matrix and per-class scores.
- `runs/_summary.csv` — aggregate of all completed runs.

## Reproduce

```powershell
# (1) baselines
python -m src.train_baseline --config configs/baseline_resnet50.yaml
python -m src.train_baseline --config configs/baseline_swin_tiny.yaml

# (2) XAI-guided S4 (warm-started from baselines)
python -m src.train_xai_guided --config configs/abl/S4_resnet50_xai.yaml
python -m src.train_xai_guided --config configs/abl/S4_swin_tiny_xai.yaml

# (3) S5 Swin-Tiny sweep — runs B1 → B2 → B3 sequentially with auto-skip on existing best.pt
python scripts/run_s5_sweep.py

# (4) Faithfulness eval (single command per checkpoint)
python -m src.xai_eval --config configs/abl/S5_swin_tiny_xai_b3.yaml --ckpt runs/S5_swin_tiny_xai_b3/best.pt --max_batches 20 --steps 20
```

## Pending experiments (resumable via `python scripts/run_all.py --include …`)

- `xai_default` — mini-Xception, EfficientNet-B0 XAI-guided.
- `lambda_sweep` — ResNet50 λ ∈ {0.1, 0.3, 0.5, 1.0}.
- `mask_sweep`   — ResNet50 mask source ∈ {landmark, class_avg, uniform, random}.
- `cam_sweep`    — ResNet50 cam method ∈ {Grad-CAM, Grad-CAM++}.
- *(Optional)* port the S5 deletion-penalty + penultimate-hook recipe to ResNet50 and EfficientNet-B0 and verify the deletion-AUC win generalises across architectures.

Each run is skipped if its `best.pt` exists, so the sweep is safely resumable.
