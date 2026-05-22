"""End-to-end orchestrator: baselines -> XAI-guided -> ablations -> aggregate.

Each step is skipped if its `runs/<run_name>/best.pt` already exists, so this
is fully resumable.

Recommended usage on a single GPU (overnight):
    python scripts/run_all.py --include baselines xai_default
    python scripts/run_all.py --include lambda_sweep mask_sweep
    python scripts/run_all.py --include all     # everything

Stages:
    baselines     : 3 backbone baselines (mini, resnet50, effnet)
    xai_default   : 3 XAI-guided runs at default lambda=0.5 / mask=landmark
    lambda_sweep  : S1 ResNet50 lambda in {0.1,0.3,0.5,1.0}
    mask_sweep    : S2 ResNet50 mask source in {landmark,class_avg,uniform,random}
    cam_sweep     : S3 ResNet50 cam in {gradcam, gradcam_pp}
    aggregate     : final summary + per-run evaluation
"""
from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

STAGES = {
    "baselines": [
        ("src.train_baseline", "configs/baseline_resnet50.yaml"),
        ("src.train_baseline", "configs/baseline_efficientnet_b0.yaml"),
        ("src.train_baseline", "configs/baseline_swin_tiny.yaml"),
    ],
    "xai_default": [
        ("src.train_xai_guided", "configs/abl/S4_resnet50_xai.yaml"),
        ("src.train_xai_guided", "configs/abl/S4_efficientnet_b0_xai.yaml"),
        ("src.train_xai_guided", "configs/abl/S4_swin_tiny_xai.yaml"),
    ],
    "lambda_sweep": [
        ("src.train_xai_guided", f"configs/abl/S1_resnet50_lam{l}.yaml")
        for l in ["0.1", "0.3", "0.5", "1"]
    ],
    "mask_sweep": [
        ("src.train_xai_guided", f"configs/abl/S2_resnet50_mask-{m}.yaml")
        for m in ["landmark", "class_avg", "uniform", "random"]
    ],
    "cam_sweep": [
        ("src.train_xai_guided", f"configs/abl/S3_resnet50_cam-{c}.yaml")
        for c in ["gradcam", "gradcam_pp"]
    ],
}

ALL_ORDER = ["baselines", "xai_default", "lambda_sweep", "cam_sweep", "mask_sweep"]


def already_done(cfg_path: str) -> bool:
    import yaml
    cfg = yaml.safe_load(Path(cfg_path).read_text(encoding="utf-8"))
    ck = Path(cfg["out_dir"]) / cfg["run_name"] / "best.pt"
    return ck.exists()


def run_one(entry: str, cfg_path: str) -> bool:
    if already_done(cfg_path):
        print(f"[skip] {cfg_path}  (best.pt exists)")
        return True
    print(f"[run]  {cfg_path}  via {entry}")
    p = subprocess.run([sys.executable, "-m", entry, "--config", cfg_path])
    return p.returncode == 0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--include", nargs="+",
                    choices=list(STAGES.keys()) + ["all", "aggregate"], required=True)
    args = ap.parse_args()

    if "all" in args.include:
        stages = list(ALL_ORDER)
    else:
        stages = [s for s in args.include if s != "aggregate"]

    for stage in stages:
        if stage not in STAGES:
            continue
        print(f"\n===== STAGE: {stage} =====")
        for entry, cfg in STAGES[stage]:
            ok = run_one(entry, cfg)
            if not ok:
                print(f"FAIL: {cfg} (continuing)")

    if "aggregate" in args.include or "all" in args.include:
        print("\n===== STAGE: aggregate =====")
        subprocess.run([sys.executable, "scripts/aggregate_results.py"])


if __name__ == "__main__":
    main()
