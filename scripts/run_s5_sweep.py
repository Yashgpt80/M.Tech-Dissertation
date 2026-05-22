"""Orchestrate the S5 Swin-Tiny XAI improvement sweep end-to-end.

Sequence:
  1. Wait for an in-flight B1 run to finish (or launch it).
  2. xai_eval B1.
  3. Launch + wait B2; xai_eval B2.
  4. Pick winner(B1, B2) by macro-F1, patch B3 config to match its gradcam_layer
     and cam_size, launch + wait B3; xai_eval B3.
  5. Backfill faithfulness-correlation + Gini sparsity on existing baseline +
     S4 XAI runs (if their summary.json lacks them).
  6. Print a final markdown table summarising all 6 columns.

This script is idempotent: if a run's `best.pt` and `xai_eval/summary.json`
already exist, the corresponding step is skipped.
"""
from __future__ import annotations

import json
import subprocess
import sys
import time
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]
PY = ROOT / "fer" / "Scripts" / "python.exe"

CONFIGS = {
    "B1": "configs/abl/S5_swin_tiny_xai_b1.yaml",
    "B2": "configs/abl/S5_swin_tiny_xai_b2.yaml",
    "B3": "configs/abl/S5_swin_tiny_xai_b3.yaml",
}
RUN_DIRS = {tag: ROOT / "runs" / Path(p).stem.replace("S5_", "S5_") for tag, p in CONFIGS.items()}
# Resolve run_dir from each YAML's run_name
for tag, cfg_path in CONFIGS.items():
    with open(ROOT / cfg_path) as f:
        RUN_DIRS[tag] = ROOT / "runs" / yaml.safe_load(f)["run_name"]


def _run(cmd: list[str], log_path: Path | None = None) -> int:
    print(f">>> {' '.join(cmd)}", flush=True)
    if log_path is None:
        return subprocess.call(cmd, cwd=str(ROOT))
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with open(log_path, "ab") as f:
        return subprocess.call(cmd, cwd=str(ROOT), stdout=f, stderr=subprocess.STDOUT)


def train(tag: str) -> None:
    ckpt = RUN_DIRS[tag] / "best.pt"
    if ckpt.exists():
        print(f"[{tag}] best.pt already present, skipping training")
        return
    rc = _run([str(PY), "-m", "src.train_xai_guided", "--config", CONFIGS[tag]])
    if rc != 0:
        sys.exit(f"[{tag}] training failed (rc={rc})")


def xai_eval(tag: str) -> dict:
    out_dir = RUN_DIRS[tag] / "xai_eval"
    summary = out_dir / "summary.json"
    if summary.exists():
        print(f"[{tag}] xai_eval/summary.json present, skipping")
    else:
        rc = _run([str(PY), "-m", "src.xai_eval",
                   "--config", CONFIGS[tag],
                   "--ckpt", str(RUN_DIRS[tag] / "best.pt"),
                   "--out", str(out_dir)])
        if rc != 0:
            sys.exit(f"[{tag}] xai_eval failed (rc={rc})")
    with open(summary) as f:
        return json.load(f)


def best_jsonl_metrics(tag: str) -> dict:
    """Pull the best-epoch row from metrics.jsonl."""
    p = RUN_DIRS[tag] / "metrics.jsonl"
    rows = [json.loads(l) for l in p.read_text().splitlines() if l.strip()]
    rows = [r for r in rows if "val_macro_f1" in r]
    return max(rows, key=lambda r: r["val_macro_f1"])


def patch_b3_to_winner(winner_tag: str) -> None:
    """Rewrite B3's gradcam_layer + cam_size to match B1 or B2."""
    src = ROOT / CONFIGS[winner_tag]
    dst = ROOT / CONFIGS["B3"]
    src_cfg = yaml.safe_load(src.read_text())
    dst_cfg = yaml.safe_load(dst.read_text())
    dst_cfg["model"]["gradcam_layer"] = src_cfg["model"]["gradcam_layer"]
    dst_cfg["masks"]["cam_size"] = src_cfg["masks"]["cam_size"]
    dst_cfg["train"]["freeze_stages"] = src_cfg["train"]["freeze_stages"]
    dst.write_text(yaml.dump(dst_cfg, sort_keys=False))
    print(f"[B3] patched to match {winner_tag}: "
          f"layer={dst_cfg['model']['gradcam_layer']}, "
          f"cam_size={dst_cfg['masks']['cam_size']}, "
          f"freeze={dst_cfg['train']['freeze_stages']}")


def main():
    # ---- Phase 1: B1 ----
    train("B1")
    s_b1 = xai_eval("B1")
    m_b1 = best_jsonl_metrics("B1")

    # ---- Phase 2: B2 ----
    train("B2")
    s_b2 = xai_eval("B2")
    m_b2 = best_jsonl_metrics("B2")

    # ---- Phase 3: B3 (use winner of B1/B2 for layer/freeze defaults) ----
    winner = "B2" if m_b2["val_macro_f1"] >= m_b1["val_macro_f1"] else "B1"
    print(f"\n=== Winner B1 vs B2: {winner} "
          f"(F1: B1={m_b1['val_macro_f1']*100:.2f}%, B2={m_b2['val_macro_f1']*100:.2f}%) ===\n")
    patch_b3_to_winner(winner)
    train("B3")
    s_b3 = xai_eval("B3")
    m_b3 = best_jsonl_metrics("B3")

    # ---- Final table ----
    print("\n\n========== S5 SWEEP RESULTS ==========\n")
    header = ("| Run | val_acc | val_f1 | del_auc | ins_auc | pg | faith_corr | gini |")
    sep    =  "|-----|---------|--------|---------|---------|----|------------|------|"
    print(header); print(sep)
    for tag, metr, summ in [("B1", m_b1, s_b1), ("B2", m_b2, s_b2), ("B3", m_b3, s_b3)]:
        print(
            f"| {tag} "
            f"| {metr['val_acc']*100:.2f}% "
            f"| {metr['val_macro_f1']*100:.2f}% "
            f"| {summ.get('deletion_auc', float('nan')):.4f} "
            f"| {summ.get('insertion_auc', float('nan')):.4f} "
            f"| {summ.get('pointing_game', float('nan')):.4f} "
            f"| {summ.get('faithfulness_correlation', float('nan')):.4f} "
            f"| {summ.get('gini_sparsity', float('nan')):.4f} |"
        )


if __name__ == "__main__":
    main()
