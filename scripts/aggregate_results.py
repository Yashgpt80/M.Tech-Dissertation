"""Aggregate metrics across all `runs/<run_name>/metrics.jsonl` files into a
single CSV/Markdown table for the dissertation.

Reads each run's last JSONL line that contains `final_test`, plus the best
epoch's val metrics. Writes:
    runs/_summary.csv
    runs/_summary.md

Usage:
    python scripts/aggregate_results.py
"""
from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import yaml


ROOT = Path("runs")


def parse_run(run_dir: Path) -> dict | None:
    cfg_path = run_dir / "config.yaml"
    metrics_path = run_dir / "metrics.jsonl"
    if not cfg_path.exists() or not metrics_path.exists():
        return None
    cfg = yaml.safe_load(cfg_path.read_text(encoding="utf-8"))
    final_test = None
    best_epoch = None
    best_val_acc = None
    best_val_f1 = None
    last_epoch = 0
    with open(metrics_path, "r", encoding="utf-8") as f:
        for line in f:
            try:
                rec = json.loads(line)
            except json.JSONDecodeError:
                continue
            if "final_test" in rec:
                final_test = rec["final_test"]
                best_epoch = rec.get("best_epoch")
            if "val_acc" in rec:
                if best_val_f1 is None or rec.get("val_macro_f1", 0) > best_val_f1:
                    best_val_f1 = rec["val_macro_f1"]
                    best_val_acc = rec["val_acc"]
                last_epoch = max(last_epoch, rec.get("epoch", 0))

    train_cfg = cfg.get("train", {})
    return {
        "run": cfg.get("run_name", run_dir.name),
        "model": cfg.get("model", {}).get("name"),
        "image_size": cfg.get("model", {}).get("image_size"),
        "lambda_attn": train_cfg.get("lambda_attn"),
        "mask_source": train_cfg.get("mask_source"),
        "cam_method": train_cfg.get("cam_method"),
        "epochs_run": last_epoch,
        "best_epoch": best_epoch,
        "best_val_acc": best_val_acc,
        "best_val_f1": best_val_f1,
        "test_acc": (final_test or {}).get("acc"),
        "test_macro_f1": (final_test or {}).get("macro_f1"),
    }


def main():
    rows = []
    for run_dir in sorted(p for p in ROOT.iterdir() if p.is_dir() and not p.name.startswith("_")):
        row = parse_run(run_dir)
        if row is not None:
            rows.append(row)

    if not rows:
        print("No completed runs found under runs/")
        return

    df = pd.DataFrame(rows).sort_values(by=["test_macro_f1", "test_acc"], ascending=False, na_position="last")
    csv_path = ROOT / "_summary.csv"
    md_path = ROOT / "_summary.md"
    df.to_csv(csv_path, index=False)
    with open(md_path, "w", encoding="utf-8") as f:
        f.write("# Run summary\n\n")
        try:
            f.write(df.to_markdown(index=False, floatfmt=".4f"))
        except ImportError:
            # `tabulate` is optional; fall back to fixed-width text
            f.write("```\n")
            f.write(df.to_string(index=False))
            f.write("\n```")
        f.write("\n")
    print(df.to_string(index=False))
    print(f"\nWrote {csv_path} and {md_path}")


if __name__ == "__main__":
    main()
