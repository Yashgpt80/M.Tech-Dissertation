"""Run a set of YAML configs sequentially.

Filters configs by glob pattern. Skips configs whose `runs/<run_name>/best.pt`
already exists (resumable). Logs per-run wall-clock to `runs/_sweep.log`.

Usage:
    python scripts/run_sweep.py --glob "configs/abl/S1_*.yaml"
    python scripts/run_sweep.py --glob "configs/abl/S4_*_baseline.yaml"
    python scripts/run_sweep.py --glob "configs/abl/*.yaml" --skip-existing
"""
from __future__ import annotations

import argparse
import datetime as dt
import subprocess
import sys
import time
from pathlib import Path

import yaml


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--glob", required=True, help="glob pattern for config yamls")
    ap.add_argument("--skip-existing", action="store_true", default=True)
    ap.add_argument("--no-skip", dest="skip_existing", action="store_false")
    ap.add_argument("--script", default=None,
                    help="Override training entrypoint. Default: auto-detect "
                    "from config (uses train_xai_guided if `masks` key present, else train_baseline).")
    args = ap.parse_args()

    cfgs = sorted(Path().glob(args.glob))
    if not cfgs:
        print(f"No configs matched: {args.glob}")
        sys.exit(1)

    log_path = Path("runs/_sweep.log")
    log_path.parent.mkdir(parents=True, exist_ok=True)

    print(f"Found {len(cfgs)} configs:")
    for p in cfgs:
        print(f"  {p}")

    for i, p in enumerate(cfgs, 1):
        cfg = yaml.safe_load(p.read_text(encoding="utf-8"))
        run_dir = Path(cfg["out_dir"]) / cfg["run_name"]
        ckpt = run_dir / "best.pt"
        if args.skip_existing and ckpt.exists():
            print(f"[{i}/{len(cfgs)}] SKIP (exists): {cfg['run_name']}")
            continue

        if args.script:
            entry = args.script
        else:
            entry = "src.train_xai_guided" if "masks" in cfg else "src.train_baseline"

        print(f"[{i}/{len(cfgs)}] {dt.datetime.now():%H:%M:%S} -> {cfg['run_name']} via {entry}")
        t0 = time.time()
        proc = subprocess.run([sys.executable, "-m", entry, "--config", str(p)])
        elapsed = time.time() - t0
        status = "OK" if proc.returncode == 0 else f"FAIL ({proc.returncode})"
        with open(log_path, "a", encoding="utf-8") as f:
            f.write(f"{dt.datetime.now().isoformat()}\t{cfg['run_name']}\t{entry}\t{status}\t{elapsed:.1f}s\n")
        print(f"           done in {elapsed/60:.1f} min - {status}")


if __name__ == "__main__":
    main()
