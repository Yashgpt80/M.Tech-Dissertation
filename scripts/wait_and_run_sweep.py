"""Wait for the in-flight B1 training (started outside this script) to write
its `final_test` marker into metrics.jsonl, then hand off to run_s5_sweep.py
which orchestrates xai_eval(B1) -> train+eval(B2) -> train+eval(B3)."""
import json, time, subprocess, sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PY = ROOT / "fer" / "Scripts" / "python.exe"
B1_LOG = ROOT / "runs" / "S5_swin_tiny_xai_b1" / "metrics.jsonl"


def b1_done() -> bool:
    if not B1_LOG.exists():
        return False
    for line in B1_LOG.read_text().splitlines():
        if not line.strip():
            continue
        try:
            row = json.loads(line)
        except json.JSONDecodeError:
            continue
        if "final_test" in row:
            return True
    return False


print("Waiting for B1 training to finish...", flush=True)
while not b1_done():
    time.sleep(60)
print("B1 finished. Launching full S5 orchestrator.", flush=True)
sys.exit(subprocess.call([str(PY), "scripts/run_s5_sweep.py"], cwd=str(ROOT)))
