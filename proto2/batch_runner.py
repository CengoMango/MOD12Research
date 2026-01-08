# batch_runner.py
"""
Fire off many `run_rollout.py` jobs in parallel.

Notes
-----
* Uses ThreadPoolExecutor because each worker just waits for an external
  process; Python threads are fine for that.
* If `run_rollout.py` itself is heavy CPU work and you want *one process
  per core*, switch to `concurrent.futures.ProcessPoolExecutor`.
"""

import datetime
import os
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

NUM_RUNS = 50
MAX_PARALLEL = os.cpu_count() or 7  # tune (or expose via CLI/env)
PYTHON_BIN = sys.executable
PROJECT_ROOT = Path(__file__).resolve().parents[2]   # overcooked_ai/
SCRIPT = str(PROJECT_ROOT / "base_dir" / "proto2" / "run_rollout.py")

LOG_DIR = Path("logs")  # one log per run
LOG_DIR.mkdir(exist_ok=True)


def _run(idx: int) -> int:
    """
    Launch a single rollout, stream stdout/err to its own log,
    and return the exit code.
    """
    ts = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    logf = LOG_DIR / f"rollout_{idx:03d}_{ts}.log"

    env = os.environ.copy()

    # ensure repo root is in PYTHONPATH

    env["PYTHONPATH"] = str(PROJECT_ROOT) + os.pathsep + env.get("PYTHONPATH", "")

    with logf.open("wb") as fh:
        proc = subprocess.Popen(
            [PYTHON_BIN, SCRIPT],
            stdout=fh,
            stderr=subprocess.STDOUT,
            cwd=PROJECT_ROOT,
            env=env,  # <-- use modified PYTHONPATH
        )
        return proc.wait()  # exit code


if __name__ == "__main__":
    print(f"Running {NUM_RUNS} roll-outs "
          f"({MAX_PARALLEL} concurrent)…\n")

    exit_codes = {}

    with ThreadPoolExecutor(max_workers=MAX_PARALLEL) as pool:
        futures = {pool.submit(_run, i): i for i in range(NUM_RUNS)}
        for fut in as_completed(futures):
            idx = futures[fut]
            code = fut.result()
            exit_codes[idx] = code
            status = "✅" if code == 0 else "❌"
            print(f"Run {idx + 1}/{NUM_RUNS} finished: {status} (exit {code})")

    # --- summary ---
    ok = sum(c == 0 for c in exit_codes.values())
    failed = NUM_RUNS - ok
    print("\nDone!")
    print(f"  Successful : {ok}")
    print(f"  Failed     : {failed}")
    if failed:
        bad = [i + 1 for i, c in exit_codes.items() if c != 0]
        print("  Failed runs:", bad)
