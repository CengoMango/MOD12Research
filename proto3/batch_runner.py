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
# Use the same Python interpreter that's running this script
PYTHON_BIN = sys.executable
# Run the rollout as a module (so package imports work):
# e.g. python -m base_dir.proto3.run_rollout
MODULE = f"base_dir.{Path(__file__).parent.name}.run_rollout"
# Repo root (cwd for subprocesses) is two levels up from this file
REPO_ROOT = Path(__file__).resolve().parents[2]
LOG_DIR = Path("logs")  # one log per run
LOG_DIR.mkdir(exist_ok=True)


def _run(idx: int) -> int:
    """
    Launch a single rollout, stream stdout/err to its own log,
    and return the exit code.
    """
    ts = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    logf = LOG_DIR / f"rollout_{idx:03d}_{ts}.log"

    # Ensure subprocess sees the repo root on PYTHONPATH so `import base_dir` works
    env = os.environ.copy()
    # Prepend REPO_ROOT to PYTHONPATH (or set it) so module imports resolve
    prev = env.get("PYTHONPATH", "")
    env["PYTHONPATH"] = str(REPO_ROOT) + (os.pathsep + prev if prev else "")

    # Debug: write which module and cwd we're launching (helps troubleshooting)
    with logf.open("wb") as fh:
      fh.write(f"Launching: python -m {MODULE}\nCWD: {REPO_ROOT}\nPYTHONPATH: {env['PYTHONPATH']}\n".encode())
      fh.flush()

      proc = subprocess.Popen(
        [PYTHON_BIN, "-m", MODULE],
        stdout=fh,
        stderr=subprocess.STDOUT,
        cwd=str(REPO_ROOT),
        env=env,
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
            status = "OK" if code == 0 else "FAIL"
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
