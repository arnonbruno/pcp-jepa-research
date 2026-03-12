#!/usr/bin/env python3
import sys
import time
import subprocess
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent / "experiments" / "phase6"))
from experiment_utils import log_progress, format_duration

LOG_DIR = Path(__file__).parent / "logs"
RESULTS_DIR = Path(__file__).parent / "results" / "neurips"
SEEDS = [42, 123, 456, 789, 1024]
N_EPISODES = 50

def run_script(script_path, args, log_name):
    print(f"Running {log_name}...")
    start = time.time()
    result = subprocess.run([sys.executable, script_path] + args, capture_output=True, text=True, cwd=Path(__file__).parent)
    LOG_DIR.mkdir(exist_ok=True)
    with open(LOG_DIR / log_name, 'w') as f:
        f.write(result.stdout)
        if result.stderr:
            f.write("\n--- STDERR ---\n")
            f.write(result.stderr)
    elapsed = time.time() - start
    if result.returncode != 0:
        print(f"ERROR in {log_name}: {result.stderr[-1000:]}")
        return False
    print(f"✓ {log_name} complete ({format_duration(elapsed)})")
    return True

RESULTS_DIR.mkdir(parents=True, exist_ok=True)

# Ant-v4 (Both SOTA and PANO)
for seed in SEEDS:
    run_script("experiments/phase6/sota_baselines.py", ["--n-episodes", str(N_EPISODES), "--seed", str(seed), "--env-id", "Ant-v4", "--results-dir", str(RESULTS_DIR), "--train-episodes", "50", "--train-epochs", "50"], f"sota_Ant-v4_seed{seed}.log")
    run_script("experiments/phase6/hopper_pano.py", ["--n-episodes", str(N_EPISODES), "--seed", str(seed), "--env-id", "Ant-v4", "--results-dir", str(RESULTS_DIR), "--train-episodes", "50", "--train-epochs", "50"], f"pano_Ant-v4_seed{seed}.log")

# Walker2d SOTA (missing)
for seed in SEEDS:
    run_script("experiments/phase6/sota_baselines.py", ["--n-episodes", str(N_EPISODES), "--seed", str(seed), "--env-id", "Walker2d-v4", "--results-dir", str(RESULTS_DIR), "--train-episodes", "50", "--train-epochs", "50"], f"sota_Walker2d-v4_seed{seed}.log")

print("All done!")