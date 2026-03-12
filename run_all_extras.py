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
#SEEDS = [42, 123, 456] # Using 3 seeds to speed up

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

# 1. New Baselines (SMA, LSTM) for Hopper, Walker2d, Ant
for env in ['Hopper-v4', 'Walker2d-v4', 'Ant-v4']:
    for seed in SEEDS:
        run_script("experiments/phase6/run_new_baselines.py", ["--seed", str(seed), "--env-id", env, "--results-dir", str(RESULTS_DIR)], f"new_baselines_{env}_seed{seed}.log")

# 2. Oracle Ablation on Hopper
for seed in SEEDS:
    run_script("experiments/phase6/run_ablation_oracle.py", ["--seed", str(seed), "--env-id", "Hopper-v4", "--results-dir", str(RESULTS_DIR)], f"ablation_oracle_Hopper-v4_seed{seed}.log")

# 3. Walker2d Diagnostics
for seed in SEEDS:
    run_script("experiments/phase6/run_walker_diagnostics.py", ["--seed", str(seed), "--results-dir", str(RESULTS_DIR)], f"walker_diagnostics_seed{seed}.log")

print("All extra experiments complete!")