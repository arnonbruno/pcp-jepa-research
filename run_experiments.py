#!/usr/bin/env python3
"""
Main multi-seed experiment runner for NeurIPS submission.
Runs all baselines and PANO across multiple seeds and environments.
"""

import sys
import os
import time
import subprocess
from pathlib import Path
from datetime import datetime

# Add to path
sys.path.insert(0, str(Path(__file__).parent / "experiments" / "phase6"))

from experiment_utils import (
    ExperimentRunner, save_checkpoint, load_checkpoint, 
    log_progress, check_nan, safe_clip, monitor_memory, format_duration
)

LOG_DIR = Path(__file__).parent / "logs"
RESULTS_DIR = Path(__file__).parent / "results" / "neurips"

SEEDS = [42, 123, 456, 789, 1024]
ENVS = ['Hopper-v4', 'Walker2d-v4']
N_EPISODES = 50

def run_experiment_script(script_path, args, log_name):
    start = time.time()
    result = subprocess.run(
        [sys.executable, script_path] + args,
        capture_output=True,
        text=True,
        cwd=Path(__file__).parent
    )
    
    LOG_DIR.mkdir(exist_ok=True)
    with open(LOG_DIR / log_name, 'w') as f:
        f.write(result.stdout)
        if result.stderr:
            f.write("\n--- STDERR ---\n")
            f.write(result.stderr)
            
    elapsed = time.time() - start
    
    if result.returncode != 0:
        log_progress("main", f"Failed {log_name} after {format_duration(elapsed)}", "ERROR")
        print(f"ERROR in {log_name}: {result.stderr[-2000:]}")
        return False
        
    print(f"✓ {log_name} complete ({format_duration(elapsed)})")
    return True

def run_all_seeds():
    """Run experiments across all seeds and environments"""
    print("\n" + "="*70)
    print("STARTING MULTI-SEED EXPERIMENTS")
    print(f"Seeds: {SEEDS}")
    print(f"Envs: {ENVS}")
    print("="*70)
    
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    
    success_all = True
    for env_id in ENVS:
        for seed in SEEDS:
            print(f"\n--- Environment: {env_id} | Seed: {seed} ---")
            
            # 1. SOTA Baselines
            log_name = f"sota_{env_id}_seed{seed}.log"
            print(f"Running SOTA Baselines...")
            args_sota = [
                "--n-episodes", str(N_EPISODES),
                "--seed", str(seed),
                "--env-id", env_id,
                "--results-dir", str(RESULTS_DIR),
                "--train-episodes", "50", # Reduced for speed, or keep 300? Let's use 100 for speed if needed, but 300 was default. 
                "--train-epochs", "50"
            ]
            sota_ok = run_experiment_script("experiments/phase6/sota_baselines.py", args_sota, log_name)
            
            # 2. PANO
            log_name = f"pano_{env_id}_seed{seed}.log"
            print(f"Running PANO...")
            args_pano = [
                "--n-episodes", str(N_EPISODES),
                "--seed", str(seed),
                "--env-id", env_id,
                "--results-dir", str(RESULTS_DIR),
                "--train-episodes", "50",
                "--train-epochs", "50"
            ]
            pano_ok = run_experiment_script("experiments/phase6/hopper_pano.py", args_pano, log_name)
            
            if not (sota_ok and pano_ok):
                success_all = False
                print(f"FAILED on Env: {env_id} | Seed: {seed}")
                # We can either continue or break. Let's continue to get as many results as possible.
                
    return success_all

def aggregate_results():
    """Run the aggregation script"""
    print("\n" + "="*70)
    print("AGGREGATING RESULTS")
    print("="*70)
    
    start = time.time()
    result = subprocess.run(
        [sys.executable, "aggregate_results.py"],
        capture_output=True,
        text=True,
        cwd=Path(__file__).parent
    )
    
    with open(LOG_DIR / "aggregation.log", 'w') as f:
        f.write(result.stdout)
        if result.stderr:
            f.write("\n--- STDERR ---\n")
            f.write(result.stderr)
            
    elapsed = time.time() - start
    
    if result.returncode != 0:
        log_progress("main", f"Aggregation FAILED", "ERROR")
        print(result.stderr)
        return False
        
    print(result.stdout)
    print(f"✓ Aggregation complete ({format_duration(elapsed)})")
    return True

def main():
    print("="*70)
    print("NEURIPS MULTI-SEED EXPERIMENT SUITE")
    print(f"Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("="*70)
    
    overall_start = time.time()
    
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    
    run_all_seeds()
    aggregate_results()
    
    total_elapsed = time.time() - overall_start
    print("\n" + "="*70)
    print("ALL EXPERIMENTS COMPLETE")
    print(f"Total time: {format_duration(total_elapsed)}")
    print(f"Finished: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("="*70)
    
    return 0

if __name__ == "__main__":
    sys.exit(main())
