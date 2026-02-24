#!/usr/bin/env python3
"""
Main experiment runner with full protection:
- Progress logging to file
- Checkpoint/resume capability  
- NaN detection and handling
- Memory monitoring
- Error recovery
"""

import sys
import os
import time
import json
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
RESULTS_DIR = Path(__file__).parent / "results" / "phase6"

def run_bouncing_ball():
    """Run Experiment 1: Bouncing Ball"""
    print("\n" + "="*70)
    print("EXPERIMENT 1: Bouncing Ball (F3-JEPA)")
    print("="*70)
    
    log_progress("main", "Starting Experiment 1: Bouncing Ball")
    start = time.time()
    
    # Run the experiment
    result = subprocess.run(
        [sys.executable, "experiments/phase5/f3_jepa.py"],
        capture_output=True,
        text=True,
        cwd=Path(__file__).parent
    )
    
    # Log output
    LOG_DIR.mkdir(exist_ok=True)
    with open(LOG_DIR / "exp1_bouncing_ball.log", 'w') as f:
        f.write(result.stdout)
        if result.stderr:
            f.write("\n--- STDERR ---\n")
            f.write(result.stderr)
    
    elapsed = time.time() - start
    
    if result.returncode != 0:
        log_progress("main", f"Experiment 1 FAILED after {format_duration(elapsed)}", "ERROR")
        print(f"ERROR: {result.stderr}")
        return False
    
    log_progress("main", f"Experiment 1 complete in {format_duration(elapsed)}")
    print(f"✓ Experiment 1 complete ({format_duration(elapsed)})")
    return True

def run_hopper_pano():
    """Run Experiment 2: Hopper PANO"""
    print("\n" + "="*70)
    print("EXPERIMENT 2: Hopper PANO vs Baselines")
    print("="*70)
    
    log_progress("main", "Starting Experiment 2: Hopper PANO")
    start = time.time()
    
    # Run the experiment
    result = subprocess.run(
        [sys.executable, "experiments/phase6/hopper_pano.py",
         "--n-episodes", "100",
         "--seed", "42",
         "--results-dir", str(RESULTS_DIR)],
        capture_output=True,
        text=True,
        cwd=Path(__file__).parent
    )
    
    # Log output
    LOG_DIR.mkdir(exist_ok=True)
    with open(LOG_DIR / "exp2_hopper_pano.log", 'w') as f:
        f.write(result.stdout)
        if result.stderr:
            f.write("\n--- STDERR ---\n")
            f.write(result.stderr)
    
    elapsed = time.time() - start
    
    if result.returncode != 0:
        log_progress("main", f"Experiment 2 FAILED after {format_duration(elapsed)}", "ERROR")
        print(f"ERROR: {result.stderr[-2000:]}")  # Last 2000 chars
        return False
    
    log_progress("main", f"Experiment 2 complete in {format_duration(elapsed)}")
    print(f"✓ Experiment 2 complete ({format_duration(elapsed)})")
    return True

def run_bulletproof():
    """Run Experiment 3: Bulletproof Negative Protocol"""
    print("\n" + "="*70)
    print("EXPERIMENT 3: Bulletproof Negative Protocol")
    print("="*70)
    
    log_progress("main", "Starting Experiment 3: Bulletproof Negative")
    start = time.time()
    
    # Run the experiment
    result = subprocess.run(
        [sys.executable, "experiments/phase6/bulletproof_negative.py",
         "--seed", "42",
         "--n-eval-episodes", "50",
         "--results-dir", str(RESULTS_DIR)],
        capture_output=True,
        text=True,
        cwd=Path(__file__).parent
    )
    
    # Log output
    LOG_DIR.mkdir(exist_ok=True)
    with open(LOG_DIR / "exp3_bulletproof.log", 'w') as f:
        f.write(result.stdout)
        if result.stderr:
            f.write("\n--- STDERR ---\n")
            f.write(result.stderr)
    
    elapsed = time.time() - start
    
    if result.returncode != 0:
        log_progress("main", f"Experiment 3 FAILED after {format_duration(elapsed)}", "ERROR")
        print(f"ERROR: {result.stderr[-2000:]}")
        return False
    
    log_progress("main", f"Experiment 3 complete in {format_duration(elapsed)}")
    print(f"✓ Experiment 3 complete ({format_duration(elapsed)})")
    return True

def run_figures():
    """Generate publication figures"""
    print("\n" + "="*70)
    print("GENERATING PUBLICATION FIGURES")
    print("="*70)
    
    log_progress("main", "Generating figures")
    start = time.time()
    
    result = subprocess.run(
        [sys.executable, "experiments/phase6/neurips_figures.py",
         "--results-dir", str(RESULTS_DIR),
         "--output-dir", str(RESULTS_DIR)],
        capture_output=True,
        text=True,
        cwd=Path(__file__).parent
    )
    
    # Log output
    with open(LOG_DIR / "figures.log", 'w') as f:
        f.write(result.stdout)
        if result.stderr:
            f.write("\n--- STDERR ---\n")
            f.write(result.stderr)
    
    elapsed = time.time() - start
    
    if result.returncode != 0:
        log_progress("main", f"Figures generation FAILED", "ERROR")
        return False
    
    log_progress("main", f"Figures generated in {format_duration(elapsed)}")
    print(f"✓ Figures generated ({format_duration(elapsed)})")
    return True

def main():
    """Run all experiments with checkpoint/resume support."""
    
    print("="*70)
    print("NEURIPS EXPERIMENT SUITE")
    print(f"Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("="*70)
    
    overall_start = time.time()
    
    # Create directories
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    
    # Check for resume
    checkpoint = load_checkpoint("experiment_suite")
    start_stage = 0
    
    if checkpoint and checkpoint.get("stage"):
        stage = checkpoint["stage"]
        print(f"\nFound checkpoint at stage: {stage}")
        
        stage_map = {
            "exp1_bouncing_ball": 1,
            "exp2_hopper_pano": 2,
            "exp3_bulletproof": 3,
            "figures": 4,
            "complete": 5
        }
        start_stage = stage_map.get(stage, 0)
        
        if start_stage == 5:
            print("All experiments already completed. Clear checkpoint to re-run.")
            return 0
    
    stages = [
        ("Experiment 1: Bouncing Ball", run_bouncing_ball, "exp1_bouncing_ball"),
        ("Experiment 2: Hopper PANO", run_hopper_pano, "exp2_hopper_pano"),
        ("Experiment 3: Bulletproof", run_bulletproof, "exp3_bulletproof"),
        ("Figures", run_figures, "figures"),
    ]
    
    # Run stages
    for i, (name, func, checkpoint_name) in enumerate(stages):
        if i < start_stage:
            print(f"\nSkipping {name} (already completed)")
            continue
        
        print(f"\n{'='*70}")
        print(f"Stage {i+1}/{len(stages)}: {name}")
        print(f"Memory: {monitor_memory():.1f} MB")
        print(f"{'='*70}")
        
        save_checkpoint("experiment_suite", checkpoint_name, (i / len(stages)) * 100, {})
        
        success = func()
        
        if not success:
            print(f"\n{'='*70}")
            print(f"FAILED at stage: {name}")
            print("Check logs in: logs/")
            print("Resume with: python run_experiments.py")
            print("="*70)
            return 1
    
    # All complete
    total_elapsed = time.time() - overall_start
    
    save_checkpoint("experiment_suite", "complete", 100.0, {
        "total_seconds": total_elapsed
    })
    
    print("\n" + "="*70)
    print("ALL EXPERIMENTS COMPLETE")
    print(f"Total time: {format_duration(total_elapsed)}")
    print(f"Finished: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("="*70)
    print("\nResults saved to: results/phase6/")
    print("Logs saved to: logs/")
    print("\nKey files:")
    print("  - hopper_pano_results.json")
    print("  - bulletproof_results.json")
    print("  - figure*.pdf")
    
    return 0

if __name__ == "__main__":
    sys.exit(main())
