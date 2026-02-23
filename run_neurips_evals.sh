#!/bin/bash
# ================================================================================
# NEURIPS REPRODUCIBILITY SCRIPT
# ================================================================================
# This script reproduces all results from the paper:
# "An Empirical Investigation into the Failure Modes of JEPA in
#  Hybrid Continuous Control"
#
# Produces:
#   results/phase5/  — Bouncing Ball experiments
#   results/phase6/  — Hopper + multi-env experiments
#   results/phase6/figure*.pdf  — Publication figures
#
# Usage:
#   ./run_neurips_evals.sh [--seed 42] [--episodes 100] [--oracle-steps 1000000]
# ================================================================================

set -euo pipefail

# Defaults
SEED=42
N_EPISODES=100
ORACLE_STEPS=1000000

# Parse arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --seed) SEED="$2"; shift 2 ;;
        --episodes) N_EPISODES="$2"; shift 2 ;;
        --oracle-steps) ORACLE_STEPS="$2"; shift 2 ;;
        *) echo "Unknown option: $1"; exit 1 ;;
    esac
done

export PYTHONHASHSEED=$SEED

echo "================================================================================"
echo "NEURIPS REPRODUCIBILITY EVALUATION"
echo "================================================================================"
echo "  Seed:           $SEED"
echo "  Episodes/method: $N_EPISODES"
echo "  Oracle steps:   $ORACLE_STEPS"
echo "  Date:           $(date)"
echo "================================================================================"

# Create results directories
mkdir -p results/phase5
mkdir -p results/phase6

# ================================================================================
# EXPERIMENT 1: BOUNCING BALL (Phase 5)
# ================================================================================
echo ""
echo "================================================================================"
echo "EXPERIMENT 1: BOUNCING BALL — VELOCITY DROPOUT"
echo "================================================================================"

cd experiments/phase5

echo "Running F3-JEPA on Bouncing Ball..."
python f3_jepa.py 2>&1 | tee ../../results/phase5/bouncing_ball.log

echo "✓ Bouncing Ball results saved"

cd ../..

# ================================================================================
# EXPERIMENT 2: HOPPER — PANO vs ALL BASELINES
# ================================================================================
echo ""
echo "================================================================================"
echo "EXPERIMENT 2: HOPPER — PANO vs BASELINES (Oracle, Frozen, EKF)"
echo "================================================================================"

cd experiments/phase6

echo "Running PANO evaluation ($N_EPISODES episodes per method)..."
python hopper_pano.py \
    --n-episodes $N_EPISODES \
    --oracle-steps $ORACLE_STEPS \
    --seed $SEED \
    --results-dir ../../results/phase6 \
    2>&1 | tee ../../results/phase6/hopper_pano.log

echo "✓ PANO results saved to results/phase6/hopper_pano_results.json"

# ================================================================================
# EXPERIMENT 3: BULLETPROOF NEGATIVE PROTOCOL
#   - Data Scaling Law
#   - Multi-Environment Ablation (Hopper, Walker2d, HalfCheetah, InvertedDoublePendulum)
#   - Impact Horizon Profiling
# ================================================================================
echo ""
echo "================================================================================"
echo "EXPERIMENT 3: BULLETPROOF NEGATIVE PROTOCOL (3 sub-experiments)"
echo "================================================================================"

echo "Running Bulletproof Negative Protocol..."
python bulletproof_negative.py \
    --seed $SEED \
    --oracle-steps $ORACLE_STEPS \
    --n-eval-episodes $(($N_EPISODES / 2)) \
    --results-dir ../../results/phase6 \
    2>&1 | tee ../../results/phase6/bulletproof.log

echo "✓ Bulletproof results saved to results/phase6/bulletproof_results.json"

cd ../..

# ================================================================================
# GENERATE PUBLICATION FIGURES
# ================================================================================
echo ""
echo "================================================================================"
echo "GENERATING NEURIPS FIGURES (data-driven from JSON results)"
echo "================================================================================"

cd experiments/phase6

python neurips_figures.py \
    --results-dir ../../results/phase6 \
    --output-dir ../../results/phase6

cd ../..

echo "✓ Figures saved to results/phase6/figure*.pdf"

# ================================================================================
# FINAL SUMMARY
# ================================================================================
echo ""
echo "================================================================================"
echo "ALL EXPERIMENTS COMPLETE"
echo "================================================================================"
echo ""
echo "Results files:"
echo "  results/phase5/bouncing_ball.log         — Bouncing Ball evaluation"
echo "  results/phase6/hopper_pano_results.json  — PANO main results (with stats)"
echo "  results/phase6/bulletproof_results.json   — Negative result validation"
echo "  results/phase6/hopper_pano.log           — PANO detailed output"
echo "  results/phase6/bulletproof.log           — Bulletproof detailed output"
echo ""
echo "Publication figures:"
echo "  results/phase6/figure1_latent_drift.pdf       — Exponential divergence"
echo "  results/phase6/figure2_data_scaling.pdf        — Data scaling asymptote"
echo "  results/phase6/figure3_performance_recovery.pdf — PANO vs baselines"
echo "  results/phase6/figure4_multi_env.pdf           — Multi-environment ablation"
echo ""
echo "Key findings:"
echo "  1. Standard Latent JEPA rollout diverges exponentially (all envs)"
echo "  2. More data does NOT fix prediction-velocity misalignment"
echo "  3. PANO (physics-anchored velocity) recovers partial performance"
echo "  4. All results include Welch's t-tests and 95% bootstrap CIs"
echo ""
echo "Cite: PANO: Physics-Anchored Neural Observers for Hybrid Control (NeurIPS 2026)"
echo "================================================================================"
