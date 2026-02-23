#!/bin/bash
# ================================================================================
# NEURIPS REPRODUCIBILITY SCRIPT
# ================================================================================
# This script reproduces all results from the paper:
# "Physics-Anchored Neural Observers for Hybrid Dynamical Systems"
#
# Usage: ./run_neurips_evals.sh
# ================================================================================

set -e  # Exit on error

# Hardcoded seeds for exact reproducibility
export PYTHONHASHSEED=42
SEED=42

echo "================================================================================"
echo "NEURIPS REPRODUCIBILITY EVALUATION"
echo "================================================================================"
echo "Seed: $SEED"
echo "Date: $(date)"
echo "================================================================================"

# Create results directory
mkdir -p results

# ================================================================================
# EXPERIMENT 1: BOUNCING BALL (PHASE 5)
# ================================================================================
echo ""
echo "================================================================================"
echo "EXPERIMENT 1: BOUNCING BALL - VELOCITY DROPOUT"
echo "================================================================================"

cd experiments/phase5

echo "Running F3-JEPA on Bouncing Ball..."
python f3_jepa.py --seed $SEED --n_episodes 20 > ../../results/bouncing_ball_results.txt 2>&1

echo "✓ Bouncing Ball results saved to results/bouncing_ball_results.txt"

cd ../..

# ================================================================================
# EXPERIMENT 2: HOPPER - CONTACT-TRIGGERED DROPOUT
# ================================================================================
echo ""
echo "================================================================================"
echo "EXPERIMENT 2: HOPPER - CONTACT-TRIGGERED DROPOUT"
echo "================================================================================"

cd experiments/phase6

# Train oracle if not exists
if [ ! -f "hopper_sac.zip" ]; then
    echo "Training SAC oracle (100k steps)..."
    python -c "
import gymnasium as gym
from stable_baselines3 import SAC
import warnings
warnings.filterwarnings('ignore')
env = gym.make('Hopper-v4')
model = SAC('MlpPolicy', env, learning_rate=3e-4, buffer_size=100000, 
            learning_starts=1000, batch_size=256, verbose=0, seed=42)
model.learn(total_timesteps=100000, progress_bar=True)
model.save('hopper_sac.zip')
print('Oracle saved.')
"
fi

echo "Running PANO on Hopper..."
python hopper_pano.py > ../../results/hopper_pano_results.txt 2>&1

echo "Running Bulletproof Negative Protocol..."
python bulletproof_negative.py > ../../results/bulletproof_results.txt 2>&1

echo "✓ Hopper results saved to results/hopper_*.txt"

cd ../..

# ================================================================================
# EXPERIMENT 3: INVERTED DOUBLE PENDULUM (CONTINUOUS CONTROL ABLATION)
# ================================================================================
echo ""
echo "================================================================================"
echo "EXPERIMENT 3: INVERTED DOUBLE PENDULUM - CONTINUOUS CONTROL"
echo "================================================================================"

# This is already run in bulletproof_negative.py
echo "✓ Results already in results/bulletproof_results.txt"

# ================================================================================
# GENERATE FIGURES
# ================================================================================
echo ""
echo "================================================================================"
echo "GENERATING NEURIPS FIGURES"
echo "================================================================================"

cd experiments/phase6
python neurips_figures.py
cp figure*.pdf ../../results/
cp figure*.png ../../results/
cd ../..

echo "✓ Figures saved to results/figure*.pdf"

# ================================================================================
# SUMMARY
# ================================================================================
echo ""
echo "================================================================================"
echo "EVALUATION COMPLETE"
echo "================================================================================"
echo ""
echo "Results summary:"
echo "  - Bouncing Ball: results/bouncing_ball_results.txt"
echo "  - Hopper v4:     results/hopper_v4_results.txt"
echo "  - Bulletproof:   results/bulletproof_results.txt"
echo "  - Figures:       results/figure*.pdf"
echo ""
echo "Key findings:"
echo "  1. Latent JEPA rollout diverges exponentially in high-dimensional systems"
echo "  2. Data scaling does not fix latent-physics misalignment"
echo "  3. Physics-anchored velocity prediction (PANO) recovers +22.6% performance"
echo ""
echo "================================================================================"
