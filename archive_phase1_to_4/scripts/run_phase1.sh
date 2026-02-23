#!/bin/bash
# Run Phase 1 experiments

echo "Starting Phase 1 experiments..."
echo "Goal: Establish the failure law for long-horizon planning"
echo ""

cd "$(dirname "$0")/.."

# Create results directory
mkdir -p results/phase1

# Run experiments
python experiments/phase1/run_all.py

echo ""
echo "Phase 1 complete. Check results/phase1/ for outputs."