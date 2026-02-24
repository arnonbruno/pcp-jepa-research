#!/bin/bash
# Simple runner - just run each experiment sequentially
set -e

cd ~/pcp-jepa-research

echo "=== EXPERIMENT 1: Bouncing Ball ===" 
cd experiments/phase5
python f3_jepa.py
cd ../..

echo ""
echo "=== EXPERIMENT 2: Hopper PANO ==="
cd experiments/phase6
python hopper_pano.py --n-episodes 100 --seed 42 --results-dir ../../results/phase6
cd ../..

echo ""
echo "=== EXPERIMENT 3: Bulletproof Negative ==="
cd experiments/phase6  
python bulletproof_negative.py --seed 42 --n-eval-episodes 50 --results-dir ../../results/phase6
cd ../..

echo ""
echo "=== GENERATING FIGURES ==="
cd experiments/phase6
python neurips_figures.py --results-dir ../../results/phase6 --output-dir ../../results/phase6
cd ../..

echo ""
echo "=== ALL COMPLETE ==="
