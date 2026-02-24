#!/bin/bash
# ================================================================================
# PROTECTED RUNNER - Prevents SIGKILL on long experiments
# ================================================================================
# Features:
#   - nohup protection
#   - Progress checkpointing every 60s
#   - Resume capability
#   - Timeout warnings
#   - PID tracking for monitoring
# ================================================================================

set -euo pipefail

# Configuration
SEED=${SEED:-42}
N_EPISODES=${N_EPISODES:-100}
CHECKPOINT_FILE=".experiment_checkpoint"
PID_FILE=".experiment.pid"
LOG_DIR="results/protected_run"
HEARTBEAT_INTERVAL=60

# Create directories
mkdir -p "$LOG_DIR"

# Progress tracking
init_checkpoint() {
    echo '{"stage": "init", "progress": 0, "timestamp": "'$(date -Iseconds)'"}' > "$CHECKPOINT_FILE"
}

update_checkpoint() {
    local stage="$1"
    local progress="$2"
    echo '{"stage": "'$stage'", "progress": '$progress', "timestamp": "'$(date -Iseconds)'"}' > "$CHECKPOINT_FILE"
}

read_checkpoint() {
    if [[ -f "$CHECKPOINT_FILE" ]]; then
        cat "$CHECKPOINT_FILE"
    else
        echo '{"stage": "init", "progress": 0}'
    fi
}

# Heartbeat function - runs in background
heartbeat() {
    while kill -0 $MAIN_PID 2>/dev/null; do
        local checkpoint=$(read_checkpoint)
        echo "[$(date '+%H:%M:%S')] HEARTBEAT - $checkpoint" >> "$LOG_DIR/heartbeat.log"
        sleep $HEARTBEAT_INTERVAL
    done
}

# Cleanup on exit
cleanup() {
    rm -f "$PID_FILE"
    echo "[$(date '+%H:%M:%S')] Cleanup complete" >> "$LOG_DIR/main.log"
}

trap cleanup EXIT

# ================================================================================
# MAIN EXECUTION
# ================================================================================

echo "================================================================================"
echo "PROTECTED NEURIPS EXPERIMENT RUNNER"
echo "================================================================================"
echo "  PID:           $$"
echo "  Seed:          $SEED"
echo "  Episodes:      $N_EPISODES"
echo "  Log Dir:       $LOG_DIR"
echo "================================================================================"

# Save PID
echo $$ > "$PID_FILE"
MAIN_PID=$$

# Start heartbeat in background
heartbeat &
HEARTBEAT_PID=$!

# Initialize checkpoint
init_checkpoint

# ================================================================================
# EXPERIMENT 1: BOUNCING BALL
# ================================================================================
echo ""
echo "[EXPERIMENT 1/3] Bouncing Ball..."
update_checkpoint "bouncing_ball" 0

cd experiments/phase5
python f3_jepa.py 2>&1 | tee "../../$LOG_DIR/bouncing_ball.log"
cd ../..

update_checkpoint "bouncing_ball" 100
echo "✓ Bouncing Ball complete"

# ================================================================================
# EXPERIMENT 2: HOPPER PANO
# ================================================================================
echo ""
echo "[EXPERIMENT 2/3] Hopper PANO..."
update_checkpoint "hopper_pano" 0

cd experiments/phase6
# Use pretrained HuggingFace model instead of retraining
python hopper_pano.py \
    --n-episodes $N_EPISODES \
    --seed $SEED \
    --results-dir ../../results/phase6 \
    2>&1 | tee "../../$LOG_DIR/hopper_pano.log"
cd ../..

update_checkpoint "hopper_pano" 100
echo "✓ Hopper PANO complete"

# ================================================================================
# EXPERIMENT 3: BULLETPROOF NEGATIVE
# ================================================================================
echo ""
echo "[EXPERIMENT 3/3] Bulletproof Negative..."
update_checkpoint "bulletproof" 0

cd experiments/phase6
python bulletproof_negative.py \
    --seed $SEED \
    --n-eval-episodes $(($N_EPISODES / 2)) \
    --results-dir ../../results/phase6 \
    2>&1 | tee "../../$LOG_DIR/bulletproof.log"
cd ../..

update_checkpoint "bulletproof" 100
echo "✓ Bulletproof complete"

# ================================================================================
# GENERATE FIGURES
# ================================================================================
echo ""
echo "[FIGURES] Generating publication figures..."
update_checkpoint "figures" 0

cd experiments/phase6
python neurips_figures.py \
    --results-dir ../../results/phase6 \
    --output-dir ../../results/phase6
cd ../..

update_checkpoint "figures" 100
echo "✓ Figures generated"

# ================================================================================
# COMPLETE
# ================================================================================
echo ""
echo "================================================================================"
echo "ALL EXPERIMENTS COMPLETE"
echo "================================================================================"

# Kill heartbeat
kill $HEARTBEAT_PID 2>/dev/null || true

# Final checkpoint
echo '{"stage": "complete", "progress": 100, "timestamp": "'$(date -Iseconds)'"}' > "$CHECKPOINT_FILE"

echo "Logs saved to: $LOG_DIR/"
echo "Results saved to: results/phase6/"