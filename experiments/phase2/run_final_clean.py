"""
Phase 2 Final - Clean Version

Tests:
A) Random vs MPPI baseline
"""

import jax
import jax.numpy as jnp
import numpy as np
from datetime import datetime
import json
import os
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from src.environments import BouncingBall, BouncingBallParams

# Config
X_TARGET = 2.0
NUM_EPISODES = 20

np.random.seed(42)
env = BouncingBall(BouncingBallParams(restitution=0.8))

print("\n" + "=" * 70)
print("PHASE 2 FINAL EVALUATION")
print("=" * 70)

# Random baseline
print("\n[Random] Running 20 episodes...")
random_success = 0
for i in range(NUM_EPISODES):
    y = np.random.uniform(1.0, 3.0)
    vy = np.random.uniform(-2.0, 2.0)
    state = jnp.array([0.0, y, 0.0, vy])
    
    for _ in range(50):
        a = np.random.uniform(-1, 1)
        state = state.at[2].add(a)
        state, _ = env.step(state)
    
    if abs(state[0] - X_TARGET) < 0.5:
        random_success += 1

random_rate = random_success / NUM_EPISODES
print(f"  Success: {random_rate:.1%}")

# Simple MPPI (fast version)
print("\n[MPPI] Running 20 episodes...")
mppi_success = 0
for ep in range(NUM_EPISODES):
    y = np.random.uniform(1.0, 3.0)
    vy = np.random.uniform(-2.0, 2.0)
    state = jnp.array([0.0, y, 0.0, vy])
    
    for step in range(50):
        # Simple planning: sample 64 trajectories
        best_cost = float('inf')
        best_action = 0.0
        
        for _ in range(64):
            # Random trajectory
            s = state
            total_cost = 0.0
            
            for h in range(10):
                a = np.random.uniform(-1, 1)
                s = s.at[2].add(a * 0.1)
                s, _ = env.step(s)
                total_cost += (s[0] - X_TARGET) ** 2
            
            total_cost += 10.0 * (s[0] - X_TARGET) ** 2
            
            if total_cost < best_cost:
                best_cost = total_cost
                best_action = a
        
        # Execute best action
        state = state.at[2].add(best_action * 0.1)
        state, _ = env.step(state)
    
    if abs(state[0] - X_TARGET) < 0.5:
        mppi_success += 1
    
    if (ep + 1) % 5 == 0:
        print(f"  Episode {ep+1}/{NUM_EPISODES}")

mppi_rate = mppi_success / NUM_EPISODES
print(f"  Success: {mppi_rate:.1%}")

# Summary
print("\n" + "=" * 70)
print("RESULTS")
print("=" * 70)
print(f"  Random: {random_rate:.1%}")
print(f"  MPPI:   {mppi_rate:.1%}")

if mppi_rate > random_rate:
    print(f"\n✓ MPPI beats random!")
    print("✓ Test A PASSED")
else:
    print(f"\n✗ MPPI does NOT beat random")

# Save
os.makedirs('results/phase2', exist_ok=True)
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
with open(f'results/phase2/final_{timestamp}.json', 'w') as f:
    json.dump({
        'random': {'success_rate': random_rate},
        'mppi': {'success_rate': mppi_rate},
        'mppi_beats_random': mppi_rate > random_rate,
    }, f, indent=2)

print(f"\nSaved to: results/phase2/final_{timestamp}.json")
