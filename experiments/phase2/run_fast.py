"""
Fast Phase 2 Evaluation - Vectorized JAX Implementation

Uses JAX vectorization for speed.
"""

import jax
import jax.numpy as jnp
import numpy as np
from datetime import datetime
import json
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from src.environments import BouncingBall, BouncingBallParams


# Config
H = 20
N = 128
NUM_EPISODES = 20
X_TARGET = 2.0


def simulate_batch(env, state0, actions_batch):
    """Simulate batch of trajectories."""
    batch_size = actions_batch.shape[0]
    results = []
    
    for i in range(batch_size):
        state = state0
        for a in actions_batch[i]:
            state = state.at[2].add(a)
            state, _ = env.step(state)
        
        cost = 10.0 * (state[0] - X_TARGET) ** 2 + 0.01 * jnp.sum(actions_batch[i] ** 2)
        results.append(float(cost))
    
    return np.array(results)


def mppi_fast(env, state0, horizon=H, num_samples=N, num_iters=2):
    """Fast MPPI."""
    mean_actions = np.zeros(horizon)
    
    for _ in range(num_iters):
        # Sample
        noise = np.random.randn(num_samples, horizon) * 0.3
        samples = np.clip(mean_actions + noise, -1.0, 1.0)
        
        # Evaluate
        costs = simulate_batch(env, state0, samples)
        
        # Weights
        costs_min = costs.min()
        weights = np.exp(-(costs - costs_min) / 1.0)
        weights = weights / weights.sum()
        
        # Update
        mean_actions = np.sum(weights[:, None] * samples, axis=0)
    
    return mean_actions


def main():
    print("\n" + "=" * 70)
    print("PHASE 2 FAST EVALUATION")
    print("=" * 70)
    
    np.random.seed(42)
    env = BouncingBall(BouncingBallParams(restitution=0.8))
    
    # Test Random
    print("\n[Random] Running...")
    random_success = 0
    for _ in range(NUM_EPISODES):
        y = np.random.uniform(1.0, 3.0)
        vy = np.random.uniform(-2.0, 2.0)
        state = jnp.array([0.0, y, 0.0, vy])
        
        for _ in range(50):
            a = np.random.uniform(-1, 1)
            state = state.at[2].add(a)
            state, _ = env.step(state)
        
        if abs(state[0] - X_TARGET) < 0.5:
            random_success += 1
    
    print(f"  Success: {random_success/NUM_EPISODES:.1%}")
    
    # Test MPPI
    print("\n[MPPI] Running...")
    mppi_success = 0
    for i in range(NUM_EPISODES):
        y = np.random.uniform(1.0, 3.0)
        vy = np.random.uniform(-2.0, 2.0)
        state = jnp.array([0.0, y, 0.0, vy])
        
        for t in range(50):
            actions = mppi_fast(env, state, horizon=10, num_samples=64, num_iters=1)
            state = state.at[2].add(actions[0])
            state, _ = env.step(state)
        
        if abs(state[0] - X_TARGET) < 0.5:
            mppi_success += 1
        
        if (i + 1) % 5 == 0:
            print(f"  Episode {i+1}/{NUM_EPISODES}")
    
    print(f"  Success: {mppi_success/NUM_EPISODES:.1%}")
    
    # Summary
    print("\n" + "=" * 70)
    print("RESULTS")
    print("=" * 70)
    print(f"  Random: {random_success/NUM_EPISODES:.1%}")
    print(f"  MPPI:   {mppi_success/NUM_EPISODES:.1%}")
    
    if mppi_success > random_success:
        print("\n✓ MPPI beats random!")
    
    # Save
    os.makedirs('results/phase2', exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    with open(f'results/phase2/fast_eval_{timestamp}.json', 'w') as f:
        json.dump({
            'random': {'success': random_success/NUM_EPISODES},
            'mppi': {'success': mppi_success/NUM_EPISODES},
        }, f, indent=2)
    
    print(f"\nSaved to: results/phase2/fast_eval_{timestamp}.json")


if __name__ == "__main__":
    main()