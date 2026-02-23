#!/usr/bin/env python3
"""
MPPI with explicit JIT warmup.
"""

import jax
import jax.numpy as jnp
import pickle
import numpy as np
import json
import time
import sys
sys.path.insert(0, '/home/ulluboz/pcp-jepa-research')

from experiments.phase2.train_action_controllable import (
    ActionControllableModel, physics_step
)

print("Starting imports...")
print(f"JAX devices: {jax.devices()}")

# Config
HORIZON = 10
N_SAMPLES = 64
N_EPISODES = 10
MAX_STEPS = 20
X_TARGET = 2.0

# Load model
print("\nLoading model...")
model = ActionControllableModel(latent_dim=32, hidden_dim=128, action_dim=1, obs_dim=2)

with open('/home/ulluboz/pcp-jepa-research/checkpoints/model_best.pkl', 'rb') as f:
    checkpoint = pickle.load(f)
params = checkpoint['params']
print(f"Loaded model (epoch {checkpoint['epoch']})")

# Pre-compile rollout function
print("\nCompiling batch rollout function...")

def batch_rollout(params, obs, actions):
    """Vectorized rollout: [N, H, 1] actions -> [N] final_x"""
    def single_rollout(action_seq):
        current = obs
        for t in range(HORIZON):
            current = model.apply(params, current, action_seq[t])
        return current[0]
    return jax.vmap(single_rollout)(actions)

# JIT compile
dummy_obs = jnp.array([1.0, 0.0])
dummy_actions = jnp.zeros((N_SAMPLES, HORIZON, 1))

print("  Compiling...")
start = time.time()
batch_rollout_jit = jax.jit(batch_rollout)
_ = batch_rollout_jit(params, dummy_obs, dummy_actions)
print(f"  Done in {time.time()-start:.1f}s")

# MPPI planning function
def mppi_plan(obs, key):
    """One MPPI planning step."""
    key, subkey = jax.random.split(key)
    actions = jax.random.uniform(subkey, (N_SAMPLES, HORIZON, 1), minval=-1.0, maxval=1.0)
    
    final_x = batch_rollout_jit(params, obs, actions)
    costs = jnp.abs(final_x - X_TARGET)
    weights = jax.nn.softmax(-costs)
    
    best_action = jnp.sum(weights[:, None] * actions[:, 0, :], axis=0)
    return best_action[0]

print("\nCompiling MPPI planner...")
mppi_plan_jit = jax.jit(mppi_plan)
_ = mppi_plan_jit(dummy_obs, jax.random.PRNGKey(0))
print("  Done")

# Run episodes
print("\n" + "=" * 50)
print("RUNNING EPISODES")
print("=" * 50)

key = jax.random.PRNGKey(42)
results = {'random': [], 'mppi': []}

for method in ['random', 'mppi']:
    print(f"\n{method.upper()} ({N_EPISODES} episodes)...")
    use_mppi = (method == 'mppi')
    
    successes = 0
    catastrophics = 0
    misses = []
    
    for ep in range(N_EPISODES):
        # Random initial state
        key, subkey = jax.random.split(key)
        x = float(jax.random.uniform(subkey, (), minval=0.5, maxval=3.5))
        key, subkey = jax.random.split(key)
        v = float(jax.random.uniform(subkey, (), minval=-0.5, maxval=0.5))
        
        for t in range(MAX_STEPS):
            obs = jnp.array([x, v])
            
            if use_mppi:
                key, subkey = jax.random.split(key)
                a = float(mppi_plan_jit(obs, subkey))
            else:
                key, subkey = jax.random.split(key)
                a = float(jax.random.uniform(subkey, (), minval=-1.0, maxval=1.0))
            
            # Real physics
            x, v = physics_step(x, v, a, dt=0.05, restitution=0.8)
        
        miss = abs(x - X_TARGET)
        if miss < 0.3:
            successes += 1
        if miss > 0.5 or abs(v) > 2.0:
            catastrophics += 1
        misses.append(miss)
        
        print(f"  Ep {ep+1}: miss={miss:.2f}, success={'Y' if miss<0.3 else 'N'}")
    
    results[f'{method}_success_rate'] = successes / N_EPISODES
    results[f'{method}_catastrophic_rate'] = catastrophics / N_EPISODES
    results[f'{method}_avg_miss'] = float(np.mean(misses))
    
    print(f"\n  Summary: success={results[f'{method}_success_rate']:.0%}, "
          f"cat={results[f'{method}_catastrophic_rate']:.0%}, "
          f"miss={results[f'{method}_avg_miss']:.2f}")

# Final comparison
print("\n" + "=" * 50)
print("COMPARISON")
print("=" * 50)
print(f"MPPI: {results['mppi_success_rate']:.0%} success, {results['mppi_avg_miss']:.2f} avg miss")
print(f"Random: {results['random_success_rate']:.0%} success, {results['random_avg_miss']:.2f} avg miss")
print(f"Improvement: {results['mppi_success_rate'] - results['random_success_rate']:+.0%}")

# Save
with open('/home/ulluboz/pcp-jepa-research/results/phase2/mppi_results.json', 'w') as f:
    json.dump(results, f, indent=2)
print("\nSaved to mppi_results.json")