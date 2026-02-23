#!/usr/bin/env python3
"""
FINAL FIX: Larger horizon AND larger action scale

Root cause: With dt=0.05 and a in [-1,1]:
- After H=30 steps, max displacement ≈ 0.5 * a_max * dt² * H²
- For a_max=1, H=30: ≈ 0.5 * 1 * 0.0025 * 900 = 1.125 units
- But with wall bounces, effective movement is much less!

Fix: Either larger horizon OR larger action scale.
"""

import jax
import jax.numpy as jnp
import numpy as np
import json

N_EPISODES = 100
MAX_STEPS = 30
X_TARGET = 2.0
TAU_SUCCESS = 0.3

# FIXED: Larger horizon AND larger action scale
HORIZON = 50
N_SAMPLES = 64
ACTION_SCALE = 2.0  # Allow stronger actions

@jax.jit
def physics_step(obs, a):
    x, v = obs[0], obs[1]
    v_new = v + a * 0.05
    x_new = x + v_new * 0.05
    x_new = jnp.where(x_new < 0, -x_new, x_new)
    v_new = jnp.where(x_new < 0, -v_new * 0.8, v_new)
    x_new = jnp.where(x_new > 4, 8 - x_new, x_new)
    v_new = jnp.where(x_new > 4, -v_new * 0.8, v_new)
    return jnp.array([x_new, v_new])

@jax.jit
def batch_rollout(obs, actions):
    def single(action_seq):
        current = obs
        for t in range(HORIZON):
            current = physics_step(current, action_seq[t])
        return current[0]
    return jax.vmap(single)(actions)

@jax.jit
def mppi_plan(obs, key):
    key, subkey = jax.random.split(key)
    # Scale actions to be larger
    actions = jax.random.uniform(subkey, (N_SAMPLES, HORIZON, 1), minval=-ACTION_SCALE, maxval=ACTION_SCALE)
    
    final_x = batch_rollout(obs, actions)
    
    # Cost: distance to target
    costs = jnp.abs(final_x - X_TARGET)
    
    # Softmax weights
    weights=jax.nn.softmax(-costs)
    
    # Best action
    return jnp.sum(weights * actions[:, 0, 0])

def run_episode(key, mppi_fn=None):
    key, subkey = jax.random.split(key)
    x = float(jax.random.uniform(subkey, (), minval=0.5, maxval=3.5))
    key, subkey = jax.random.split(key)
    v = float(jax.random.uniform(subkey, (), minval=-0.5, maxval=0.5))
    
    for _ in range(MAX_STEPS):
        obs = jnp.array([x, v])
        if mppi_fn:
            key, subkey = jax.random.split(key)
            a = float(mppi_fn(obs, subkey))
            # Clip action when executing (but allow planning with larger scale)
            a = float(np.clip(a, -1.0, 1.0))
        else:
            key, subkey = jax.random.split(key)
            a = float(jax.random.uniform(subkey, (), minval=-1.0, maxval=1.0))
        
        obs_next = physics_step(obs, a)
        x, v = float(obs_next[0]), float(obs_next[1])
    
    return abs(x - X_TARGET) < TAU_SUCCESS

def main():
    print("="*60)
    print("FINAL FIX: H=50, ACTION_SCALE=2.0")
    print("="*60)
    
    # Test reachability
    print("\nReachability test:")
    print(f"  H={HORIZON}, action_scale={ACTION_SCALE}")
    print(f"  Max displacement ≈ 0.5 * {ACTION_SCALE} * 0.05² * {HORIZON}² = {0.5 * ACTION_SCALE * 0.0025 * HORIZON**2:.2f} units")
    
    key = jax.random.PRNGKey(42)
    
    # Test from x=0.5
    obs = jnp.array([0.5, 0.0])
    key, subkey = jax.random.split(key)
    actions = jax.random.uniform(subkey, (N_SAMPLES, HORIZON, 1), minval=-ACTION_SCALE, maxval=ACTION_SCALE)
    final_x = batch_rollout(obs, actions)
    
    print(f"\nFrom x=0.5:")
    print(f"  Final x range: [{float(jnp.min(final_x)):.2f}, {float(jnp.max(final_x)):.2f}]")
    print(f"  Can reach target x=2.0? {float(jnp.min(final_x)) <= X_TARGET <= float(jnp.max(final_x))}")
    
    # Warmup
    print("\nWarming up JIT...")
    _ = mppi_plan(jnp.array([1.0, 0.0]), key)
    print("Done")
    
    # Random
    print(f"\nRANDOM ({N_EPISODES} episodes)...")
    key = jax.random.PRNGKey(42)
    random_success = 0
    for _ in range(N_EPISODES):
        key, subkey = jax.random.split(key)
        if run_episode(subkey, None):
            random_success += 1
    
    # MPPI
    print(f"\nMPPI H={HORIZON} ({N_EPISODES} episodes)...")
    key = jax.random.PRNGKey(42)
    mppi_success = 0
    for _ in range(N_EPISODES):
        key, subkey = jax.random.split(key)
        if run_episode(subkey, mppi_plan):
            mppi_success += 1
    
    random_sr = random_success / N_EPISODES
    mppi_sr = mppi_success / N_EPISODES
    imp = mppi_sr - random_sr
    
    print("\n" + "="*60)
    print("RESULT")
    print("="*60)
    print(f"\nRandom:    {random_sr:.1%}")
    print(f"MPPI H={HORIZON}: {mppi_sr:.1%}")
    print(f"Improvement: {imp:+.1%}")
    
    if imp > 0.1:
        print("\n✅ MPPI WORKS!")
        print("   The fix was larger horizon + action scale")
    elif imp > 0:
        print("\n⚠️ Small improvement")
    else:
        print("\n❌ Still issues")
    
    results = {
        'random': random_sr,
        'mppi': mppi_sr,
        'improvement': imp,
        'horizon': HORIZON,
        'action_scale': ACTION_SCALE
    }
    
    with open('/home/ulluboz/pcp-jepa-research/results/phase2/final_fixed_mppi.json', 'w') as f:
        json.dump(results, f, indent=2)

if __name__ == '__main__':
    main()