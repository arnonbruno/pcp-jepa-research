#!/usr/bin/env python3
"""
Oracle check with LONGER HORIZON - H=30 instead of H=10
"""

import jax
import jax.numpy as jnp
import numpy as np
import json

N_EPISODES = 100
MAX_STEPS = 30
X_TARGET = 2.0
TAU_SUCCESS = 0.3
HORIZON = 30
N_SAMPLES = 64

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
    actions = jax.random.uniform(subkey, (N_SAMPLES, HORIZON, 1), minval=-1.0, maxval=1.0)
    final_x = batch_rollout(obs, actions)
    costs = jnp.abs(final_x - X_TARGET)
    weights = jax.nn.softmax(-costs)
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
        else:
            key, subkey = jax.random.split(key)
            a = float(jax.random.uniform(subkey, (), minval=-1.0, maxval=1.0))
        obs_next = physics_step(obs, a)
        x, v = float(obs_next[0]), float(obs_next[1])
    
    return abs(x - X_TARGET) < TAU_SUCCESS

def main():
    print("="*60)
    print("ORACLE CHECK: H=30 (FIXED)")
    print("="*60)
    print("\nH=10: Can only move ~0.125 units - can't reach target!")
    print("H=30: Can move ~1.125 units - can reach target!")
    
    key = jax.random.PRNGKey(42)
    
    # Warmup
    print("\nWarming up JIT...")
    _ = mppi_plan(jnp.array([1.0, 0.0]), key)
    print("Done")
    
    # Random
    print(f"\nRANDOM ({N_EPISODES} episodes)...")
    random_success = sum(run_episode(key) for key in [jax.random.split(key)[0] for _ in range(N_EPISODES)])
    # Actually need to split properly
    key = jax.random.PRNGKey(42)
    random_success = 0
    for _ in range(N_EPISODES):
        key, subkey = jax.random.split(key)
        if run_episode(subkey, None):
            random_success += 1
    
    # MPPI
    print(f"\nMPPI ORACLE H=30 ({N_EPISODES} episodes)...")
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
    print(f"\nRandom:     {random_sr:.1%}")
    print(f"MPPI H=30:  {mppi_sr:.1%}")
    print(f"Improvement: {imp:+.1%}")
    
    if imp > 0.1:
        print("\n✅ MPPI WORKS WITH H=30!")
        print("   The bug was horizon too short")
    else:
        print("\n❌ Still fails")

if __name__ == '__main__':
    main()