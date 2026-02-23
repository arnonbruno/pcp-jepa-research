#!/usr/bin/env python3
"""
ORACLE CHECK: MPPI with TRUE SIMULATOR (Optimized)

Faster version with JIT-compiled physics.
"""

import jax
import jax.numpy as jnp
import numpy as np
import json
import time
import sys
sys.path.insert(0, '/home/ulluboz/pcp-jepa-research')

# Config
N_EPISODES = 100  # Fewer for faster test
MAX_STEPS = 30
HORIZON = 10
N_SAMPLES = 64
X_TARGET = 2.0
TAU_SUCCESS = 0.3

# JIT-compiled physics
@jax.jit
def physics_step_jax(obs, a):
    """JAX physics step with bounce."""
    x, v = obs[0], obs[1]
    dt, restitution = 0.05, 0.8
    
    v_new = v + a * dt
    x_new = x + v_new * dt
    
    # Bounces
    x_new = jnp.where(x_new < 0, -x_new, x_new)
    v_new = jnp.where(x_new < 0, -v_new * restitution, v_new)
    x_new = jnp.where(x_new > 4, 8 - x_new, x_new)
    v_new = jnp.where(x_new > 4, -v_new * restitution, v_new)
    
    return jnp.array([x_new, v_new])

@jax.jit
def batch_rollout_true(obs, actions):
    """Vectorized rollout with TRUE physics."""
    def single(action_seq):
        current = obs
        for t in range(HORIZON):
            current = physics_step_jax(current, action_seq[t])
        return current[0]
    return jax.vmap(single)(actions)

@jax.jit
def mppi_oracle(obs, key):
    """MPPI using JIT-compiled true physics."""
    key, subkey = jax.random.split(key)
    actions = jax.random.uniform(subkey, (N_SAMPLES, HORIZON, 1), minval=-1.0, maxval=1.0)
    
    final_x = batch_rollout_true(obs, actions)
    costs = jnp.abs(final_x - X_TARGET)
    weights = jax.nn.softmax(-costs)
    
    # Weighted sum of first actions
    best_action = jnp.sum(weights * actions[:, 0, 0])  # Scalar
    return best_action

def run_episode(key, use_mppi=False):
    key, subkey = jax.random.split(key)
    x = float(jax.random.uniform(subkey, (), minval=0.5, maxval=3.5))
    key, subkey = jax.random.split(key)
    v = float(jax.random.uniform(subkey, (), minval=-0.5, maxval=0.5))
    
    for _ in range(MAX_STEPS):
        obs = jnp.array([x, v])
        if use_mppi:
            key, subkey = jax.random.split(key)
            a = float(mppi_oracle(obs, subkey))
        else:
            key, subkey = jax.random.split(key)
            a = float(jax.random.uniform(subkey, (), minval=-1.0, maxval=1.0))
        
        obs_next = physics_step_jax(obs, a)
        x, v = float(obs_next[0]), float(obs_next[1])
    
    miss = abs(x - X_TARGET)
    return {'miss': miss, 'success': miss < TAU_SUCCESS}

def main():
    print("="*60)
    print("ORACLE CHECK: MPPI with TRUE PHYSICS (JIT)")
    print("="*60)
    
    key = jax.random.PRNGKey(42)
    
    # Warmup JIT
    print("\nWarming up JIT...")
    _ = physics_step_jax(jnp.array([1.0, 0.0]), 0.5)
    _ = batch_rollout_true(jnp.array([1.0, 0.0]), jnp.zeros((N_SAMPLES, HORIZON, 1)))
    _ = mppi_oracle(jnp.array([1.0, 0.0]), key)
    print("Done")
    
    results = {}
    
    for method in ['random', 'mppi_oracle']:
        print(f"\n{method.upper()} ({N_EPISODES} episodes)...")
        use_mppi = (method == 'mppi_oracle')
        
        successes = 0
        misses = []
        
        start = time.time()
        for ep in range(N_EPISODES):
            key, subkey = jax.random.split(key)
            r = run_episode(subkey, use_mppi)
            
            if r['success']:
                successes += 1
            misses.append(r['miss'])
        
        sr = successes / N_EPISODES
        avg_miss = np.mean(misses)
        ci_sr = 1.96 * np.sqrt(sr * (1-sr) / N_EPISODES)
        
        results[method] = {
            'success_rate': float(sr),
            'success_ci': float(ci_sr),
            'avg_miss': float(avg_miss),
            'elapsed': time.time() - start
        }
        
        print(f"  Success: {sr:.1%} ± {ci_sr:.1%}")
        print(f"  Avg miss: {avg_miss:.3f}")
        print(f"  Time: {results[method]['elapsed']:.1f}s")
    
    # Result
    print("\n" + "="*60)
    print("RESULT")
    print("="*60)
    
    imp = results['mppi_oracle']['success_rate'] - results['random']['success_rate']
    
    print(f"\nRandom:      {results['random']['success_rate']:.1%}")
    print(f"MPPI Oracle: {results['mppi_oracle']['success_rate']:.1%}")
    print(f"Improvement: {imp:+.1%}")
    
    if imp > 0.05:
        print("\n✅ ORACLE MPPI BEATS RANDOM")
        print("   → Planning helps for this task")
        print("   → Issue is MODEL quality, not planner")
    elif imp > 0:
        print("\n⚠️  ORACLE BARELY BETTER")
        print("   → Weak planning benefit")
    else:
        print("\n❌ ORACLE DOES NOT BEAT RANDOM")
        print("   → Task doesn't benefit from planning OR planner broken")
    
    with open('/home/ulluboz/pcp-jepa-research/results/phase2/oracle_jit.json', 'w') as f:
        json.dump(results, f, indent=2)
    print("\nSaved to oracle_jit.json")


if __name__ == '__main__':
    main()