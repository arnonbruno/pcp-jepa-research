#!/usr/bin/env python3
"""
ORACLE CHECK: MPPI with TRUE SIMULATOR

This is the decisive test:
- If MPPI with true physics doesn't beat random → task/planner is broken
- If it does beat random → model is the issue
"""

import jax
import jax.numpy as jnp
import numpy as np
import json
import time
import sys
sys.path.insert(0, '/home/ulluboz/pcp-jepa-research')

from experiments.phase2.train_action_controllable import physics_step

# Config
N_EPISODES = 300
MAX_STEPS = 30
HORIZON = 10
N_SAMPLES = 64
X_TARGET = 2.0
TAU_SUCCESS = 0.3

def true_physics_rollout(x0, v0, actions):
    """Rollout using TRUE physics."""
    x, v = x0, v0
    for a in actions:
        x, v = physics_step(x, v, float(a[0]) if hasattr(a, '__len__') else float(a))
    return x

def mppi_oracle(obs, key, horizon=HORIZON, n_samples=N_SAMPLES):
    """MPPI using TRUE physics for rollouts."""
    x0, v0 = float(obs[0]), float(obs[1])
    
    # Sample action sequences
    key, subkey = jax.random.split(key)
    actions = jax.random.uniform(subkey, (n_samples, horizon, 1), minval=-1.0, maxval=1.0)
    
    # Evaluate with TRUE physics
    final_xs = []
    for i in range(n_samples):
        final_x = true_physics_rollout(x0, v0, actions[i])
        final_xs.append(final_x)
    
    final_xs = jnp.array(final_xs)
    costs = jnp.abs(final_xs - X_TARGET)
    weights = jax.nn.softmax(-costs)
    
    # Weighted average
    best_action = jnp.sum(weights[:, None] * actions[:, 0, :], axis=0)
    return float(best_action[0])

def run_episode(key, use_mppi=False):
    key, subkey = jax.random.split(key)
    x = float(jax.random.uniform(subkey, (), minval=0.5, maxval=3.5))
    key, subkey = jax.random.split(key)
    v = float(jax.random.uniform(subkey, (), minval=-0.5, maxval=0.5))
    
    for _ in range(MAX_STEPS):
        if use_mppi:
            obs = jnp.array([x, v])
            key, subkey = jax.random.split(key)
            a = mppi_oracle(obs, subkey)
        else:
            key, subkey = jax.random.split(key)
            a = float(jax.random.uniform(subkey, (), minval=-1.0, maxval=1.0))
        
        x, v = physics_step(x, v, a)
    
    miss = abs(x - X_TARGET)
    return {
        'final_x': x,
        'miss': miss,
        'success': miss < TAU_SUCCESS,
        'catastrophic': miss > 0.5
    }

def main():
    print("="*60)
    print("ORACLE CHECK: MPPI with TRUE PHYSICS")
    print("="*60)
    print(f"\nThis is the decisive test:")
    print(f"  - If MPPI+true_physics doesn't beat random → task/planner broken")
    print(f"  - If it beats random → model is the issue")
    print()
    
    key = jax.random.PRNGKey(42)
    results = {}
    
    for method in ['random', 'mppi_oracle']:
        print(f"\n{method.upper()} ({N_EPISODES} episodes)...")
        use_mppi = (method == 'mppi_oracle')
        
        successes = 0
        catastrophics = 0
        misses = []
        final_xs = []
        
        start = time.time()
        for ep in range(N_EPISODES):
            key, subkey = jax.random.split(key)
            r = run_episode(subkey, use_mppi)
            
            if r['success']:
                successes += 1
            if r['catastrophic']:
                catastrophics += 1
            misses.append(r['miss'])
            final_xs.append(r['final_x'])
            
            if (ep + 1) % 50 == 0:
                elapsed = time.time() - start
                print(f"  {ep+1}/{N_EPISODES} ({elapsed:.0f}s)")
        
        sr = successes / N_EPISODES
        cr = catastrophics / N_EPISODES
        avg_miss = np.mean(misses)
        ci_sr = 1.96 * np.sqrt(sr * (1-sr) / N_EPISODES)
        
        results[method] = {
            'success_rate': float(sr),
            'success_ci': float(ci_sr),
            'catastrophic_rate': float(cr),
            'avg_miss': float(avg_miss),
            'final_x_std': float(np.std(final_xs))
        }
        
        print(f"\n  Success: {sr:.1%} ± {ci_sr:.1%}")
        print(f"  Catastrophic: {cr:.1%}")
        print(f"  Avg miss: {avg_miss:.3f}")
        print(f"  Final x std: {results[method]['final_x_std']:.3f}")
    
    # Comparison
    print("\n" + "="*60)
    print("RESULT")
    print("="*60)
    
    imp = results['mppi_oracle']['success_rate'] - results['random']['success_rate']
    
    print(f"\nRandom:      {results['random']['success_rate']:.1%} success")
    print(f"MPPI Oracle: {results['mppi_oracle']['success_rate']:.1%} success")
    print(f"Improvement: {imp:+.1%}")
    
    print("\n" + "="*60)
    print("DIAGNOSIS")
    print("="*60)
    
    if imp > 0.05:
        print("\n✅ MPPI ORACLE BEATS RANDOM")
        print("   → Task benefits from planning")
        print("   → Issue is with MODEL, not planner")
        print("   → Need to fix model to match physics better")
    elif imp > 0:
        print("\n⚠️  MPPI ORACLE BARELY BEATS RANDOM")
        print("   → Task has weak planning benefit")
        print("   → Need harder task or better planner settings")
    else:
        print("\n❌ MPPI ORACLE DOES NOT BEAT RANDOM")
        print("   → Task does NOT benefit from planning")
        print("   → OR MPPI implementation is broken")
        print("   → Need to check:")
        print("      - Cost function definition")
        print("      - Action scaling")
        print("      - Horizon vs task timescale")
    
    # Save
    with open('/home/ulluboz/pcp-jepa-research/results/phase2/oracle_check.json', 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved to oracle_check.json")


if __name__ == '__main__':
    main()