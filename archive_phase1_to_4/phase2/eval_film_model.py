#!/usr/bin/env python3
"""
Evaluate the trained FiLM model with more episodes.
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

# Config
N_EPISODES = 500
MAX_STEPS = 30
HORIZON = 10
N_SAMPLES = 64
X_TARGET = 2.0
TAU_SUCCESS = 0.3
TAU_CAT = 0.5
V_MAX = 2.0

def create_mppi(model, params):
    def batch_rollout(obs, actions):
        def single(action_seq):
            current = obs
            for t in range(HORIZON):
                current = model.apply(params, current, action_seq[t])
            return current[0]
        return jax.vmap(single)(actions)
    
    rollout_jit = jax.jit(batch_rollout)
    
    def plan(obs, key):
        key, subkey = jax.random.split(key)
        actions = jax.random.uniform(subkey, (N_SAMPLES, HORIZON, 1), minval=-1.0, maxval=1.0)
        final_x = rollout_jit(obs, actions)
        costs = jnp.abs(final_x - X_TARGET)
        weights = jax.nn.softmax(-costs)
        return jnp.sum(weights[:, None] * actions[:, 0, :], axis=0)[0]
    
    return jax.jit(plan)

def run_episode(key, use_mppi, mppi_plan=None):
    key, subkey = jax.random.split(key)
    x = float(jax.random.uniform(subkey, (), minval=0.5, maxval=3.5))
    key, subkey = jax.random.split(key)
    v = float(jax.random.uniform(subkey, (), minval=-0.5, maxval=0.5))
    
    for _ in range(MAX_STEPS):
        obs = jnp.array([x, v])
        if use_mppi and mppi_plan:
            key, subkey = jax.random.split(key)
            a = float(mppi_plan(obs, subkey))
        else:
            key, subkey = jax.random.split(key)
            a = float(jax.random.uniform(subkey, (), minval=-1.0, maxval=1.0))
        x, v = physics_step(x, v, a)
    
    miss = abs(x - X_TARGET)
    return {
        'miss': miss,
        'success': miss < TAU_SUCCESS,
        'catastrophic': miss > TAU_CAT or abs(v) > V_MAX
    }

def main():
    print("Loading trained FiLM model...")
    model = ActionControllableModel(latent_dim=32, hidden_dim=128, action_dim=1, obs_dim=2)
    
    with open('/home/ulluboz/pcp-jepa-research/checkpoints/model_best.pkl', 'rb') as f:
        ckpt = pickle.load(f)
    params = ckpt['params']
    print(f"Loaded model (epoch {ckpt['epoch']}, sens={ckpt['metrics']['sensitivity']:.3f})")
    
    # Verify action influence
    print("\nAction influence test:")
    for x0, v0 in [(1.0, 0.0), (2.0, 0.5), (3.0, -0.3)]:
        obs = jnp.array([x0, v0])
        obs_p, obs_m = obs, obs
        for _ in range(30):
            obs_p = model.apply(params, obs_p, jnp.array([1.0]))
            obs_m = model.apply(params, obs_m, jnp.array([-1.0]))
        print(f"  ({x0}, {v0}): Δx = {abs(float(obs_p[0] - obs_m[0])):.2f}")
    
    # Create planner
    print("\nCompiling MPPI...")
    mppi = create_mppi(model, params)
    _ = mppi(jnp.array([1.0, 0.0]), jax.random.PRNGKey(0))
    print("Done")
    
    key = jax.random.PRNGKey(42)
    results = {}
    
    for method in ['random', 'mppi']:
        print(f"\n{method.upper()} ({N_EPISODES} episodes)...")
        use_mppi = method == 'mppi'
        
        successes, cats, misses = 0, 0, []
        start = time.time()
        
        for ep in range(N_EPISODES):
            key, subkey = jax.random.split(key)
            r = run_episode(subkey, use_mppi, mppi if use_mppi else None)
            if r['success']: successes += 1
            if r['catastrophic']: cats += 1
            misses.append(r['miss'])
            
            if (ep+1) % 100 == 0:
                elapsed = time.time() - start
                eta = elapsed / (ep+1) * (N_EPISODES - ep - 1)
                print(f"  {ep+1}/{N_EPISODES} ({elapsed:.0f}s, ETA: {eta:.0f}s)")
        
        sr = successes / N_EPISODES
        cr = cats / N_EPISODES
        avg_miss = np.mean(misses)
        std_miss = np.std(misses)
        
        # 95% CI for success rate (binomial)
        ci_sr = 1.96 * np.sqrt(sr * (1-sr) / N_EPISODES)
        
        results[method] = {
            'success_rate': sr,
            'success_ci': ci_sr,
            'catastrophic_rate': cr,
            'avg_miss': avg_miss,
            'std_miss': std_miss
        }
        
        print(f"\n  Success: {sr:.1%} ± {ci_sr:.1%}")
        print(f"  Catastrophic: {cr:.1%}")
        print(f"  Miss: {avg_miss:.3f} ± {std_miss:.3f}")
    
    # Summary
    print("\n" + "="*50)
    print("SUMMARY")
    print("="*50)
    imp = results['mppi']['success_rate'] - results['random']['success_rate']
    ci_imp = np.sqrt(results['random']['success_ci']**2 + results['mppi']['success_ci']**2)
    print(f"MPPI vs Random: {results['mppi']['success_rate']:.1%} vs {results['random']['success_rate']:.1%}")
    print(f"Improvement: {imp:+.1%} ± {ci_imp:.1%}")
    
    # Save
    output = {
        'n_episodes': N_EPISODES,
        'results': results,
        'action_influence_checkpoint': ckpt['metrics']['sensitivity']
    }
    with open('/home/ulluboz/pcp-jepa-research/results/phase2/film_mppi_500ep.json', 'w') as f:
        json.dump(output, f, indent=2)
    print(f"\nSaved to film_mppi_500ep.json")

if __name__ == '__main__':
    main()