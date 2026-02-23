#!/usr/bin/env python3
"""
Fast MPPI with vectorized rollouts.

Key: Use jax.vmap for parallel trajectory evaluation.
"""

import jax
import jax.numpy as jnp
import pickle
import numpy as np
import json
import time
from dataclasses import dataclass
import sys
sys.path.insert(0, '/home/ulluboz/pcp-jepa-research')

from experiments.phase2.train_action_controllable import (
    ActionControllableModel, Config, physics_step
)


@dataclass
class MPPIConfig:
    latent_dim: int = 32
    hidden_dim: int = 128
    action_dim: int = 1
    obs_dim: int = 2
    
    horizon: int = 10
    n_samples: int = 64
    temperature: float = 1.0
    
    x_target: float = 2.0
    tau_catastrophic: float = 0.5
    v_max: float = 2.0
    
    n_episodes: int = 20
    max_steps: int = 30
    
    dt: float = 0.05
    restitution: float = 0.8


def batch_rollout(model, params, obs, actions):
    """
    Vectorized rollout over N trajectories.
    
    Args:
        obs: [obs_dim]
        actions: [N, H, action_dim]
    
    Returns:
        final_x: [N] - final x positions
        costs: [N] - trajectory costs
    """
    N, H, _ = actions.shape
    
    def rollout_single(action_seq):
        """Rollout one trajectory."""
        current_obs = obs
        for t in range(H):
            current_obs = model.apply(params, current_obs, action_seq[t])
        return current_obs[0]  # Final x
    
    # Vectorize over N trajectories
    final_x = jax.vmap(rollout_single)(actions)
    
    # Cost: distance to target
    costs = jnp.abs(final_x - 2.0)
    
    return final_x, costs


def mppi_plan(model, params, obs, key, cfg: MPPIConfig):
    """MPPI with vectorized rollouts."""
    N, H = cfg.n_samples, cfg.horizon
    
    # Sample actions
    key, subkey = jax.random.split(key)
    actions = jax.random.uniform(subkey, (N, H, 1), minval=-1.0, maxval=1.0)
    
    # Evaluate all trajectories in parallel
    final_x, costs = batch_rollout(model, params, obs, actions)
    
    # Softmax weights
    weights = jax.nn.softmax(-costs / cfg.temperature)
    
    # Weighted average of first actions
    first_actions = actions[:, 0, :]
    best_action = jnp.sum(weights[:, None] * first_actions, axis=0)
    
    return best_action[0]


def run_episode(model, params, key, cfg: MPPIConfig, use_mppi: bool):
    """Run one episode."""
    # Random initial state
    key, subkey = jax.random.split(key)
    x = float(jax.random.uniform(subkey, (), minval=0.5, maxval=3.5))
    key, subkey = jax.random.split(key)
    v = float(jax.random.uniform(subkey, (), minval=-0.5, maxval=0.5))
    
    for t in range(cfg.max_steps):
        obs = jnp.array([x, v])
        
        if use_mppi:
            key, subkey = jax.random.split(key)
            a = float(mppi_plan(model, params, obs, subkey, cfg))
        else:
            key, subkey = jax.random.split(key)
            a = float(jax.random.uniform(subkey, (), minval=-1.0, maxval=1.0))
        
        # REAL physics
        x, v = physics_step(x, v, a, cfg.dt, cfg.restitution)
    
    # Metrics
    miss = abs(x - cfg.x_target)
    is_catastrophic = miss > cfg.tau_catastrophic or abs(v) > cfg.v_max
    is_success = miss < 0.3
    
    return {
        'final_x': x,
        'final_v': v,
        'miss_distance': miss,
        'is_catastrophic': is_catastrophic,
        'is_success': is_success
    }


def main():
    cfg = MPPIConfig()
    key = jax.random.PRNGKey(42)
    
    # Load model
    print("Loading trained model...")
    model = ActionControllableModel(
        latent_dim=cfg.latent_dim,
        hidden_dim=cfg.hidden_dim,
        action_dim=cfg.action_dim,
        obs_dim=cfg.obs_dim
    )
    
    checkpoint_path = '/home/ulluboz/pcp-jepa-research/checkpoints/model_best.pkl'
    with open(checkpoint_path, 'rb') as f:
        checkpoint = pickle.load(f)
    params = checkpoint['params']
    print(f"Loaded model (sensitivity={checkpoint['metrics']['sensitivity']:.3f})")
    
    # Evaluate
    results = {}
    
    for method in ['random', 'mppi']:
        print(f"\n{method.upper()} ({cfg.n_episodes} episodes)...")
        
        use_mppi = (method == 'mppi')
        successes = 0
        catastrophics = 0
        misses = []
        
        for ep in range(cfg.n_episodes):
            key, subkey = jax.random.split(key)
            result = run_episode(model, params, subkey, cfg, use_mppi)
            
            if result['is_success']:
                successes += 1
            if result['is_catastrophic']:
                catastrophics += 1
            misses.append(result['miss_distance'])
            
            if (ep + 1) % 5 == 0:
                print(f"  Episode {ep+1}/{cfg.n_episodes}")
        
        results[f'{method}_success_rate'] = successes / cfg.n_episodes
        results[f'{method}_catastrophic_rate'] = catastrophics / cfg.n_episodes
        results[f'{method}_avg_miss'] = float(np.mean(misses))
        
        print(f"  Success: {results[f'{method}_success_rate']:.1%}")
        print(f"  Catastrophic: {results[f'{method}_catastrophic_rate']:.1%}")
        print(f"  Avg miss: {results[f'{method}_avg_miss']:.3f}")
    
    # Compare
    print("\n" + "=" * 50)
    print("COMPARISON")
    print("=" * 50)
    improvement = results['mppi_success_rate'] - results['random_success_rate']
    print(f"MPPI vs Random: {results['mppi_success_rate']:.1%} vs {results['random_success_rate']:.1%}")
    print(f"Improvement: {improvement:+.1%}")
    
    # Save
    output = {
        'config': {'horizon': cfg.horizon, 'n_samples': cfg.n_samples, 'n_episodes': cfg.n_episodes},
        'results': results
    }
    
    output_path = '/home/ulluboz/pcp-jepa-research/results/phase2/mppi_fast_results.json'
    with open(output_path, 'w') as f:
        json.dump(output, f, indent=2)
    print(f"\nResults saved to {output_path}")


if __name__ == '__main__':
    main()
