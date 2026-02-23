#!/usr/bin/env python3
"""
MPPI Planning with Action-Controllable World Model

Key fixes:
1. Load saved checkpoint (not random model)
2. Correct catastrophic metric: |x_T - x*| > τ (not divergence > 2)
3. Small MPPI settings: N=128, H=15, iters=2
4. No gradient tracking in planner
"""

import jax
import jax.numpy as jnp
import pickle
import numpy as np
import json
import time
from dataclasses import dataclass
from typing import Tuple, Dict
import sys
sys.path.insert(0, '/home/ulluboz/pcp-jepa-research')

from experiments.phase2.train_action_controllable import (
    ActionControllableModel, Config, physics_step
)

# ============================================================
# CONFIG
# ============================================================

@dataclass
class MPPIConfig:
    # Model
    latent_dim: int = 32
    hidden_dim: int = 128
    action_dim: int = 1
    obs_dim: int = 2
    
    # MPPI
    horizon: int = 15
    n_samples: int = 128
    n_iters: int = 2
    temperature: float = 1.0
    
    # Task
    x_target: float = 2.0
    tau_catastrophic: float = 0.5  # |x_T - x*| > 0.5 is catastrophic
    v_max: float = 2.0  # |v| > 2.0 is catastrophic
    
    # Episodes
    n_episodes: int = 20
    max_steps: int = 50
    
    # Physics
    dt: float = 0.05
    restitution: float = 0.8


# ============================================================
# MPPI PLANNER
# ============================================================

class MPPIPlanner:
    """Model Predictive Path Integral Control."""
    
    def __init__(self, model, params, cfg: MPPIConfig):
        self.model = model
        self.params = params
        self.cfg = cfg
    
    def rollout(self, obs: jnp.ndarray, actions: jnp.ndarray) -> Tuple[jnp.ndarray, jnp.ndarray]:
        """
        Roll out trajectory with given action sequence.
        
        Args:
            obs: Initial observation [obs_dim]
            actions: Action sequence [H, action_dim]
        
        Returns:
            states: State sequence [H+1, obs_dim]
            final_x: Final x position
        """
        H = actions.shape[0]
        states = [obs]
        
        current_obs = obs
        for t in range(H):
            current_obs = self.model.apply(self.params, current_obs, actions[t])
            states.append(current_obs)
        
        return jnp.stack(states), current_obs[0]  # Return final x
    
    def compute_cost(self, final_x: jnp.ndarray, states: jnp.ndarray) -> jnp.ndarray:
        """
        Compute cost for trajectory.
        
        Cost = |x_final - x_target| + penalty for leaving bounds
        """
        # Distance to target
        cost = jnp.abs(final_x - self.cfg.x_target)
        
        # Penalize leaving bounds
        x_positions = states[:, 0]
        out_of_bounds = jnp.sum(jnp.maximum(0, -x_positions) + jnp.maximum(0, x_positions - 4.0))
        cost += 10.0 * out_of_bounds
        
        return cost
    
    def plan(self, obs: jnp.ndarray, key: jax.random.PRNGKey) -> jnp.ndarray:
        """
        Plan action sequence using MPPI.
        
        Returns:
            Best first action
        """
        H = self.cfg.horizon
        N = self.cfg.n_samples
        
        # Initialize mean action sequence
        action_mean = jnp.zeros((H, self.cfg.action_dim))
        
        for _ in range(self.cfg.n_iters):
            # Sample action sequences
            key, subkey = jax.random.split(key)
            noise = jax.random.normal(subkey, (N, H, self.cfg.action_dim))
            actions = action_mean + noise * 0.5
            actions = jnp.clip(actions, -1.0, 1.0)
            
            # Compute costs for all trajectories
            costs = []
            for i in range(N):
                states, final_x = self.rollout(obs, actions[i])
                cost = self.compute_cost(final_x, states)
                costs.append(cost)
            
            costs = jnp.array(costs)
            
            # Softmax weights
            weights = jax.nn.softmax(-costs / self.cfg.temperature)
            
            # Update mean
            action_mean = jnp.sum(weights[:, None, None] * actions, axis=0)
        
        return action_mean[0]  # Return first action


# ============================================================
# EVALUATION
# ============================================================

def run_episode(model, params, cfg: MPPIConfig, key: jax.random.PRNGKey, 
                use_mppi: bool = True) -> Dict:
    """
    Run one episode with MPPI or random actions.
    
    Returns:
        Dictionary with episode metrics
    """
    # Initialize
    key, subkey = jax.random.split(key)
    x = float(jax.random.uniform(subkey, (), minval=0.5, maxval=3.5))
    key, subkey = jax.random.split(key)
    v = float(jax.random.uniform(subkey, (), minval=-0.5, maxval=0.5))
    
    planner = MPPIPlanner(model, params, cfg) if use_mppi else None
    
    trajectory = [(x, v)]
    actions_taken = []
    
    for t in range(cfg.max_steps):
        obs = jnp.array([x, v])
        
        if use_mppi:
            key, subkey = jax.random.split(key)
            action = planner.plan(obs, subkey)
        else:
            key, subkey = jax.random.split(key)
            action = jax.random.uniform(subkey, (1,), minval=-1.0, maxval=1.0)
        
        a = float(action[0] if hasattr(action, '__len__') else action)
        actions_taken.append(a)
        
        # Apply action in REAL physics (not model)
        x, v = physics_step(x, v, a, cfg.dt, cfg.restitution)
        trajectory.append((x, v))
    
    # Compute metrics
    final_x, final_v = trajectory[-1]
    miss_distance = abs(final_x - cfg.x_target)
    
    # Catastrophic: terminal miss or velocity blowup
    is_catastrophic = miss_distance > cfg.tau_catastrophic or abs(final_v) > cfg.v_max
    
    # Success: close to target
    is_success = miss_distance < 0.3
    
    return {
        'trajectory': trajectory,
        'actions': actions_taken,
        'final_x': final_x,
        'final_v': final_v,
        'miss_distance': miss_distance,
        'is_catastrophic': is_catastrophic,
        'is_success': is_success
    }


def evaluate(model, params, cfg: MPPIConfig, key: jax.random.PRNGKey):
    """Run full evaluation comparing MPPI vs Random."""
    
    print("=" * 60)
    print("MPPI EVALUATION")
    print("=" * 60)
    
    results = {'mppi': [], 'random': []}
    
    for method in ['random', 'mppi']:
        print(f"\n{method.upper()} ({cfg.n_episodes} episodes)...")
        
        use_mppi = (method == 'mppi')
        successes = 0
        catastrophics = 0
        miss_distances = []
        
        for ep in range(cfg.n_episodes):
            key, subkey = jax.random.split(key)
            result = run_episode(model, params, cfg, subkey, use_mppi)
            results[method].append(result)
            
            if result['is_success']:
                successes += 1
            if result['is_catastrophic']:
                catastrophics += 1
            miss_distances.append(result['miss_distance'])
            
            if (ep + 1) % 5 == 0:
                print(f"  Episode {ep+1}: success={successes}, cat={catastrophics}")
        
        # Summary
        results[f'{method}_success_rate'] = successes / cfg.n_episodes
        results[f'{method}_catastrophic_rate'] = catastrophics / cfg.n_episodes
        results[f'{method}_avg_miss'] = np.mean(miss_distances)
        
        print(f"\n  Success rate: {results[f'{method}_success_rate']:.1%}")
        print(f"  Catastrophic rate: {results[f'{method}_catastrophic_rate']:.1%}")
        print(f"  Avg miss distance: {results[f'{method}_avg_miss']:.3f}")
    
    # Compare
    print("\n" + "=" * 60)
    print("COMPARISON")
    print("=" * 60)
    print(f"  MPPI success: {results['mppi_success_rate']:.1%} vs Random: {results['random_success_rate']:.1%}")
    print(f"  MPPI catastrophic: {results['mppi_catastrophic_rate']:.1%} vs Random: {results['random_catastrophic_rate']:.1%}")
    
    improvement = results['mppi_success_rate'] - results['random_success_rate']
    reduction = (results['random_catastrophic_rate'] - results['mppi_catastrophic_rate']) / max(results['random_catastrophic_rate'], 0.01)
    
    print(f"\n  Improvement: {improvement:+.1%} success rate")
    print(f"  Catastrophic reduction: {reduction:+.1%}")
    
    return results


# ============================================================
# MAIN
# ============================================================

def main():
    # Config
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
    print(f"Loaded model from epoch {checkpoint['epoch']}")
    print(f"Training metrics: pred_loss={checkpoint['metrics']['loss_pred']:.4f}, "
          f"sensitivity={checkpoint['metrics']['sensitivity']:.3f}")
    
    # Evaluate
    start_time = time.time()
    results = evaluate(model, params, cfg, key)
    elapsed = time.time() - start_time
    
    print(f"\nEvaluation took {elapsed:.1f}s")
    
    # Save results
    output = {
        'config': {
            'horizon': cfg.horizon,
            'n_samples': cfg.n_samples,
            'n_iters': cfg.n_iters,
            'x_target': cfg.x_target,
            'tau_catastrophic': cfg.tau_catastrophic,
            'n_episodes': cfg.n_episodes,
        },
        'results': {
            'mppi_success_rate': results['mppi_success_rate'],
            'mppi_catastrophic_rate': results['mppi_catastrophic_rate'],
            'mppi_avg_miss': results['mppi_avg_miss'],
            'random_success_rate': results['random_success_rate'],
            'random_catastrophic_rate': results['random_catastrophic_rate'],
            'random_avg_miss': results['random_avg_miss'],
        },
        'elapsed_s': elapsed
    }
    
    output_path = '/home/ulluboz/pcp-jepa-research/results/phase2/mppi_evaluation.json'
    with open(output_path, 'w') as f:
        json.dump(output, f, indent=2)
    print(f"\nResults saved to {output_path}")


if __name__ == '__main__':
    main()
