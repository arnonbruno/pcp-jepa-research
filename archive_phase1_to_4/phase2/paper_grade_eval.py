#!/usr/bin/env python3
"""
Paper-grade evaluation:
- Multiple training seeds (3-5)
- 200-500 episodes per evaluation
- Confidence intervals for all metrics
"""

import jax
import jax.numpy as jnp
import pickle
import numpy as np
import json
import time
from dataclasses import dataclass, asdict
from typing import Dict, List, Tuple
import sys
import os
sys.path.insert(0, '/home/ulluboz/pcp-jepa-research')

from experiments.phase2.train_action_controllable import (
    ActionControllableModel, Config as TrainConfig, train, generate_dataset, physics_step
)

# ============================================================
# CONFIG
# ============================================================

@dataclass
class EvalConfig:
    # Training seeds
    n_seeds: int = 3
    train_epochs: int = 100  # Faster training for multiple runs
    
    # Evaluation
    n_episodes: int = 200
    max_steps: int = 30
    
    # MPPI
    horizon: int = 10
    n_samples: int = 64
    
    # Task
    x_target: float = 2.0
    tau_catastrophic: float = 0.5  # |x_T - x*| > 0.5
    v_max: float = 2.0  # |v| > 2.0
    tau_success: float = 0.3  # |x_T - x*| < 0.3
    
    # Physics
    dt: float = 0.05
    restitution: float = 0.8


# ============================================================
# MPPI PLANNER (JIT-compiled)
# ============================================================

def create_mppi_planner(model, params, cfg: EvalConfig):
    """Create JIT-compiled MPPI planner."""
    
    def batch_rollout(obs, actions):
        """Vectorized rollout: [N, H, 1] -> [N] final_x"""
        def single_rollout(action_seq):
            current = obs
            for t in range(cfg.horizon):
                current = model.apply(params, current, action_seq[t])
            return current[0]
        return jax.vmap(single_rollout)(actions)
    
    batch_rollout_jit = jax.jit(batch_rollout)
    
    def mppi_plan(obs, key):
        key, subkey = jax.random.split(key)
        actions = jax.random.uniform(subkey, (cfg.n_samples, cfg.horizon, 1), 
                                      minval=-1.0, maxval=1.0)
        final_x = batch_rollout_jit(obs, actions)
        costs = jnp.abs(final_x - cfg.x_target)
        weights = jax.nn.softmax(-costs)
        return jnp.sum(weights[:, None] * actions[:, 0, :], axis=0)[0]
    
    return jax.jit(mppi_plan)


# ============================================================
# EPISODE RUNNER
# ============================================================

def run_episode(key, cfg: EvalConfig, use_mppi: bool, mppi_plan=None) -> Dict:
    """Run single episode."""
    # Random initial state
    key, subkey = jax.random.split(key)
    x = float(jax.random.uniform(subkey, (), minval=0.5, maxval=3.5))
    key, subkey = jax.random.split(key)
    v = float(jax.random.uniform(subkey, (), minval=-0.5, maxval=0.5))
    
    for t in range(cfg.max_steps):
        obs = jnp.array([x, v])
        
        if use_mppi and mppi_plan is not None:
            key, subkey = jax.random.split(key)
            a = float(mppi_plan(obs, subkey))
        else:
            key, subkey = jax.random.split(key)
            a = float(jax.random.uniform(subkey, (), minval=-1.0, maxval=1.0))
        
        # Real physics
        x, v = physics_step(x, v, a, cfg.dt, cfg.restitution)
    
    # Metrics
    miss = abs(x - cfg.x_target)
    return {
        'final_x': x,
        'final_v': v,
        'miss_distance': miss,
        'is_success': miss < cfg.tau_success,
        'is_catastrophic': miss > cfg.tau_catastrophic or abs(v) > cfg.v_max
    }


# ============================================================
# MAIN EVALUATION
# ============================================================

def evaluate_seed(seed: int, cfg: EvalConfig) -> Dict:
    """Train model with seed and evaluate."""
    print(f"\n{'='*60}")
    print(f"SEED {seed}")
    print('='*60)
    
    key = jax.random.PRNGKey(seed)
    
    # Train model
    print("Training model...")
    train_cfg = TrainConfig(
        epochs=cfg.train_epochs,
        n_train=3000,
        n_val=300,
    )
    
    model = ActionControllableModel(
        latent_dim=32, hidden_dim=128, action_dim=1, obs_dim=2
    )
    
    # Initialize and train
    key, subkey = jax.random.split(key)
    params = model.init(subkey, jnp.zeros(2), jnp.zeros(1))
    
    # Generate data and train (simplified)
    data = generate_dataset(train_cfg, key)
    
    import optax
    optimizer = optax.adam(1e-3)
    opt_state = optimizer.init(params)
    
    train_obs = data['train']['obs']
    train_actions = data['train']['actions']
    train_next_obs = data['train']['next_obs']
    
    # Loss functions
    def pred_loss(params, obs, actions, next_obs):
        pred = jax.vmap(lambda o, a: model.apply(params, o, a))(obs, actions)
        return jnp.mean((pred - next_obs) ** 2)
    
    def sens_loss(params, obs, actions, epsilon=0.1):
        def compute_sens(o, a):
            def pred_x(a_):
                return model.apply(params, o, jnp.array([a_]))[0]
            return jnp.abs(jax.grad(pred_x)(a[0]))
        sensitivities = jax.vmap(compute_sens)(obs, actions)
        return jnp.mean(jax.nn.relu(epsilon - sensitivities)), jnp.mean(sensitivities)
    
    @jax.jit
    def update(params, opt_state, obs, actions, next_obs):
        l_pred = pred_loss(params, obs, actions, next_obs)
        l_sens, sens_metric = sens_loss(params, obs, actions)
        loss = l_pred + 0.5 * l_sens
        grads = jax.grad(lambda p: loss)(params)
        updates, opt_state = optimizer.update(grads, opt_state)
        return optax.apply_updates(params, updates), opt_state, loss, sens_metric
    
    # Training loop
    for epoch in range(train_cfg.epochs):
        # Shuffle
        key, subkey = jax.random.split(key)
        perm = jax.random.permutation(subkey, len(train_obs))
        
        for i in range(0, len(train_obs), 256):
            params, opt_state, loss, sens = update(
                params, opt_state,
                train_obs[perm][i:i+256],
                train_actions[perm][i:i+256],
                train_next_obs[perm][i:i+256]
            )
        
        if epoch % 20 == 0:
            print(f"  Epoch {epoch}: loss={float(loss):.4f}, sens={float(sens):.3f}")
    
    # Verify action influence
    print("\nVerifying action influence...")
    test_states = [(1.0, 0.0), (2.0, 0.5), (3.0, -0.3)]
    delta_xs = []
    for x0, v0 in test_states:
        obs = jnp.array([x0, v0])
        obs_p, obs_m = obs, obs
        for _ in range(30):
            obs_p = model.apply(params, obs_p, jnp.array([1.0]))
            obs_m = model.apply(params, obs_m, jnp.array([-1.0]))
        delta_xs.append(abs(float(obs_p[0] - obs_m[0])))
    action_influence = np.mean(delta_xs)
    print(f"  Action influence: {action_influence:.2f}")
    
    # Create MPPI planner
    mppi_plan = create_mppi_planner(model, params, cfg)
    
    # Warmup JIT
    print("\nWarming up JIT...")
    dummy_obs = jnp.array([1.0, 0.0])
    _ = mppi_plan(dummy_obs, jax.random.PRNGKey(0))
    print("  Done")
    
    # Run evaluations
    results = {'action_influence': action_influence}
    
    for method in ['random', 'mppi']:
        print(f"\n{method.upper()} ({cfg.n_episodes} episodes)...")
        use_mppi = (method == 'mppi')
        
        successes = 0
        catastrophics = 0
        misses = []
        
        for ep in range(cfg.n_episodes):
            key, subkey = jax.random.split(key)
            result = run_episode(subkey, cfg, use_mppi, mppi_plan if use_mppi else None)
            
            if result['is_success']:
                successes += 1
            if result['is_catastrophic']:
                catastrophics += 1
            misses.append(result['miss_distance'])
            
            if (ep + 1) % 50 == 0:
                print(f"  Episode {ep+1}/{cfg.n_episodes}")
        
        results[f'{method}_success_rate'] = successes / cfg.n_episodes
        results[f'{method}_catastrophic_rate'] = catastrophics / cfg.n_episodes
        results[f'{method}_avg_miss'] = float(np.mean(misses))
        results[f'{method}_std_miss'] = float(np.std(misses))
        
        print(f"  Success: {results[f'{method}_success_rate']:.1%} ± "
              f"{1.96*np.sqrt(results[f'{method}_success_rate']*(1-results[f'{method}_success_rate'])/cfg.n_episodes):.1%}")
        print(f"  Catastrophic: {results[f'{method}_catastrophic_rate']:.1%}")
        print(f"  Miss: {results[f'{method}_avg_miss']:.3f} ± {results[f'{method}_std_miss']:.3f}")
    
    return results


def compute_ci(values: List[float], confidence: float = 0.95) -> Tuple[float, float]:
    """Compute confidence interval."""
    n = len(values)
    mean = np.mean(values)
    se = np.std(values, ddof=1) / np.sqrt(n)
    z = 1.96 if confidence == 0.95 else 2.576  # 95% or 99%
    return mean, mean - z * se, mean + z * se


def main():
    cfg = EvalConfig()
    
    print("="*60)
    print("PAPER-GRADE EVALUATION")
    print("="*60)
    print(f"Seeds: {cfg.n_seeds}")
    print(f"Episodes per evaluation: {cfg.n_episodes}")
    print(f"Training epochs: {cfg.train_epochs}")
    
    # Run evaluations
    all_results = []
    for seed in range(42, 42 + cfg.n_seeds):
        results = evaluate_seed(seed, cfg)
        all_results.append(results)
    
    # Aggregate results
    print("\n" + "="*60)
    print("AGGREGATED RESULTS (mean ± 95% CI)")
    print("="*60)
    
    # Action influence
    ai_values = [r['action_influence'] for r in all_results]
    ai_mean, ai_lo, ai_hi = compute_ci(ai_values)
    print(f"\nAction Influence: {ai_mean:.2f} [{ai_lo:.2f}, {ai_hi:.2f}]")
    
    # Success rates
    for method in ['random', 'mppi']:
        sr_values = [r[f'{method}_success_rate'] for r in all_results]
        sr_mean, sr_lo, sr_hi = compute_ci(sr_values)
        
        cr_values = [r[f'{method}_catastrophic_rate'] for r in all_results]
        cr_mean, cr_lo, cr_hi = compute_ci(cr_values)
        
        miss_values = [r[f'{method}_avg_miss'] for r in all_results]
        miss_mean, miss_lo, miss_hi = compute_ci(miss_values)
        
        print(f"\n{method.upper()}:")
        print(f"  Success: {sr_mean:.1%} [{sr_lo:.1%}, {sr_hi:.1%}]")
        print(f"  Catastrophic: {cr_mean:.1%} [{cr_lo:.1%}, {cr_hi:.1%}]")
        print(f"  Miss: {miss_mean:.3f} [{miss_lo:.3f}, {miss_hi:.3f}]")
    
    # Improvement
    improvement = [r['mppi_success_rate'] - r['random_success_rate'] for r in all_results]
    imp_mean, imp_lo, imp_hi = compute_ci(improvement)
    print(f"\nIMPROVEMENT: {imp_mean:+.1%} [{imp_lo:+.1%}, {imp_hi:+.1%}]")
    
    # Save
    output = {
        'config': asdict(cfg),
        'seeds': all_results,
        'aggregated': {
            'action_influence': {'mean': ai_mean, 'ci_lo': ai_lo, 'ci_hi': ai_hi},
            'random_success': {'mean': np.mean([r['random_success_rate'] for r in all_results])},
            'mppi_success': {'mean': np.mean([r['mppi_success_rate'] for r in all_results])},
            'improvement': {'mean': imp_mean, 'ci_lo': imp_lo, 'ci_hi': imp_hi},
        }
    }
    
    output_path = '/home/ulluboz/pcp-jepa-research/results/phase2/paper_grade_results.json'
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, 'w') as f:
        json.dump(output, f, indent=2)
    print(f"\nResults saved to {output_path}")


if __name__ == '__main__':
    main()