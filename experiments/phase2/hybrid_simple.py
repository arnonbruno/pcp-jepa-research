#!/usr/bin/env python3
"""
Option B: Simplified Hybrid Dynamics Model

Physics-based prediction with learned residual corrections.
"""

import jax
import jax.numpy as jnp
import flax.linen as nn
import optax
import numpy as np
import json
import time
from dataclasses import dataclass
import sys
sys.path.insert(0, '/home/ulluboz/pcp-jepa-research')

from experiments.phase2.train_action_controllable import physics_step

# ============================================================
# CONFIG
# ============================================================

@dataclass
class Config:
    hidden_dim: int = 64
    n_train: int = 5000
    epochs: int = 100
    batch_size: int = 256
    lr: float = 1e-3
    dt: float = 0.05
    restitution: float = 0.8
    n_episodes: int = 300
    max_steps: int = 30
    horizon: int = 10
    n_samples: int = 64
    x_target: float = 2.0


# ============================================================
# PURE PHYSICS PREDICTOR
# ============================================================

def pure_physics_step(x, v, a, dt=0.05, restitution=0.8):
    """Pure physics prediction - JAX compatible."""
    v_new = v + a * dt
    x_new = x + v_new * dt
    
    # Lower bounce
    x_new = jnp.where(x_new < 0, -x_new, x_new)
    v_new = jnp.where(x_new < 0, -v_new * restitution, v_new)
    
    # Upper bounce  
    x_new = jnp.where(x_new > 4, 8 - x_new, x_new)
    v_new = jnp.where(x_new > 4, -v_new * restitution, v_new)
    
    return x_new, v_new


# ============================================================
# RESIDUAL MODEL
# ============================================================

class ResidualModel(nn.Module):
    """Learn residual correction to physics prediction."""
    hidden_dim: int = 64
    
    @nn.compact
    def __call__(self, obs, action):
        x, v = obs[0], obs[1]
        a = action[0]
        
        # Physics prediction
        x_phys, v_phys = pure_physics_step(x, v, a)
        
        # Neural residual
        features = jnp.concatenate([obs, action, jnp.array([x_phys, v_phys])])
        h = nn.Dense(self.hidden_dim)(features)
        h = nn.relu(h)
        h = nn.Dense(self.hidden_dim)(h)
        h = nn.relu(h)
        
        # Small residual correction
        residual = nn.Dense(2)(h) * 0.1
        
        # Final prediction
        x_pred = x_phys + residual[0]
        v_pred = v_phys + residual[1]
        
        # Ensure bounds
        x_pred = jnp.clip(x_pred, 0.0, 4.0)
        
        return jnp.array([x_pred, v_pred])


# ============================================================
# TRAINING
# ============================================================

def generate_data(cfg: Config, key):
    """Generate training data."""
    obs_list, action_list, next_obs_list = [], [], []
    
    for _ in range(cfg.n_train):
        key, subkey = jax.random.split(key)
        x0 = float(jax.random.uniform(subkey, (), minval=0.5, maxval=3.5))
        key, subkey = jax.random.split(key)
        v0 = float(jax.random.uniform(subkey, (), minval=-0.5, maxval=0.5))
        key, subkey = jax.random.split(key)
        a = float(jax.random.uniform(subkey, (), minval=-1.0, maxval=1.0))
        
        x1, v1 = physics_step(x0, v0, a, cfg.dt, cfg.restitution)
        
        obs_list.append([x0, v0])
        action_list.append([a])
        next_obs_list.append([x1, v1])
    
    return jnp.array(obs_list), jnp.array(action_list), jnp.array(next_obs_list)


def train(cfg: Config, key):
    """Train residual model."""
    print("="*60)
    print("TRAINING HYBRID MODEL (Physics + Residual)")
    print("="*60)
    
    model = ResidualModel(hidden_dim=cfg.hidden_dim)
    key, subkey = jax.random.split(key)
    params = model.init(subkey, jnp.zeros(2), jnp.zeros(1))
    
    print("\nGenerating data...")
    obs, actions, next_obs = generate_data(cfg, key)
    
    optimizer = optax.adam(cfg.lr)
    opt_state = optimizer.init(params)
    
    @jax.jit
    def loss_fn(params, obs, actions, next_obs):
        pred = jax.vmap(lambda o, a: model.apply(params, o, a))(obs, actions)
        return jnp.mean((pred - next_obs) ** 2)
    
    @jax.jit
    def update(params, opt_state, obs, actions, next_obs):
        loss, grads = jax.value_and_grad(loss_fn)(params, obs, actions, next_obs)
        updates, opt_state = optimizer.update(grads, opt_state)
        return optax.apply_updates(params, updates), opt_state, loss
    
    print("\nTraining...")
    for epoch in range(cfg.epochs):
        # Shuffle
        key, subkey = jax.random.split(key)
        perm = jax.random.permutation(subkey, len(obs))
        
        for i in range(0, len(obs), cfg.batch_size):
            params, opt_state, loss = update(
                params, opt_state, 
                obs[perm[i:i+cfg.batch_size]], 
                actions[perm[i:i+cfg.batch_size]], 
                next_obs[perm[i:i+cfg.batch_size]]
            )
        
        if epoch % 20 == 0:
            print(f"  Epoch {epoch}: loss={float(loss):.6f}")
    
    return model, params


# ============================================================
# EVALUATION
# ============================================================

def create_mppi(model, params, cfg: Config):
    def batch_rollout(obs, actions):
        def single(action_seq):
            current = obs
            for t in range(cfg.horizon):
                current = model.apply(params, current, action_seq[t])
            return current[0]
        return jax.vmap(single)(actions)
    
    rollout_jit = jax.jit(batch_rollout)
    
    def plan(obs, key):
        key, subkey = jax.random.split(key)
        actions = jax.random.uniform(subkey, (cfg.n_samples, cfg.horizon, 1), minval=-1.0, maxval=1.0)
        final_x = rollout_jit(obs, actions)
        costs = jnp.abs(final_x - cfg.x_target)
        weights = jax.nn.softmax(-costs)
        return jnp.sum(weights[:, None] * actions[:, 0, :], axis=0)[0]
    
    return jax.jit(plan)


def run_episode(key, cfg: Config, use_mppi=False, mppi_plan=None):
    key, subkey = jax.random.split(key)
    x = float(jax.random.uniform(subkey, (), minval=0.5, maxval=3.5))
    key, subkey = jax.random.split(key)
    v = float(jax.random.uniform(subkey, (), minval=-0.5, maxval=0.5))
    
    for _ in range(cfg.max_steps):
        obs = jnp.array([x, v])
        if use_mppi and mppi_plan:
            key, subkey = jax.random.split(key)
            a = float(mppi_plan(obs, subkey))
        else:
            key, subkey = jax.random.split(key)
            a = float(jax.random.uniform(subkey, (), minval=-1.0, maxval=1.0))
        x, v = physics_step(x, v, a)
    
    miss = abs(x - cfg.x_target)
    return {'miss': miss, 'success': miss < 0.3, 'catastrophic': miss > 0.5 or abs(v) > 2.0}


def evaluate(model, params, cfg: Config, key):
    print("\n" + "="*60)
    print("EVALUATION")
    print("="*60)
    
    # Test prediction accuracy
    print("\n1. Single-step prediction:")
    errors = []
    for _ in range(100):
        key, subkey = jax.random.split(key)
        x0 = float(jax.random.uniform(subkey, (), minval=0.5, maxval=3.5))
        key, subkey = jax.random.split(key)
        v0 = float(jax.random.uniform(subkey, (), minval=-0.5, maxval=0.5))
        key, subkey = jax.random.split(key)
        a = float(jax.random.uniform(subkey, (), minval=-1.0, maxval=1.0))
        
        x_real, v_real = physics_step(x0, v0, a)
        pred = model.apply(params, jnp.array([x0, v0]), jnp.array([a]))
        
        errors.append([abs(float(pred[0]) - x_real), abs(float(pred[1]) - v_real)])
    
    print(f"  Mean x error: {np.mean(errors, axis=0)[0]:.4f}")
    print(f"  Mean v error: {np.mean(errors, axis=0)[1]:.4f}")
    
    # Test wall behavior
    print("\n2. Wall bounce test:")
    obs = jnp.array([0.5, -0.5])
    model_traj = [float(obs[0])]
    for _ in range(10):
        obs = model.apply(params, obs, jnp.array([-0.3]))
        model_traj.append(float(obs[0]))
    
    x, v = 0.5, -0.5
    real_traj = [x]
    for _ in range(10):
        x, v = physics_step(x, v, -0.3)
        real_traj.append(x)
    
    print(f"  Model: {model_traj[:5]}")
    print(f"  Real:  {real_traj[:5]}")
    print(f"  Gap:   {abs(model_traj[4] - real_traj[4]):.3f}")
    
    # MPPI
    print("\n3. MPPI evaluation:")
    mppi = create_mppi(model, params, cfg)
    _ = mppi(jnp.array([1.0, 0.0]), jax.random.PRNGKey(0))
    
    results = {}
    for method in ['random', 'mppi']:
        print(f"\n  {method.upper()} ({cfg.n_episodes} episodes)...")
        use_mppi = method == 'mppi'
        
        successes, cats, misses = 0, 0, []
        for ep in range(cfg.n_episodes):
            key, subkey = jax.random.split(key)
            r = run_episode(subkey, cfg, use_mppi, mppi if use_mppi else None)
            if r['success']: successes += 1
            if r['catastrophic']: cats += 1
            misses.append(r['miss'])
        
        sr = successes / cfg.n_episodes
        cr = cats / cfg.n_episodes
        avg_miss = np.mean(misses)
        
        results[method] = {'success_rate': sr, 'catastrophic_rate': cr, 'avg_miss': avg_miss}
        print(f"    Success: {sr:.1%}")
        print(f"    Catastrophic: {cr:.1%}")
    
    return results


def main():
    cfg = Config()
    key = jax.random.PRNGKey(42)
    
    model, params = train(cfg, key)
    results = evaluate(model, params, cfg, key)
    
    print("\n" + "="*60)
    print("SUMMARY")
    print("="*60)
    imp = results['mppi']['success_rate'] - results['random']['success_rate']
    print(f"MPPI vs Random: {results['mppi']['success_rate']:.1%} vs {results['random']['success_rate']:.1%}")
    print(f"Improvement: {imp:+.1%}")
    
    if imp > 0:
        print("\n✅ HYBRID MODEL WORKS!")
    else:
        print("\n⚠️  Still issues")
    
    with open('/home/ulluboz/pcp-jepa-research/results/phase2/hybrid_simple_results.json', 'w') as f:
        json.dump({'results': results, 'improvement': imp}, f, indent=2)


if __name__ == '__main__':
    main()