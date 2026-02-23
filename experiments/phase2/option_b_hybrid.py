#!/usr/bin/env python3
"""
Option B: Hybrid Dynamics with Event Gate

Architecture:
- Known physics: x' = x + v*dt, v' = v + a*dt (with bounces)
- Neural residual: learns corrections to physics predictions
- Event gate: detects when physics events (bounces) occur

This guarantees physics correctness while allowing neural flexibility.
"""

import jax
import jax.numpy as jnp
import flax.linen as nn
import optax
import pickle
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
    # Model
    hidden_dim: int = 64
    latent_dim: int = 16
    
    # Training
    n_train: int = 5000
    n_val: int = 500
    epochs: int = 100
    batch_size: int = 256
    lr: float = 1e-3
    
    # Physics
    dt: float = 0.05
    restitution: float = 0.8
    x_min: float = 0.0
    x_max: float = 4.0
    
    # Loss weights
    lambda_residual: float = 1.0
    lambda_event: float = 0.5
    
    # Evaluation
    n_episodes: int = 200
    max_steps: int = 30
    horizon: int = 10
    n_samples: int = 64
    x_target: float = 2.0


# ============================================================
# HYBRID MODEL
# ============================================================

class HybridDynamicsModel(nn.Module):
    """
    Hybrid physics-neural dynamics model.
    
    Prediction = Physics + Neural Residual
    
    Physics: known equations (free flight + bounces)
    Neural: learns corrections/events
    """
    hidden_dim: int = 64
    latent_dim: int = 16
    
    @nn.compact
    def physics_step(self, obs, action):
        """Analytical physics step (no learning) - JAX compatible."""
        x, v = obs[0], obs[1]
        a = action[0]
        dt = 0.05
        restitution = 0.8
        x_min, x_max = 0.0, 4.0
        
        # Free flight
        v_new = v + a * dt
        x_new = x + v_new * dt
        
        # Bounce (JAX-compatible with jnp.where)
        # Lower bound bounce
        x_new_lower = 2 * x_min - x_new
        v_new_lower = -v_new * restitution
        
        # Upper bound bounce
        x_new_upper = 2 * x_max - x_new
        v_new_upper = -v_new * restitution
        
        # Apply bounces
        x_new = jnp.where(x_new < x_min, x_new_lower, x_new)
        v_new = jnp.where(x_new < x_min, v_new_lower, v_new)
        
        x_new = jnp.where(x_new > x_max, x_new_upper, x_new)
        v_new = jnp.where(x_new > x_max, v_new_upper, v_new)
        
        return jnp.array([x_new, v_new])
    
    @nn.compact
    def event_detector(self, obs, next_obs):
        """
        Detect if a physics event (bounce) occurred.
        
        Returns probability of event.
        """
        # Events are characterized by sudden velocity reversal
        v_before = obs[1]
        v_after = next_obs[1]
        
        # Event if velocity sign changed significantly
        velocity_change = jnp.abs(v_after + v_before)  # Reversal indicator
        
        # Neural enhancement (can learn subtle patterns)
        features = jnp.concatenate([obs, jnp.array([velocity_change])])
        h = nn.Dense(self.hidden_dim)(features)
        h = nn.relu(h)
        event_prob = nn.sigmoid(nn.Dense(1)(h))
        
        return event_prob[0]
    
    @nn.compact
    def residual_net(self, obs, action, physics_pred):
        """
        Learn residual correction to physics prediction.
        
        Residual is small by construction (physics is good).
        """
        # Input: current state + action + physics prediction
        features = jnp.concatenate([obs, action, physics_pred])
        
        h = nn.Dense(self.hidden_dim)(features)
        h = nn.relu(h)
        h = nn.Dense(self.hidden_dim)(h)
        h = nn.relu(h)
        
        # Small residual
        residual = nn.Dense(2)(h) * 0.1  # Scale down residuals
        
        return residual
    
    @nn.compact
    def __call__(self, obs, action):
        """Full forward pass: physics + residual."""
        # Physics prediction
        physics_pred = self.physics_step(obs, action)
        
        # Neural residual
        residual = self.residual_net(obs, action, physics_pred)
        
        # Combined prediction
        pred = physics_pred + residual
        
        # Ensure constraints (clamp to bounds)
        pred = pred.at[0].set(jnp.clip(pred[0], 0.0, 4.0))
        
        return pred
    
    @nn.compact
    def predict_with_event(self, obs, action):
        """Predict with event detection."""
        pred = self(obs, action)
        event_prob = self.event_detector(obs, pred)
        return pred, event_prob


# ============================================================
# DATA GENERATION
# ============================================================

def generate_data(cfg: Config, key):
    """Generate training data with physics events labeled."""
    data = {'train': [], 'val': []}
    
    for split, n in [('train', cfg.n_train), ('val', cfg.n_val)]:
        observations = []
        actions = []
        next_observations = []
        event_labels = []  # 1 if bounce occurred, 0 otherwise
        
        for _ in range(n):
            key, subkey = jax.random.split(key)
            x0 = float(jax.random.uniform(subkey, (), minval=0.5, maxval=3.5))
            key, subkey = jax.random.split(key)
            v0 = float(jax.random.uniform(subkey, (), minval=-0.5, maxval=0.5))
            key, subkey = jax.random.split(key)
            a = float(jax.random.uniform(subkey, (), minval=-1.0, maxval=1.0))
            
            x1, v1 = physics_step(x0, v0, a, cfg.dt, cfg.restitution)
            
            # Detect event (bounce)
            event = 1.0 if (x0 + (v0 + a * cfg.dt) * cfg.dt < cfg.x_min or 
                           x0 + (v0 + a * cfg.dt) * cfg.dt > cfg.x_max) else 0.0
            
            observations.append([x0, v0])
            actions.append([a])
            next_observations.append([x1, v1])
            event_labels.append([event])
        
        data[split] = {
            'obs': jnp.array(observations),
            'actions': jnp.array(actions),
            'next_obs': jnp.array(next_observations),
            'events': jnp.array(event_labels)
        }
    
    return data


# ============================================================
# TRAINING
# ============================================================

def train_hybrid(cfg: Config, key):
    """Train hybrid model."""
    print("="*60)
    print("TRAINING HYBRID DYNAMICS MODEL")
    print("="*60)
    
    model = HybridDynamicsModel(hidden_dim=cfg.hidden_dim)
    
    # Initialize
    key, subkey = jax.random.split(key)
    params = model.init(subkey, jnp.zeros(2), jnp.zeros(1))
    
    # Data
    print("\nGenerating data...")
    data = generate_data(cfg, key)
    
    # Optimizer
    optimizer = optax.adam(cfg.lr)
    opt_state = optimizer.init(params)
    
    # Loss
    def loss_fn(params, obs, actions, next_obs, events):
        pred, event_prob = jax.vmap(lambda o, a: model.apply(params, o, a, method=model.predict_with_event))(obs, actions)
        
        # Prediction loss
        pred_loss = jnp.mean((pred - next_obs) ** 2)
        
        # Event detection loss
        event_loss = jnp.mean(jax.vmap(lambda p, e: -(e * jnp.log(p + 1e-6) + (1-e) * jnp.log(1-p + 1e-6)))(event_prob, events[:, 0]))
        
        return pred_loss + cfg.lambda_event * event_loss, (pred_loss, event_loss)
    
    @jax.jit
    def update(params, opt_state, obs, actions, next_obs, events):
        (loss, (pred_l, event_l)), grads = jax.value_and_grad(loss_fn, has_aux=True)(params, obs, actions, next_obs, events)
        updates, opt_state = optimizer.update(grads, opt_state)
        params = optax.apply_updates(params, updates)
        return params, opt_state, loss, pred_l, event_l
    
    # Training loop
    print("\nTraining...")
    train_obs = data['train']['obs']
    train_actions = data['train']['actions']
    train_next_obs = data['train']['next_obs']
    train_events = data['train']['events']
    
    for epoch in range(cfg.epochs):
        # Shuffle
        key, subkey = jax.random.split(key)
        perm = jax.random.permutation(subkey, len(train_obs))
        
        epoch_losses = []
        for i in range(0, len(train_obs), cfg.batch_size):
            batch_obs = train_obs[perm[i:i+cfg.batch_size]]
            batch_actions = train_actions[perm[i:i+cfg.batch_size]]
            batch_next_obs = train_next_obs[perm[i:i+cfg.batch_size]]
            batch_events = train_events[perm[i:i+cfg.batch_size]]
            
            params, opt_state, loss, pred_l, event_l = update(
                params, opt_state, batch_obs, batch_actions, batch_next_obs, batch_events
            )
            epoch_losses.append((float(loss), float(pred_l), float(event_l)))
        
        if epoch % 10 == 0:
            avg_loss = np.mean([l[0] for l in epoch_losses])
            avg_pred = np.mean([l[1] for l in epoch_losses])
            avg_event = np.mean([l[2] for l in epoch_losses])
            print(f"  Epoch {epoch}: loss={avg_loss:.4f} (pred={avg_pred:.4f}, event={avg_event:.4f})")
    
    return model, params


# ============================================================
# EVALUATION
# ============================================================

def create_mppi(model, params, cfg: Config):
    """Create MPPI planner with hybrid model."""
    
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
    """Evaluate hybrid model."""
    print("\n" + "="*60)
    print("EVALUATION")
    print("="*60)
    
    # Test prediction accuracy
    print("\n1. Prediction accuracy:")
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
    
    errors = np.array(errors)
    print(f"  Mean x error: {np.mean(errors[:,0]):.4f}")
    print(f"  Mean v error: {np.mean(errors[:,1]):.4f}")
    
    # Test wall behavior
    print("\n2. Wall behavior test:")
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
    
    print(f"  Model:  {model_traj[:5]}")
    print(f"  Real:   {real_traj[:5]}")
    print(f"  Gap:    {abs(model_traj[4] - real_traj[4]):.3f}")
    
    # MPPI planning
    print("\n3. MPPI planning test:")
    mppi = create_mppi(model, params, cfg)
    _ = mppi(jnp.array([1.0, 0.0]), jax.random.PRNGKey(0))  # Warmup
    
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
        print(f"    Miss: {avg_miss:.3f}")
    
    return results


# ============================================================
# MAIN
# ============================================================

def main():
    cfg = Config()
    key = jax.random.PRNGKey(42)
    
    # Train
    model, params = train_hybrid(cfg, key)
    
    # Evaluate
    results = evaluate(model, params, cfg, key)
    
    # Summary
    print("\n" + "="*60)
    print("SUMMARY")
    print("="*60)
    imp = results['mppi']['success_rate'] - results['random']['success_rate']
    print(f"MPPI vs Random: {results['mppi']['success_rate']:.1%} vs {results['random']['success_rate']:.1%}")
    print(f"Improvement: {imp:+.1%}")
    
    if imp > 0:
        print("\n✅ HYBRID MODEL WORKS! Physics + Residuals = Correct Planning")
    else:
        print("\n⚠️  Still issues — may need more training or architecture tuning")
    
    # Save
    with open('/home/ulluboz/pcp-jepa-research/results/phase2/hybrid_results.json', 'w') as f:
        json.dump({'config': cfg.__dict__, 'results': results, 'improvement': imp}, f, indent=2)
    print("\nSaved to hybrid_results.json")


if __name__ == '__main__':
    main()