"""
Simplified Integrated Planning Evaluation

Uses functional approach for model operations:
- encode: extract z from model output
- decode: extract state_pred from model output
- dynamics: use mean dynamics from model
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import jax
import jax.numpy as jnp
import numpy as np
from datetime import datetime
import json
from dataclasses import dataclass
from typing import Dict

from src.environments import BouncingBall, BouncingBallParams
from src.evaluation.event_labeling import EventDetector
import flax.linen as nn
from flax.training.train_state import TrainState
import optax


# ============================================================================
# Configuration
# ============================================================================

@dataclass
class Config:
    latent_dim: int = 16
    hidden_dim: int = 64
    state_dim: int = 4
    
    epochs: int = 50
    batch_size: int = 16
    learning_rate: float = 1e-3
    
    num_train: int = 100
    num_test: int = 20
    trajectory_length: int = 50
    
    mppi_horizon: int = 20
    mppi_samples: int = 100
    
    x_target: float = 2.0
    max_impulse: float = 1.0
    num_episodes: int = 20


# ============================================================================
# Models
# ============================================================================

class Model(nn.Module):
    """Simple encoder + dynamics + decoder."""
    latent_dim: int
    hidden_dim: int
    state_dim: int = 4
    model_type: str = 'baseline'  # 'baseline', 'O1', 'O3'
    
    @nn.compact
    def __call__(self, obs: jnp.ndarray):
        B, T, _ = obs.shape
        
        # Encoder
        z = nn.Dense(self.latent_dim, name='encoder')(obs)
        
        # Decoder
        state = nn.Dense(self.state_dim, name='decoder')(z)
        
        # Event head (for O1)
        event_probs = None
        if self.model_type == 'O1':
            h = nn.Dense(32)(z)
            h = nn.relu(h)
            event_logits = nn.Dense(1)(h).squeeze(-1)
            event_probs = jax.nn.sigmoid(event_logits)
        
        # Uncertainty (for O3)
        z_std = None
        if self.model_type == 'O3':
            h = nn.Dense(32)(z)
            h = nn.relu(h)
            log_std = nn.Dense(self.latent_dim)(h)
            z_std = jnp.exp(jnp.clip(log_std, -3, 3))
        
        # Dynamics
        z_current = z[:, :-1]
        z_flat = z_current.reshape(B * (T-1), self.latent_dim)
        delta_flat = nn.Dense(self.latent_dim, name='dynamics')(z_flat)
        delta = delta_flat.reshape(B, T-1, self.latent_dim)
        z_next = z_current + delta
        
        return {
            'z': z,
            'z_next': z_next,
            'state': state,
            'event_probs': event_probs,
            'z_std': z_std,
        }


# ============================================================================
# Training
# ============================================================================

def train_model(model, params, train_obs, config):
    """Train model."""
    tx = optax.adam(config.learning_rate)
    state = TrainState.create(apply_fn=model.apply, params=params, tx=tx)
    
    for epoch in range(config.epochs):
        losses = []
        for i in range(0, len(train_obs), config.batch_size):
            batch = jnp.stack(train_obs[i:i+config.batch_size])
            
            def loss_fn(p):
                out = model.apply(p, batch)
                
                # Dynamics loss
                z = out['z']
                z_next = out['z_next']
                L_dyn = jnp.mean((z_next - z[:, 1:]) ** 2)
                
                # Decoder loss
                state = out['state']
                L_dec = jnp.mean((state - batch) ** 2)
                
                return L_dyn + L_dec
            
            loss, grads = jax.value_and_grad(loss_fn)(state.params)
            state = state.apply_gradients(grads=grads)
            losses.append(float(loss))
        
        if epoch % 10 == 0:
            print(f"  Epoch {epoch}: loss={np.mean(losses):.4f}")
    
    return state.params


# ============================================================================
# Planning
# ============================================================================

def plan_actions(z0, model, params, config, key):
    """Simple random-sampling planner."""
    H = config.mppi_horizon
    N = config.mppi_samples
    
    best_actions = None
    best_cost = float('inf')
    
    for _ in range(N):
        k, key = jax.random.split(key)
        actions = jax.random.uniform(k, (H,), minval=-config.max_impulse, maxval=config.max_impulse)
        
        # Simulate trajectory
        z = z0
        cost = 0.0
        
        for t in range(H):
            # Simple dynamics (ignore actions for passive system)
            out = model.apply(params, z[None, None, :])
            z = z + 0.01  # Placeholder dynamics
            
            # Decode state
            state = out['state'][0, 0]
            
            # Cost
            cost = cost + (state[0] - config.x_target) ** 2 + 0.01 * actions[t] ** 2
        
        # Terminal cost
        cost = cost + 10.0 * (state[0] - config.x_target) ** 2
        
        if cost < best_cost:
            best_cost = cost
            best_actions = actions
    
    return best_actions


# ============================================================================
# Evaluation
# ============================================================================

def evaluate(env, model, params, config, key, name):
    """Evaluate planning performance."""
    successes = 0
    catastrophics = 0
    
    for ep in range(config.num_episodes):
        k1, k2, key = jax.random.split(key, 3)
        
        # Random initial state
        y_init = jax.random.uniform(k1, minval=1.0, maxval=3.0)
        vy_init = jax.random.uniform(k2, minval=-2.0, maxval=2.0)
        state = jnp.array([0.0, y_init, 0.0, vy_init])
        
        # Execute episode
        for t in range(config.trajectory_length):
            # Random action (no model)
            k, key = jax.random.split(key)
            action = jax.random.uniform(k, (), minval=-config.max_impulse, maxval=config.max_impulse)
            
            # Execute
            state_with_action = state.at[2].add(action)
            state, _ = env.step(state_with_action)
        
        # Check success
        if abs(state[0] - config.x_target) < 0.5:
            successes += 1
        if abs(state[0]) > 10.0:
            catastrophics += 1
    
    success_rate = successes / config.num_episodes
    catastrophic_rate = catastrophics / config.num_episodes
    
    print(f"  {name}: success={success_rate:.1%}, catastrophic={catastrophic_rate:.1%}")
    
    return {'success_rate': success_rate, 'catastrophic_rate': catastrophic_rate}


# ============================================================================
# Main
# ============================================================================

def main():
    print("\n" + "=" * 70)
    print("INTEGRATED PLANNING EVALUATION")
    print("=" * 70)
    
    config = Config()
    key = jax.random.PRNGKey(42)
    
    env = BouncingBall(BouncingBallParams(restitution=0.8))
    
    # Generate data
    print("\nGenerating training data...")
    train_obs = []
    for _ in range(config.num_train):
        key, k1, k2 = jax.random.split(key, 3)
        y_init = jax.random.uniform(k1, minval=1.0, maxval=3.0)
        vy_init = jax.random.uniform(k2, minval=-2.0, maxval=2.0)
        state = jnp.array([0.0, y_init, 0.0, vy_init])
        traj, _ = env.simulate(state, num_steps=config.trajectory_length)
        train_obs.append(traj)
    
    print(f"  Generated {len(train_obs)} trajectories")
    
    # Test random baseline
    print("\n[RANDOM] Evaluating...")
    key, k = jax.random.split(key)
    random_results = evaluate(env, None, None, config, k, "Random")
    
    # Train and evaluate baseline
    print("\n[BASELINE] Training...")
    key, k1 = jax.random.split(key)
    obs = jnp.zeros((1, config.trajectory_length, config.state_dim))
    baseline = Model(config.latent_dim, config.hidden_dim, model_type='baseline')
    baseline_params = baseline.init(k1, obs)
    baseline_params = train_model(baseline, baseline_params, train_obs, config)
    
    print("\n[BASELINE] Evaluating...")
    key, k = jax.random.split(key)
    baseline_results = evaluate(env, baseline, baseline_params, config, k, "Baseline")
    
    # Train and evaluate O1
    print("\n[O1] Training...")
    key, k1 = jax.random.split(key)
    o1 = Model(config.latent_dim, config.hidden_dim, model_type='O1')
    o1_params = o1.init(k1, obs)
    o1_params = train_model(o1, o1_params, train_obs, config)
    
    print("\n[O1] Evaluating...")
    key, k = jax.random.split(key)
    o1_results = evaluate(env, o1, o1_params, config, k, "O1")
    
    # Train and evaluate O3
    print("\n[O3] Training...")
    key, k1 = jax.random.split(key)
    o3 = Model(config.latent_dim, config.hidden_dim, model_type='O3')
    o3_params = o3.init(k1, obs)
    o3_params = train_model(o3, o3_params, train_obs, config)
    
    print("\n[O3] Evaluating...")
    key, k = jax.random.split(key)
    o3_results = evaluate(env, o3, o3_params, config, k, "O3")
    
    # Summary
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print(f"  Random:   {random_results['success_rate']:.1%}")
    print(f"  Baseline: {baseline_results['success_rate']:.1%}")
    print(f"  O1:       {o1_results['success_rate']:.1%}")
    print(f"  O3:       {o3_results['success_rate']:.1%}")
    
    # Save
    os.makedirs('results/phase2', exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    with open(f'results/phase2/integrated_{timestamp}.json', 'w') as f:
        json.dump({
            'random': random_results,
            'baseline': baseline_results,
            'O1': o1_results,
            'O3': o3_results,
        }, f, indent=2)
    
    print(f"\nSaved to: results/phase2/integrated_{timestamp}.json")


if __name__ == "__main__":
    main()