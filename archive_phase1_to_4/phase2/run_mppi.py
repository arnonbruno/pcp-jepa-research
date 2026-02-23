"""
MPPI with Learned Dynamics - Fixed Version

Uses state directly for cost computation (no latent decoding needed).
Simpler and more robust approach.
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
from typing import Optional

from src.environments import BouncingBall, BouncingBallParams
import flax.linen as nn
from flax.training.train_state import TrainState
import optax


# ============================================================================
# Config
# ============================================================================

@dataclass
class Config:
    latent_dim: int = 16
    hidden_dim: int = 64
    state_dim: int = 4
    
    epochs: int = 20
    batch_size: int = 32
    learning_rate: float = 1e-3
    num_train: int = 50
    
    mppi_horizon: int = 20
    mppi_samples: int = 256
    mppi_iterations: int = 3
    mppi_temperature: float = 1.0
    mppi_sigma: float = 0.3
    
    trajectory_length: int = 50
    x_target: float = 2.0
    max_impulse: float = 1.0
    num_episodes: int = 20
    
    o3_beta: float = 0.3


# ============================================================================
# Simple Direct Model (no latent issues)
# ============================================================================

class DirectModel(nn.Module):
    """Simple model: predict state directly."""
    hidden_dim: int
    
    @nn.compact
    def __call__(self, state: jnp.ndarray, action: Optional[float] = None):
        """Predict next state from current state."""
        # Simple MLP
        h = nn.Dense(self.hidden_dim)(state)
        h = nn.relu(h)
        h = nn.Dense(self.hidden_dim)(h)
        h = nn.relu(h)
        
        # Predict delta
        delta = nn.Dense(4)(h)
        
        return state + delta * 0.1  # Small delta


# ============================================================================
# MPPI Planner (State-based)
# ============================================================================

class MPPIPlanner:
    """MPPI planner using state-based model."""
    
    def __init__(self, model, params, config, model_type='baseline', beta=0.0):
        self.model = model
        self.params = params
        self.config = config
        self.model_type = model_type
        self.beta = beta
        self.mean_actions = np.zeros(config.mppi_horizon)
    
    def rollout(self, state0: jnp.ndarray, actions: np.ndarray, key=None) -> float:
        """Rollout and return total cost."""
        H = len(actions)
        state = state0
        total_cost = 0.0
        
        for t in range(H):
            a = float(actions[t])
            
            # Apply action (horizontal impulse)
            state = state.at[2].add(a * 0.1)
            
            # Predict next state
            state = self.model.apply(self.params, state)
            
            # Cost
            x = float(state[0])
            total_cost = total_cost + (x - self.config.x_target) ** 2 + 0.01 * a ** 2
        
        # Terminal cost
        x_final = float(state[0])
        total_cost = total_cost + 10.0 * (x_final - self.config.x_target) ** 2
        
        return total_cost
    
    def plan(self, state0: jnp.ndarray, key: jax.random.PRNGKey) -> np.ndarray:
        """Plan action sequence."""
        H = self.config.mppi_horizon
        N = self.config.mppi_samples
        lam = self.config.mppi_temperature
        sigma = self.config.mppi_sigma
        
        mean_actions = np.roll(self.mean_actions, -1)
        mean_actions[-1] = 0.0
        
        for _ in range(self.config.mppi_iterations):
            k, key = jax.random.split(key)
            noise = jax.random.normal(k, (N, H)) * sigma
            samples = mean_actions + noise
            samples = np.clip(samples, -self.config.max_impulse, self.config.max_impulse)
            
            costs = np.array([self.rollout(state0, samples[i]) for i in range(N)])
            
            costs_min = costs.min()
            weights = np.exp(-(costs - costs_min) / lam)
            weights = weights / weights.sum()
            
            mean_actions = np.sum(weights[:, None] * samples, axis=0)
        
        self.mean_actions = mean_actions
        return mean_actions


# ============================================================================
# Training
# ============================================================================

def train_model(model, params, train_obs, config):
    """Train model to predict next state."""
    tx = optax.adam(config.learning_rate)
    state = TrainState.create(apply_fn=model.apply, params=params, tx=tx)
    
    for epoch in range(config.epochs):
        losses = []
        for traj in train_obs:
            for t in range(len(traj) - 1):
                s_t = traj[t]
                s_t1 = traj[t + 1]
                
                def loss_fn(p):
                    pred = model.apply(p, s_t)
                    return jnp.mean((pred - s_t1) ** 2)
                
                loss, grads = jax.value_and_grad(loss_fn)(state.params)
                state = state.apply_gradients(grads=grads)
                losses.append(float(loss))
        
        if epoch % 10 == 0:
            print(f"  Epoch {epoch}: loss={np.mean(losses):.4f}")
    
    return state.params


# ============================================================================
# Execution
# ============================================================================

def execute_episode(env, model, params, config, planner, initial_state, key):
    """Execute closed-loop episode."""
    state = initial_state
    total_cost = 0.0
    
    for t in range(config.trajectory_length):
        # Plan
        k, key = jax.random.split(key)
        actions = planner.plan(state, k)
        
        # Execute first action
        action = float(actions[0])
        state_with_action = state.at[2].add(action)
        state, _ = env.step(state_with_action)
        
        # Cost
        total_cost = total_cost + (state[0] - config.x_target) ** 2 + 0.01 * action ** 2
    
    # Terminal cost
    total_cost = total_cost + 10.0 * (state[0] - config.x_target) ** 2
    
    success = abs(state[0] - config.x_target) < 0.5
    catastrophic = abs(state[0]) > 10.0
    
    return {
        'total_cost': float(total_cost),
        'success': success,
        'catastrophic': catastrophic,
        'final_x': float(state[0]),
    }


# ============================================================================
# Main
# ============================================================================

def main():
    print("\n" + "=" * 70)
    print("MPPI WITH LEARNED DYNAMICS (State-based)")
    print("=" * 70)
    
    config = Config()
    key = jax.random.PRNGKey(42)
    
    env = BouncingBall(BouncingBallParams(restitution=0.8))
    
    # Generate training data
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
    
    # Random baseline
    print("\n[RANDOM] Evaluating...")
    random_success = 0
    for ep in range(config.num_episodes):
        key, k1, k2 = jax.random.split(key, 3)
        y_init = jax.random.uniform(k1, minval=1.0, maxval=3.0)
        vy_init = jax.random.uniform(k2, minval=-2.0, maxval=2.0)
        state = jnp.array([0.0, y_init, 0.0, vy_init])
        
        for t in range(config.trajectory_length):
            k, key = jax.random.split(key)
            action = jax.random.uniform(k, (), minval=-config.max_impulse, maxval=config.max_impulse)
            state_with_action = state.at[2].add(action)
            state, _ = env.step(state_with_action)
        
        if abs(state[0] - config.x_target) < 0.5:
            random_success += 1
    
    random_rate = random_success / config.num_episodes
    print(f"  Success: {random_rate:.1%}")
    
    # Train model
    print("\n[MODEL] Training...")
    key, k1 = jax.random.split(key)
    model = DirectModel(config.hidden_dim)
    state0 = jnp.zeros(4)
    params = model.init(k1, state0)
    params = train_model(model, params, train_obs, config)
    
    # Test MPPI
    print("\n[MPPI] Evaluating...")
    planner = MPPIPlanner(model, params, config)
    
    mppi_success = 0
    for ep in range(config.num_episodes):
        key, k1, k2 = jax.random.split(key, 3)
        y_init = jax.random.uniform(k1, minval=1.0, maxval=3.0)
        vy_init = jax.random.uniform(k2, minval=-2.0, maxval=2.0)
        state = jnp.array([0.0, y_init, 0.0, vy_init])
        
        metrics = execute_episode(env, model, params, config, planner, state, k2)
        if metrics['success']:
            mppi_success += 1
    
    mppi_rate = mppi_success / config.num_episodes
    print(f"  Success: {mppi_rate:.1%}")
    
    # Summary
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print(f"  Random: {random_rate:.1%}")
    print(f"  MPPI:   {mppi_rate:.1%}")
    
    if mppi_rate > random_rate:
        print("\n✓ MPPI beats random!")
    else:
        print("\n✗ MPPI does NOT beat random")
    
    # Save
    os.makedirs('results/phase2', exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    with open(f'results/phase2/mppi_state_{timestamp}.json', 'w') as f:
        json.dump({
            'random': {'success_rate': random_rate},
            'mppi': {'success_rate': mppi_rate},
        }, f, indent=2)
    
    print(f"\nSaved to: results/phase2/mppi_state_{timestamp}.json")


if __name__ == "__main__":
    main()