#!/usr/bin/env python3
"""
PHASE 5: L4-A/B with JAX + GPU Acceleration

L4-A: Freeze encoder, train controller head
L4-B: End-to-end training
"""

import jax
import jax.numpy as jnp
from jax import random
import numpy as np


# Force GPU
print("JAX devices:", jax.devices())


# ============================================================================
# ENVIRONMENT
# ============================================================================

class BouncingBall:
    def __init__(self, restitution=0.8):
        self.g = 9.81
        self.e = restitution
        self.dt = 0.05
        self.x_target = 2.0
        self.tau = 0.3
        self.x_bounds = (0, 3)
        
    def reset(self, seed):
        np.random.seed(seed)
        start_positions = [0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5]
        self.x = start_positions[seed % len(start_positions)]
        self.v = 0.0
        return np.array([self.x, self.v])
    
    def step(self, a):
        a = np.clip(a, -2.0, 2.0)
        self.v += (-self.g + a) * self.dt
        self.x += self.v * self.dt
        
        if self.x < self.x_bounds[0]:
            self.x = -self.x * self.e
            self.v = -self.v * self.e
        elif self.x > self.x_bounds[1]:
            self.x = self.x_bounds[1] - (self.x - self.x_bounds[1]) * self.e
            self.v = -self.v * self.e
            
        return np.array([float(self.x), float(self.v)])


def generate_data(n_episodes, horizon, restitution, seed, obs_noise=0.05, dropout=0.1):
    np.random.seed(seed)
    env = BouncingBall(restitution=restitution)
    
    trajectories = []
    for ep in range(n_episodes):
        state = env.reset(seed=ep + seed)
        x, v = float(state[0]), float(state[1])
        
        traj = {'obs': [], 'acts': []}
        
        for step in range(horizon):
            obs = np.array([x])
            if obs_noise > 0:
                obs = obs + np.random.randn() * obs_noise
            if dropout > 0 and np.random.rand() < dropout:
                obs = obs * 0
            
            a_expert = float(np.clip(1.5 * (2.0 - x) + (-2.0) * (-v), -2.0, 2.0))
            
            next_state = env.step(a_expert)
            x_next, v_next = float(next_state[0]), float(next_state[1])
            
            traj['obs'].append(obs)
            traj['acts'].append(a_expert)
            
            x, v = x_next, v_next
        
        trajectories.append(traj)
    
    return trajectories


# ============================================================================
# MODEL (JAX)
# ============================================================================

def init_model(key, obs_dim=1, action_dim=1, latent_dim=8, belief_dim=16):
    k1, k2, k3 = random.split(key, 3)
    
    params = {
        'W_enc': random.normal(k1, (obs_dim, latent_dim)) * 0.3,
        'b_enc': jnp.zeros(latent_dim),
        'W_gru': random.normal(k2, (latent_dim + 1, belief_dim)) * 0.3,  # +1 for action
        'b_gru': jnp.zeros(belief_dim),
        'W_ctrl': random.normal(k3, (belief_dim, action_dim)) * 0.3,
        'b_ctrl': jnp.zeros(action_dim),
    }
    return params


def encode(params, obs):
    z = jnp.tanh(jnp.dot(obs, params['W_enc']) + params['b_enc'])
    return z


def gru_step(params, belief, z, a):
    # Ensure proper dimensions
    if hasattr(a, 'shape'):
        a_flat = a.flatten()
    else:
        a_flat = jnp.array([a])
    x = jnp.concatenate([z, a_flat])
    # W_gru should be (latent + action) x belief
    new = jnp.tanh(jnp.dot(x, params['W_gru']) + params['b_gru'])
    return 0.9 * belief + 0.1 * new


def controller(params, belief):
    return jnp.dot(belief, params['W_ctrl']).flatten() + params['b_ctrl']


def forward(params, obs_seq, act_seq):
    """Forward pass."""
    T = len(obs_seq)
    beliefs = []
    belief = jnp.zeros(16)  # belief_dim
    
    for t in range(T):
        z = encode(params, obs_seq[t])
        a_prev = act_seq[t-1] if t > 0 else jnp.zeros(1)
        belief = gru_step(params, belief, z, a_prev)
        beliefs.append(belief)
    
    return jnp.array(beliefs)


def loss_fn(params, trajectories, freeze_encoder=False):
    """Imitation loss."""
    total = 0.0
    count = 0
    
    for traj in trajectories:
        obs = jnp.array([jnp.array(o).reshape(-1) for o in traj['obs']])
        acts = jnp.array(traj['acts']).reshape(-1, 1)
        
        beliefs = forward(params, obs, acts)
        
        for t in range(len(beliefs)):
            a_pred = controller(params, beliefs[t])
            a_pred = jnp.clip(a_pred, -2.0, 2.0)
            total += jnp.mean((a_pred - acts[t]) ** 2)
            count += 1
    
    return total / max(count, 1)


# ============================================================================
# TRAINING
# ============================================================================

def train_l4a(trajectories, n_epochs=100, lr=0.01):
    """L4-A: Freeze encoder, train controller only."""
    key = random.PRNGKey(42)
    params = init_model(key)
    
    # First train JEPA (L1+L2) a bit
    print("  Pre-training JEPA (L1+L2)...")
    for epoch in range(20):
        grads = jax.grad(loss_fn)(params, trajectories)
        for k in ['W_enc', 'b_enc', 'W_gru', 'b_gru']:
            params[k] = params[k] - lr * grads[k]
    
    # NOW freeze encoder, train only controller
    print("  Training controller (L4-A - frozen encoder)...")
    for epoch in range(n_epochs):
        grads = jax.grad(loss_fn)(params, trajectories)
        # ONLY update controller
        params['W_ctrl'] = params['W_ctrl'] - lr * grads['W_ctrl']
        params['b_ctrl'] = params['b_ctrl'] - lr * grads['b_ctrl']
        
        if (epoch + 1) % 20 == 0:
            loss = loss_fn(params, trajectories)
            print(f"    Epoch {epoch+1}: loss={float(loss):.4f}")
    
    return params


def train_l4b(trajectories, n_epochs=100, lr=0.01):
    """L4-B: End-to-end training."""
    key = random.PRNGKey(42)
    params = init_model(key)
    
    print("  Training end-to-end (L4-B)...")
    for epoch in range(n_epochs):
        grads = jax.grad(loss_fn)(params, trajectories)
        # Update ALL parameters
        for k in params:
            params[k] = params[k] - lr * grads[k]
        
        if (epoch + 1) % 20 == 0:
            loss = loss_fn(params, trajectories)
            print(f"    Epoch {epoch+1}: loss={float(loss):.4f}")
    
    return params


def evaluate(params, n_episodes, restitution, obs_noise, dropout, seed):
    env = BouncingBall(restitution=restitution)
    
    successes = 0
    terminal_misses = []
    
    for ep in range(n_episodes):
        state = env.reset(seed=ep + seed + 1000)
        x, v = float(state[0]), float(state[1])
        
        belief = np.zeros(16)
        last_a = 0.0
        
        for step in range(30):
            obs = np.array([x])
            if obs_noise > 0 and np.random.rand() < obs_noise:
                obs = obs + np.random.randn() * obs_noise
            if dropout > 0 and np.random.rand() < dropout:
                obs = obs * 0
            
            obs = jnp.array(obs)
            z = encode(params, obs)
            belief = gru_step(params, belief, z, last_a)
            
            a = float(controller(params, belief)[0])
            a = float(np.clip(a, -2.0, 2.0))
            last_a = a
            
            state = env.step(a)
            x, v = float(state[0]), float(state[1])
            
            if abs(x - env.x_target) < env.tau:
                successes += 1
                break
        
        terminal_misses.append(abs(x - env.x_target))
    
    return successes / n_episodes, np.mean(terminal_misses)


def main():
    print("="*70)
    print("PHASE 5: L4 with JAX + GPU")
    print("="*70)
    
    restitution = 0.8
    obs_noise = 0.05
    dropout = 0.1
    
    print("\n1. Generating data...")
    train_trajs = generate_data(500, 30, restitution, seed=42, obs_noise=obs_noise, dropout=dropout)
    print(f"   Generated {len(train_trajs)} trajectories")
    
    # L4-A
    print("\n2. L4-A: Frozen encoder + controller head...")
    params_l4a = train_l4a(train_trajs, n_epochs=100, lr=0.01)
    
    print("\n3. Evaluating L4-A...")
    rate_l4a, miss_l4a = evaluate(params_l4a, 200, restitution, obs_noise, dropout, seed=42)
    print(f"   L4-A success: {rate_l4a:.1%}, miss: {miss_l4a:.3f}")
    
    # L4-B
    print("\n4. L4-B: End-to-end training...")
    params_l4b = train_l4b(train_trajs, n_epochs=100, lr=0.01)
    
    print("\n5. Evaluating L4-B...")
    rate_l4b, miss_l4b = evaluate(params_l4b, 200, restitution, obs_noise, dropout, seed=42)
    print(f"   L4-B success: {rate_l4b:.1%}, miss: {miss_l4b:.3f}")
    
    print("\n" + "="*70)
    print("COMPARISON")
    print("="*70)
    print(f"PD (full state):   71.0%")
    print(f"PD (partial):      56.5%")
    print(f"L4-A (frozen):     {rate_l4a:.1%}")
    print(f"L4-B (e2e):        {rate_l4b:.1%}")
    
    gap_a = (rate_l4a - 0.565) / (0.71 - 0.565)
    gap_b = (rate_l4b - 0.565) / (0.71 - 0.565)
    print(f"\nGap closed: L4-A={gap_a:.1%}, L4-B={gap_b:.1%}")


if __name__ == '__main__':
    main()
