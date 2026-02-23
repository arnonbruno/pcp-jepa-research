#!/usr/bin/env python3
"""
PHASE 5: JEPA Belief Model (Steps 5-6) - FIXED

With proper gradient updates and training.
"""

import numpy as np
from typing import List, Tuple
import os


# ============================================================================
# ENVIRONMENT + DATASET
# ============================================================================

class BouncingBallGravity:
    """Bouncing ball with configurable restitution."""
    
    def __init__(self, tau=0.3, restitution=0.8):
        self.g = 9.81
        self.e = restitution
        self.dt = 0.05
        self.x_target = 2.0
        self.tau = tau
        self.x_bounds = (0, 3)
        
    def reset(self, seed=None):
        if seed is not None:
            np.random.seed(seed)
        start_positions = [0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5]
        self.x = start_positions[seed % len(start_positions)]
        self.v = 0.0
        return np.array([self.x, self.v])
    
    def step(self, a):
        a = np.clip(a, -2.0, 2.0)
        self.v += (-self.g + a) * self.dt
        self.x += self.v * self.dt
        
        bounced = False
        if self.x < self.x_bounds[0]:
            self.x = -self.x * self.e
            self.v = -self.v * self.e
            bounced = True
        elif self.x > self.x_bounds[1]:
            self.x = self.x_bounds[1] - (self.x - self.x_bounds[1]) * self.e
            self.v = -self.v * self.e
            bounced = True
            
        return np.array([self.x, self.v]), bounced


class PartialObservation:
    """Partial observation: position only + noise + dropout."""
    
    def __init__(self, noise_std=0.0, dropout_prob=0.0):
        self.noise_std = noise_std
        self.dropout_prob = dropout_prob
    
    def observe(self, state):
        x, v = state
        obs = np.array([x])
        
        if self.noise_std > 0:
            obs = obs + np.random.randn(*obs.shape) * self.noise_std
        
        if self.dropout_prob > 0 and np.random.rand() < self.dropout_prob:
            obs = np.zeros_like(obs)
        
        return obs


def generate_dataset(
    n_episodes=1000,
    horizon=30,
    restitution=0.8,
    policy='random',
    seed=0,
    obs_noise=0.0,
    obs_dropout=0.0
):
    """Generate trajectory dataset with labels."""
    
    np.random.seed(seed)
    env = BouncingBallGravity(restitution=restitution)
    obs_model = PartialObservation(noise_std=obs_noise, dropout_prob=obs_dropout)
    
    trajectories = []
    
    for ep in range(n_episodes):
        state = env.reset(seed=ep + seed)
        x, v = state[0], state[1]
        
        traj = {
            'observations': [],
            'states': [],
            'actions': [],
            'contacts': [],
            'times_to_impact': [],
            'restitutions': [],
        }
        
        for step in range(horizon):
            observation = obs_model.observe(state)
            
            if policy == 'random':
                a = np.random.uniform(-2.0, 2.0)
            elif policy == 'pd':
                a = 1.5 * (2.0 - x) + (-2.0) * (-v)
                a = np.clip(a, -2.0, 2.0)
            
            next_state, bounced = env.step(a)
            x_next, v_next = next_state[0], next_state[1]
            
            # Time to impact
            time_to_impact = 0
            if bounced:
                contact = 1
            else:
                x_look, v_look = x_next, v_next
                tti = 0
                for look_step in range(1, 20):
                    v_look += (-9.81 + a) * env.dt
                    x_look += v_look * env.dt
                    if x_look < 0 or x_look > 3:
                        tti = look_step
                        break
                time_to_impact = tti
                contact = 1 if tti <= 5 else 0
            
            traj['observations'].append(observation)
            traj['states'].append([x, v])
            traj['actions'].append(a)
            traj['contacts'].append(contact)
            traj['times_to_impact'].append(time_to_impact)
            traj['restitutions'].append(restitution)
            
            state = next_state
            x, v = x_next, v_next
        
        trajectories.append(traj)
    
    return trajectories


# ============================================================================
# JEPA BELIEF MODEL WITH PROPER TRAINING
# ============================================================================

class JEPABeliefModel:
    """JEPA Belief Model with trainable weights."""
    
    def __init__(self, obs_dim=1, action_dim=1, latent_dim=16, hidden_dim=32, belief_dim=32):
        self.obs_dim = obs_dim
        self.action_dim = action_dim
        self.latent_dim = latent_dim
        self.hidden_dim = hidden_dim
        self.belief_dim = belief_dim
        
        np.random.seed(42)
        
        # Encoder
        scale = 0.3
        self.W_enc1 = np.random.randn(obs_dim, hidden_dim) * scale
        self.b_enc1 = np.zeros(hidden_dim)
        self.W_enc2 = np.random.randn(hidden_dim, latent_dim) * scale
        self.b_enc2 = np.zeros(latent_dim)
        
        # GRU
        self.W_gru = np.random.randn(latent_dim + action_dim, belief_dim) * scale
        self.b_gru = np.zeros(belief_dim)
        
        # Predictor
        self.W_pred = np.random.randn(belief_dim + action_dim, belief_dim) * scale
        self.b_pred = np.zeros(belief_dim)
        
        # Event heads
        self.W_contact = np.random.randn(belief_dim, 1) * scale
        self.b_contact = np.zeros(1)
        self.W_tti = np.random.randn(belief_dim, 1) * scale
        self.b_tti = np.zeros(1)
        
        # Controller head (L4)
        self.W_ctrl = np.random.randn(belief_dim, action_dim) * scale
        self.b_ctrl = np.zeros(action_dim)
    
    def encode(self, obs):
        h = np.tanh(np.dot(obs, self.W_enc1) + self.b_enc1)
        z = np.tanh(np.dot(h, self.W_enc2) + self.b_enc2)
        return z
    
    def gru_step(self, belief, z, a):
        if a.ndim == 0:
            a = np.array([a])
        elif a.shape[-1] != self.action_dim:
            a = a.reshape(-1)
        x = np.concatenate([z, a])
        new_belief = np.tanh(np.dot(x, self.W_gru) + self.b_gru)
        return 0.9 * belief + 0.1 * new_belief
    
    def predict(self, belief, a):
        if a.ndim == 0:
            a = np.array([a])
        elif a.shape[-1] != self.action_dim:
            a = a.reshape(-1)
        x = np.concatenate([belief, a])
        return np.tanh(np.dot(x, self.W_pred) + self.b_pred)
    
    def contact_head(self, belief):
        logit = np.dot(belief, self.W_contact) + self.b_contact
        return 1 / (1 + np.exp(-np.clip(logit, -500, 500)))
    
    def tti_head(self, belief):
        return np.clip(np.dot(belief, self.W_tti) + self.b_tti, 0, 20)
    
    def controller_head(self, belief):
        return np.dot(belief, self.W_ctrl) + self.b_ctrl
    
    def forward(self, observations, actions):
        T = len(observations)
        beliefs = []
        a_prev = np.zeros(self.action_dim)
        belief = np.zeros(self.belief_dim)
        
        for t in range(T):
            z = self.encode(observations[t])
            if t > 0:
                a_prev = actions[t-1] if np.isscalar(actions[t-1]) else actions[t-1].flatten()
                if a_prev.ndim > 1:
                    a_prev = a_prev.reshape(-1)
            belief = self.gru_step(belief, z, a_prev)
            beliefs.append(belief)
        
        return beliefs


def compute_gradients(model, trajectories, k_values=[1, 2, 5], lr=0.001):
    """Compute gradients for L1 + L2 losses using finite differences."""
    
    # Collect all parameters
    params = [
        model.W_enc1, model.b_enc1, model.W_enc2, model.b_enc2,
        model.W_gru, model.b_gru,
        model.W_pred, model.b_pred,
        model.W_contact, model.b_contact,
        model.W_tti, model.b_tti,
    ]
    
    param_names = [
        'W_enc1', 'b_enc1', 'W_enc2', 'b_enc2',
        'W_gru', 'b_gru',
        'W_pred', 'b_pred',
        'W_contact', 'b_contact',
        'W_tti', 'b_tti',
    ]
    
    # Compute loss with numerical gradients
    epsilon = 1e-5
    grads = {}
    
    for name, param in zip(param_names, params):
        grad = np.zeros_like(param)
        it = np.nditer(param, flags=['multi_index'], op_flags=['readwrite'])
        while not it.finished:
            idx = it.multi_index
            old_val = param[idx]
            
            param[idx] = old_val + epsilon
            loss_plus = compute_total_loss(model, trajectories, k_values)
            
            param[idx] = old_val - epsilon
            loss_minus = compute_total_loss(model, trajectories, k_values)
            
            grad[idx] = (loss_plus - loss_minus) / (2 * epsilon)
            param[idx] = old_val
            it.iternext()
        
        grads[name] = grad
    
    return grads


def compute_total_loss(model, trajectories, k_values=[1, 2, 5]):
    """Compute L1 + L2 total loss."""
    
    jepa_loss = 0.0
    event_loss = 0.0
    count = 0
    
    for traj in trajectories:
        obs = traj['observations']
        acts = traj['actions']
        contacts = traj['contacts']
        ttis = traj['times_to_impact']
        
        if len(obs) < max(k_values) + 1:
            continue
        
        beliefs = model.forward(obs, acts)
        
        # JEPA loss
        for k in k_values:
            for t in range(len(beliefs) - k):
                # Predict k steps
                b_pred = beliefs[t]
                for step in range(k):
                    if t + step < len(acts):
                        a = acts[t + step] if np.isscalar(acts[t + step]) else acts[t + step].flatten()
                        if a.ndim > 1:
                            a = a.reshape(-1)
                        b_pred = model.predict(b_pred, a)
                
                target = beliefs[t + k]
                jepa_loss += np.mean((b_pred - target) ** 2)
                count += 1
        
        # Event loss
        for t, (b, c, tau) in enumerate(zip(beliefs, contacts, ttis)):
            pred_c = model.contact_head(b)[0]
            pred_c = np.clip(pred_c, 1e-7, 1 - 1e-7)
            event_loss += -c * np.log(pred_c) - (1 - c) * np.log(1 - pred_c)
            
            pred_tau = model.tti_head(b)[0]
            event_loss += abs(pred_tau - tau)
    
    jepa_loss = jepa_loss / max(count, 1)
    event_loss = event_loss / max(len(beliefs), 1)
    
    return jepa_loss + 0.5 * event_loss


def train_with_gradients(model, trajectories, n_epochs=50, lr=0.001):
    """Train with numerical gradients."""
    
    k_values = [1, 2, 5]
    
    print(f"Training with gradient descent...")
    
    for epoch in range(n_epochs):
        # Compute gradients
        grads = compute_gradients(model, trajectories, k_values, lr)
        
        # Update weights
        model.W_enc1 -= lr * grads['W_enc1']
        model.b_enc1 -= lr * grads['b_enc1']
        model.W_enc2 -= lr * grads['W_enc2']
        model.b_enc2 -= lr * grads['b_enc2']
        model.W_gru -= lr * grads['W_gru']
        model.b_gru -= lr * grads['b_gru']
        model.W_pred -= lr * grads['W_pred']
        model.b_pred -= lr * grads['b_pred']
        model.W_contact -= lr * grads['W_contact']
        model.b_contact -= lr * grads['b_contact']
        model.W_tti -= lr * grads['W_tti']
        model.b_tti -= lr * grads['b_tti']
        
        if (epoch + 1) % 10 == 0:
            loss = compute_total_loss(model, trajectories, k_values)
            print(f"  Epoch {epoch+1}: loss = {loss:.4f}")
    
    return model


def evaluate_controller(model, n_episodes=200, restitution=0.8, obs_noise=0.05, obs_dropout=0.1, seed=0):
    """Evaluate JEPA + linear controller."""
    
    np.random.seed(seed)
    env = BouncingBallGravity(restitution=restitution)
    obs_model = PartialObservation(noise_std=obs_noise, dropout_prob=obs_dropout)
    
    # Train controller head via imitation
    train_trajs = generate_dataset(
        n_episodes=300, restitution=restitution, policy='pd',
        seed=seed, obs_noise=obs_noise, obs_dropout=obs_dropout
    )
    
    X = []
    y = []
    for traj in train_trajs:
        beliefs = model.forward(traj['observations'], traj['actions'])
        for b, a in zip(beliefs, traj['actions']):
            X.append(b)
            y.append(a if np.isscalar(a) else a.flatten()[0])
    
    X = np.array(X)
    y = np.array(y)
    
    W_ctrl = np.linalg.lstsq(X, y, rcond=None)[0]
    
    # Evaluate
    successes = 0
    terminal_misses = []
    
    for ep in range(n_episodes):
        state = env.reset(seed=ep + seed + 1000)
        x, v = state[0], state[1]
        
        belief = np.zeros(model.belief_dim)
        a_prev = np.zeros(model.action_dim)
        
        for step in range(30):
            observation = obs_model.observe(state)
            z = model.encode(observation)
            belief = model.gru_step(belief, z, a_prev)
            
            a = np.dot(belief, W_ctrl)
            a = np.clip(a, -2.0, 2.0)
            
            state, _ = env.step(a)
            x, v = state[0], state[1]
            a_prev = np.array([a])
            
            if abs(x - env.x_target) < env.tau:
                successes += 1
                break
        
        terminal_misses.append(abs(x - env.x_target))
    
    return {
        'success_rate': successes / n_episodes,
        'terminal_miss': np.mean(terminal_misses),
    }


def main():
    print("="*70)
    print("PHASE 5: JEPA Belief Model - PROPER TRAINING")
    print("="*70)
    
    # Generate data
    print("\n1. Generating dataset...")
    trajectories = generate_dataset(
        n_episodes=300,
        horizon=30,
        restitution=0.8,
        policy='pd',
        seed=42,
        obs_noise=0.05,
        obs_dropout=0.1
    )
    print(f"   Generated {len(trajectories)} trajectories")
    
    # Create model
    print("\n2. Creating JEPA model...")
    model = JEPABeliefModel(
        obs_dim=1,
        action_dim=1,
        latent_dim=16,
        hidden_dim=32,
        belief_dim=32
    )
    
    # Train
    print("\n3. Training with gradients...")
    model = train_with_gradients(model, trajectories, n_epochs=30, lr=0.01)
    
    # Evaluate
    print("\n4. Evaluating JEPA + linear controller...")
    result = evaluate_controller(
        model,
        n_episodes=200,
        restitution=0.8,
        obs_noise=0.05,
        obs_dropout=0.1,
        seed=42
    )
    
    print(f"\n   JEPA + linear controller:")
    print(f"     Success rate: {result['success_rate']:.1%}")
    print(f"     Terminal miss: {result['terminal_miss']:.3f}")
    
    # Compare
    print("\n" + "="*70)
    print("COMPARISON")
    print("="*70)
    print(f"PD (full state):      71.0%")
    print(f"PD (partial, no v):  56.5%")
    print(f"JEPA + linear ctrl:  {result['success_rate']:.1%}")
    
    gap_closed = (result['success_rate'] - 0.565) / (0.71 - 0.565)
    print(f"\nGap closed: {gap_closed:.1%}")
    
    if result['success_rate'] > 0.60:
        print("\n✓ L1+L2 is working!")
    else:
        print("\n✗ L1+L2 still not enough.")


if __name__ == '__main__':
    main()
