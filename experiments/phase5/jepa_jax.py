#!/usr/bin/env python3
"""
PHASE 5: JEPA Belief Model with JAX (FIXED)
"""

import jax
import jax.numpy as jnp
from jax import random
import numpy as np


# ============================================================================
# ENVIRONMENT
# ============================================================================

class BouncingBallGravity:
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
        
        if self.x < self.x_bounds[0]:
            self.x = -self.x * self.e
            self.v = -self.v * self.e
        elif self.x > self.x_bounds[1]:
            self.x = self.x_bounds[1] - (self.x - self.x_bounds[1]) * self.e
            self.v = -self.v * self.e
            
        return np.array([float(self.x), float(self.v)])


def generate_dataset(n_episodes, horizon, restitution, policy, seed, obs_noise=0.0, dropout=0.0):
    np.random.seed(seed)
    env = BouncingBallGravity(restitution=restitution)
    
    trajectories = []
    for ep in range(n_episodes):
        state = env.reset(seed=ep + seed)
        x, v = float(state[0]), float(state[1])
        
        traj = {'obs': [], 'states': [], 'acts': [], 'contacts': [], 'ttis': []}
        
        for step in range(horizon):
            # Partial obs: position only
            obs = np.array([float(x)])
            if obs_noise > 0:
                obs = obs + np.random.randn() * obs_noise
            if dropout > 0 and np.random.rand() < dropout:
                obs = obs * 0
            
            # Action
            if policy == 'pd':
                a = float(np.clip(1.5 * (2.0 - x) + (-2.0) * (-v), -2.0, 2.0))
            else:
                a = float(np.random.uniform(-2.0, 2.0))
            
            next_state = env.step(a)
            x_next, v_next = next_state[0], next_state[1]
            
            # Labels
            x_look, v_look = x_next, v_next
            tti = 0
            for ls in range(1, 20):
                v_look += (-9.81 + a) * env.dt
                x_look += v_look * env.dt
                if x_look < 0 or x_look > 3:
                    tti = ls
                    break
            contact = 1 if tti <= 5 else 0
            
            traj['obs'].append(obs)
            traj['states'].append([x, v])
            traj['acts'].append(a)
            traj['contacts'].append(contact)
            traj['ttis'].append(tti)
            
            x, v = x_next, v_next
        
        trajectories.append(traj)
    
    return trajectories


# ============================================================================
# JEPA MODEL (JAX)
# ============================================================================

def init_model(key, obs_dim=1, action_dim=1, latent_dim=16, hidden_dim=32, belief_dim=32):
    k1, k2, k3, k4, k5 = random.split(key, 5)
    
    params = {
        'W_enc1': random.normal(k1, (obs_dim, hidden_dim)) * 0.3,
        'b_enc1': jnp.zeros(hidden_dim),
        'W_enc2': random.normal(k2, (hidden_dim, latent_dim)) * 0.3,
        'b_enc2': jnp.zeros(latent_dim),
        
        'W_gru': random.normal(k3, (latent_dim + action_dim, belief_dim)) * 0.3,
        'b_gru': jnp.zeros(belief_dim),
        
        'W_pred': random.normal(k4, (belief_dim + action_dim, belief_dim)) * 0.3,
        'b_pred': jnp.zeros(belief_dim),
        
        'W_contact': random.normal(k5, (belief_dim, 1)) * 0.3,
        'b_contact': jnp.zeros(1),
        'W_tti': random.normal(key, (belief_dim, 1)) * 0.3,
        'b_tti': jnp.zeros(1),
        
        'W_ctrl': random.normal(key, (belief_dim, action_dim)) * 0.3,
        'b_ctrl': jnp.zeros(action_dim),
    }
    return params


def encode(params, obs):
    h = jnp.tanh(jnp.dot(obs, params['W_enc1']) + params['b_enc1'])
    z = jnp.tanh(jnp.dot(h, params['W_enc2']) + params['b_enc2'])
    return z


def gru_step(params, belief, z, a):
    x = jnp.concatenate([z, jnp.array([a]).reshape(-1)])
    new_belief = jnp.tanh(jnp.dot(x, params['W_gru']) + params['b_gru'])
    return 0.9 * belief + 0.1 * new_belief


def predict(params, belief, a):
    x = jnp.concatenate([belief, jnp.array([a]).reshape(-1)])
    return jnp.tanh(jnp.dot(x, params['W_pred']) + params['b_pred'])


def forward_trajectory(params, obs_seq, act_seq):
    T = len(obs_seq)
    beliefs = []
    belief = jnp.zeros(32)
    
    for t in range(T):
        z = encode(params, obs_seq[t])
        a_prev = act_seq[t-1] if t > 0 else jnp.zeros(1)
        belief = gru_step(params, belief, z, a_prev)
        beliefs.append(belief)
    
    return jnp.array(beliefs)


def loss_fn(params, trajectories, k_values=[1, 2, 5]):
    jepa_loss = 0.0
    contact_loss = 0.0
    tti_loss = 0.0
    count = 0
    
    for traj in trajectories:
        obs = jnp.array([jnp.array(o).reshape(-1) for o in traj['obs']])
        acts = jnp.array(traj['acts']).reshape(-1, 1)
        contacts = jnp.array(traj['contacts'])
        ttis = jnp.array(traj['ttis'])
        
        if len(obs) < max(k_values) + 1:
            continue
        
        beliefs = forward_trajectory(params, obs, acts)
        
        for k in k_values:
            for t in range(len(beliefs) - k):
                b_pred = beliefs[t]
                for step in range(k):
                    b_pred = predict(params, b_pred, acts[t + step])
                
                target = beliefs[t + k]
                jepa_loss += jnp.mean((b_pred - target) ** 2)
                count += 1
        
        for t in range(len(beliefs)):
            b = beliefs[t]
            
            logit = jnp.dot(b, params['W_contact']) + params['b_contact']
            pred_c = jax.nn.sigmoid(logit)[0]
            pred_c = jnp.clip(pred_c, 1e-7, 1 - 1e-7)
            contact_loss += -contacts[t] * jnp.log(pred_c) - (1 - contacts[t]) * jnp.log(1 - pred_c)
            
            pred_t = jnp.dot(b, params['W_tti']) + params['b_tti']
            pred_t = jnp.clip(pred_t, 0, 20)[0]
            tti_loss += (pred_t - ttis[t]) ** 2
    
    jepa_loss = jepa_loss / max(count, 1)
    contact_loss = contact_loss / max(len(beliefs), 1)
    tti_loss = tti_loss / max(len(beliefs), 1)
    
    total = jepa_loss + 1.0 * contact_loss + 0.5 * tti_loss
    return total, (jepa_loss, contact_loss, tti_loss)


def train(trajectories, n_epochs=100, lr=0.001):
    key = random.PRNGKey(42)
    params = init_model(key)
    
    grad_fn = jax.grad(loss_fn, has_aux=True)
    
    print(f"Training on {len(trajectories)} trajectories...")
    
    for epoch in range(n_epochs):
        grads, (jepa, contact, tti) = grad_fn(params, trajectories)
        
        for key in params:
            params[key] = params[key] - lr * grads[key]
        
        if (epoch + 1) % 20 == 0:
            print(f"  Epoch {epoch+1}: total={jepa + contact + tti:.4f} (jepa={jepa:.4f}, c={contact:.4f}, tti={tti:.4f})")
    
    return params


def evaluate(params, n_episodes=200, restitution=0.8, obs_noise=0.05, dropout=0.1, seed=0):
    np.random.seed(seed)
    env = BouncingBallGravity(restitution=restitution)
    
    # Train controller head
    train_trajs = generate_dataset(300, 30, restitution, 'pd', seed, obs_noise, dropout)
    
    X = []
    y = []
    for traj in train_trajs:
        obs = jnp.array([jnp.array(o).reshape(-1) for o in traj['obs']])
        acts = jnp.array(traj['acts']).reshape(-1, 1)
        beliefs = forward_trajectory(params, obs, acts)
        
        for b, a in zip(beliefs, acts):
            X.append(np.array(b))
            y.append(np.array(float(a.flatten()[0])))
    
    X = np.array(X)
    y = np.array(y)
    W_ctrl = np.linalg.lstsq(X, y, rcond=None)[0]
    
    # Evaluate
    successes = 0
    
    for ep in range(n_episodes):
        state = env.reset(seed=ep + seed + 1000)
        x, v = float(state[0]), float(state[1])
        
        belief = np.zeros(32)
        a_prev = 0.0
        
        for step in range(30):
            obs = np.array([float(x)])
            if obs_noise > 0 and np.random.rand() < obs_noise:
                obs = obs + np.random.randn() * obs_noise
            if dropout > 0 and np.random.rand() < dropout:
                obs = obs * 0
            
            obs = jnp.array(obs)
            z = encode(params, obs)
            belief = gru_step(params, belief, z, a_prev)
            
            a = np.dot(np.array(belief), W_ctrl)
            a = float(np.clip(a, -2.0, 2.0))
            
            state = env.step(a)
            x, v = float(state[0]), float(state[1])
            a_prev = a
            
            if abs(x - env.x_target) < env.tau:
                successes += 1
                break
    
    return successes / n_episodes


def main():
    print("="*70)
    print("PHASE 5: JEPA Belief Model (JAX - FIXED)")
    print("="*70)
    
    print("\n1. Generating dataset...")
    trajectories = generate_dataset(500, 30, 0.8, 'pd', 42, 0.05, 0.1)
    print(f"   Generated {len(trajectories)} trajectories")
    
    print("\n2. Training JEPA model...")
    params = train(trajectories, n_epochs=100, lr=0.001)
    
    print("\n3. Evaluating...")
    success_rate = evaluate(params, n_episodes=200, restitution=0.8, seed=42)
    
    print(f"\n   JEPA + linear controller: {success_rate:.1%}")
    
    print("\n" + "="*70)
    print("COMPARISON")
    print("="*70)
    print(f"PD (full state):      71.0%")
    print(f"PD (partial, no v):  56.5%")
    print(f"JEPA + linear ctrl:  {success_rate:.1%}")
    
    gap_closed = (success_rate - 0.565) / (0.71 - 0.565)
    print(f"\nGap closed: {gap_closed:.1%}")
    
    if success_rate > 0.60:
        print("\n✓ L1+L2 is working!")
    else:
        print("\n✗ L1+L2 not enough - need L3/L4")


if __name__ == '__main__':
    main()
