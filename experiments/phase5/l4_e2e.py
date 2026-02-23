#!/usr/bin/env python3
"""
PHASE 5: L4-B - End-to-End Control-Aware Training

Train encoder+GRU jointly with imitation loss.
This directly shapes the belief for control.
"""

import numpy as np


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
        
        traj = {'obs': [], 'states': [], 'acts': []}
        
        for step in range(horizon):
            obs = np.array([x])
            if obs_noise > 0:
                obs = obs + np.random.randn() * obs_noise
            if dropout > 0 and np.random.rand() < dropout:
                obs = obs * 0
            
            # Expert PD
            a_expert = 1.5 * (2.0 - x) + (-2.0) * (-v)
            a_expert = float(np.clip(a_expert, -2.0, 2.0))
            
            next_state = env.step(a_expert)
            x_next, v_next = float(next_state[0]), float(next_state[1])
            
            traj['obs'].append(obs)
            traj['states'].append([x, v])
            traj['acts'].append(a_expert)
            
            x, v = x_next, v_next
        
        trajectories.append(traj)
    
    return trajectories


# ============================================================================
# MODEL
# ============================================================================

class JEPAController:
    def __init__(self, obs_dim=1, action_dim=1, latent_dim=8, belief_dim=16):
        self.obs_dim = obs_dim
        self.action_dim = action_dim
        self.latent_dim = latent_dim
        self.belief_dim = belief_dim
        
        np.random.seed(42)
        
        # Encoder
        self.W_enc = np.random.randn(obs_dim, latent_dim) * 0.3
        self.b_enc = np.zeros(latent_dim)
        
        # GRU
        self.W_gru = np.random.randn(latent_dim + action_dim, belief_dim) * 0.3
        self.b_gru = np.zeros(belief_dim)
        
        # Controller
        self.W_ctrl = np.random.randn(belief_dim, action_dim) * 0.3
        self.b_ctrl = np.zeros(action_dim)
    
    def encode(self, obs):
        return np.tanh(np.dot(obs, self.W_enc) + self.b_enc)
    
    def gru_step(self, belief, z, a):
        x = np.concatenate([z, np.array([a])])
        new = np.tanh(np.dot(x, self.W_gru) + self.b_gru)
        return 0.9 * belief + 0.1 * new
    
    def controller(self, belief):
        return np.dot(belief, self.W_ctrl) + self.b_ctrl
    
    def forward(self, observations, actions):
        beliefs = []
        belief = np.zeros(self.belief_dim)
        
        for t in range(len(observations)):
            z = self.encode(observations[t])
            a_prev = actions[t-1] if t > 0 else 0.0
            belief = self.gru_step(belief, z, a_prev)
            beliefs.append(belief)
        
        return beliefs


def numerical_grad(model, trajectories, lr=0.001, eps=1e-4):
    """Compute gradients via numerical differentiation."""
    
    params = {
        'W_enc': model.W_enc, 'b_enc': model.b_enc,
        'W_gru': model.W_gru, 'b_gru': model.b_gru,
        'W_ctrl': model.W_ctrl, 'b_ctrl': model.b_ctrl,
    }
    
    # Loss function
    def compute_loss():
        total = 0.0
        for traj in trajectories:
            obs = traj['obs']
            acts = traj['acts']
            
            beliefs = model.forward(obs, acts)
            
            for t, (belief, a_expert) in enumerate(zip(beliefs, acts)):
                a_pred = model.controller(belief)[0]
                a_pred = np.clip(a_pred, -2.0, 2.0)
                total += (a_pred - a_expert) ** 2
        
        return total / max(len(trajectories), 1)
    
    # Compute gradients for each parameter
    grads = {}
    for name, param in params.items():
        grad = np.zeros_like(param)
        
        it = np.nditer(param, flags=['multi_index'], op_flags=['readwrite'])
        while not it.finished:
            idx = it.multi_index
            old = param[idx]
            
            param[idx] = old + eps
            loss_plus = compute_loss()
            
            param[idx] = old - eps
            loss_minus = compute_loss()
            
            grad[idx] = (loss_plus - loss_minus) / (2 * eps)
            param[idx] = old
            it.iternext()
        
        grads[name] = grad
    
    return grads


def train_end_to_end(model, trajectories, n_epochs=100, lr=0.01):
    """L4-B: Train encoder + controller jointly."""
    
    print(f"Training end-to-end (L4-B)...")
    
    for epoch in range(n_epochs):
        # Compute numerical gradients
        grads = numerical_grad(model, trajectories, lr=lr)
        
        # Update
        model.W_enc -= lr * grads['W_enc']
        model.b_enc -= lr * grads['b_enc']
        model.W_gru -= lr * grads['W_gru']
        model.b_gru -= lr * grads['b_gru']
        model.W_ctrl -= lr * grads['W_ctrl']
        model.b_ctrl -= lr * grads['b_ctrl']
        
        if (epoch + 1) % 20 == 0:
            # Compute loss
            total = 0.0
            count = 0
            for traj in trajectories:
                obs = traj['obs']
                acts = traj['acts']
                beliefs = model.forward(obs, acts)
                
                for belief, a_expert in zip(beliefs, acts):
                    a_pred = model.controller(belief)[0]
                    a_pred = np.clip(a_pred, -2.0, 2.0)
                    total += (a_pred - a_expert) ** 2
                    count += 1
            
            print(f"  Epoch {epoch+1}: imitation loss={total/max(count,1):.4f}")
    
    return model


def evaluate(model, n_episodes, restitution, obs_noise, dropout, seed):
    env = BouncingBall(restitution=restitution)
    
    successes = 0
    terminal_misses = []
    bounce_counts = []
    
    for ep in range(n_episodes):
        state = env.reset(seed=ep + seed + 1000)
        x, v = float(state[0]), float(state[1])
        
        belief = np.zeros(model.belief_dim)
        bounces = 0
        prev_x = x
        
        for step in range(30):
            obs = np.array([x])
            if obs_noise > 0 and np.random.rand() < obs_noise:
                obs = obs + np.random.randn() * obs_noise
            if dropout > 0 and np.random.rand() < dropout:
                obs = obs * 0
            
            z = model.encode(obs)
            a_prev = 0.0 if step == 0 else last_a
            belief = model.gru_step(belief, z, a_prev)
            
            a = model.controller(belief)[0]
            a = float(np.clip(a, -2.0, 2.0))
            last_a = a
            
            next_state = env.step(a)
            x, v = float(next_state[0]), float(next_state[1])
            
            if (prev_x < 0.1 and x > prev_x) or (prev_x > 2.9 and x < prev_x):
                bounces += 1
            prev_x = x
            
            if abs(x - env.x_target) < env.tau:
                successes += 1
                break
        
        terminal_misses.append(abs(x - env.x_target))
        bounce_counts.append(bounces)
    
    return {
        'success_rate': successes / n_episodes,
        'terminal_miss': np.mean(terminal_misses),
        'bounce_count': np.mean(bounce_counts),
    }


def main():
    print("="*70)
    print("PHASE 5: L4-B - End-to-End Control-Aware Training")
    print("="*70)
    
    restitution = 0.8
    obs_noise = 0.05
    dropout = 0.1
    
    print("\n1. Generating training data...")
    train_trajs = generate_data(300, 30, restitution, seed=42, 
                                obs_noise=obs_noise, dropout=dropout)
    print(f"   Generated {len(train_trajs)} trajectories")
    
    print("\n2. Creating model...")
    model = JEPAController(obs_dim=1, action_dim=1, latent_dim=8, belief_dim=16)
    
    print("\n3. Training end-to-end (L4-B)...")
    model = train_end_to_end(model, train_trajs, n_epochs=50, lr=0.01)
    
    print("\n4. Evaluating...")
    result = evaluate(model, 200, restitution, obs_noise, dropout, seed=42)
    
    print(f"\n   JEPA + end-to-end ctrl:")
    print(f"     Success rate: {result['success_rate']:.1%}")
    print(f"     Terminal miss: {result['terminal_miss']:.3f}")
    print(f"     Bounce count: {result['bounce_count']:.1f}")
    
    print("\n" + "="*70)
    print("COMPARISON")
    print("="*70)
    print(f"PD (full state):          71.0%")
    print(f"PD (partial):             56.5%")
    print(f"L4-A (frozen enc):       56.5%")
    print(f"L4-B (end-to-end):       {result['success_rate']:.1%}")
    
    gap_closed = (result['success_rate'] - 0.565) / (0.71 - 0.565)
    print(f"\nGap closed: {gap_closed:.1%}")
    
    if result['success_rate'] > 0.60:
        print("\n✓ L4-B works! End-to-end training shapes belief for control.")
    else:
        print("\n✗ L4-B failed. Need L3 (impact-consistency).")


if __name__ == '__main__':
    main()
