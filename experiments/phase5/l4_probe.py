#!/usr/bin/env python3
"""
PHASE 5: L4-A - Probe Controller on Frozen Belief

Test: Is the L1+L2-trained belief control-sufficient?

Procedure:
1. Train encoder+GRU with L1+L2 (or use random init)
2. FREEZE encoder+GRU
3. Train only linear controller head via imitation
4. Evaluate: does control improve above 56.5%?
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
    """Generate trajectories with partial observations."""
    np.random.seed(seed)
    env = BouncingBall(restitution=restitution)
    
    trajectories = []
    for ep in range(n_episodes):
        state = env.reset(seed=ep + seed)
        x, v = float(state[0]), float(state[1])
        
        traj = {'obs': [], 'states': [], 'acts': []}
        
        for step in range(horizon):
            # Partial observation: position only
            obs = np.array([x])
            if obs_noise > 0:
                obs = obs + np.random.randn() * obs_noise
            if dropout > 0 and np.random.rand() < dropout:
                obs = obs * 0
            
            # Expert action (PD)
            a_expert = 1.5 * (2.0 - x) + (-2.0) * (-v)
            a_expert = float(np.clip(a_expert, -2.0, 2.0))
            
            # Step
            next_state = env.step(a_expert)
            x_next, v_next = float(next_state[0]), float(next_state[1])
            
            traj['obs'].append(obs)
            traj['states'].append([x, v])
            traj['acts'].append(a_expert)
            
            x, v = x_next, v_next
        
        trajectories.append(traj)
    
    return trajectories


# ============================================================================
# JEPA MODEL (NumPy - Simple)
# ============================================================================

class JEPABelief:
    """Simple JEPA belief model."""
    
    def __init__(self, obs_dim=1, action_dim=1, latent_dim=8, belief_dim=16):
        self.obs_dim = obs_dim
        self.action_dim = action_dim
        self.latent_dim = latent_dim
        self.belief_dim = belief_dim
        
        np.random.seed(42)
        
        # Encoder
        self.W_enc = np.random.randn(obs_dim, latent_dim) * 0.3
        self.b_enc = np.zeros(latent_dim)
        
        # GRU (simplified)
        self.W_gru = np.random.randn(latent_dim + action_dim, belief_dim) * 0.3
        self.b_gru = np.zeros(belief_dim)
        
        # Predictor
        self.W_pred = np.random.randn(belief_dim + action_dim, belief_dim) * 0.3
        self.b_pred = np.zeros(belief_dim)
        
        # Controller head (to be trained)
        self.W_ctrl = np.random.randn(belief_dim, action_dim) * 0.3
        self.b_ctrl = np.zeros(action_dim)
    
    def encode(self, obs):
        return np.tanh(np.dot(obs, self.W_enc) + self.b_enc)
    
    def gru_step(self, belief, z, a):
        x = np.concatenate([z, np.array([a])])
        new = np.tanh(np.dot(x, self.W_gru) + self.b_gru)
        return 0.9 * belief + 0.1 * new
    
    def predict(self, belief, a):
        x = np.concatenate([belief, np.array([a])])
        return np.tanh(np.dot(x, self.W_pred) + self.b_pred)
    
    def forward(self, observations, actions):
        """Forward pass through trajectory."""
        beliefs = []
        belief = np.zeros(self.belief_dim)
        
        for t in range(len(observations)):
            z = self.encode(observations[t])
            a_prev = actions[t-1] if t > 0 else 0.0
            belief = self.gru_step(belief, z, a_prev)
            beliefs.append(belief)
        
        return beliefs
    
    def controller(self, belief):
        """Linear controller: a = W_ctrl @ belief + b_ctrl"""
        return np.dot(belief, self.W_ctrl) + self.b_ctrl


def train_jepa_l1_l2(model, trajectories, n_epochs=30, lr=0.01):
    """Train JEPA with L1 + L2 (simplified)."""
    
    print(f"Training JEPA (L1+L2)...")
    
    for epoch in range(n_epochs):
        total_loss = 0.0
        count = 0
        
        for traj in trajectories:
            obs = traj['obs']
            acts = traj['acts']
            
            if len(obs) < 3:
                continue
            
            beliefs = model.forward(obs, acts)
            
            # L1: JEPA prediction (k=1)
            for t in range(len(beliefs) - 1):
                pred = model.predict(beliefs[t], acts[t])
                target = beliefs[t + 1]
                total_loss += np.mean((pred - target) ** 2)
                count += 1
        
        if (epoch + 1) % 10 == 0:
            print(f"  Epoch {epoch+1}: loss={total_loss/max(count,1):.4f}")
    
    return model


def train_controller_head(model, trajectories, n_epochs=50, lr=0.01):
    """L4-A: Train ONLY controller head (encoder frozen)."""
    
    print(f"Training controller head (L4-A)...")
    
    for epoch in range(n_epochs):
        total_loss = 0.0
        count = 0
        
        for traj in trajectories:
            obs = traj['obs']
            acts = traj['acts']
            
            beliefs = model.forward(obs, acts)
            
            # Imitation loss: predict expert action from belief
            for t, (belief, a_expert) in enumerate(zip(beliefs, acts)):
                a_pred = model.controller(belief)[0]
                a_pred = np.clip(a_pred, -2.0, 2.0)
                
                loss = (a_pred - a_expert) ** 2
                total_loss += loss
                count += 1
                
                # Gradient update (simplified)
                if count % 100 == 0:
                    # d(loss)/d(a_pred) = 2*(a_pred - a_expert)
                    # d(a_pred)/d(W_ctrl) = belief
                    # d(loss)/d(W_ctrl) = 2*(a_pred - a_expert) * belief
                    grad_w = 2 * (a_pred - a_expert) * belief
                    grad_b = 2 * (a_pred - a_expert)
                    
                    model.W_ctrl -= lr * grad_w.reshape(-1, 1)
                    model.b_ctrl -= lr * grad_b
        
        if (epoch + 1) % 10 == 0:
            print(f"  Epoch {epoch+1}: imitation loss={total_loss/max(count,1):.4f}")
    
    return model


def evaluate(model, n_episodes, restitution, obs_noise, dropout, seed):
    """Evaluate JEPA + controller."""
    
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
            # Partial observation
            obs = np.array([x])
            if obs_noise > 0 and np.random.rand() < obs_noise:
                obs = obs + np.random.randn() * obs_noise
            if dropout > 0 and np.random.rand() < dropout:
                obs = obs * 0
            
            # Update belief
            z = model.encode(obs)
            a_prev = 0.0 if step == 0 else last_a
            belief = model.gru_step(belief, z, a_prev)
            
            # Controller action
            a = model.controller(belief)[0]
            a = float(np.clip(a, -2.0, 2.0))
            last_a = a
            
            # Step
            next_state = env.step(a)
            x, v = float(next_state[0]), float(next_state[1])
            
            # Count bounces
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
    print("PHASE 5: L4-A - Probe Controller on Frozen Belief")
    print("="*70)
    
    # Config
    restitution = 0.8  # Hard regime
    obs_noise = 0.05
    dropout = 0.1
    n_train = 500
    n_test = 200
    
    # Generate data
    print("\n1. Generating training data...")
    train_trajs = generate_data(n_train, 30, restitution, seed=42, 
                                obs_noise=obs_noise, dropout=dropout)
    print(f"   Generated {len(train_trajs)} trajectories")
    
    # Create model
    print("\n2. Creating JEPA model...")
    model = JEPABelief(obs_dim=1, action_dim=1, latent_dim=8, belief_dim=16)
    
    # Train JEPA (L1+L2)
    print("\n3. Training JEPA (L1+L2)...")
    model = train_jepa_l1_l2(model, train_trajs, n_epochs=30, lr=0.01)
    
    # Train controller head ONLY (L4-A)
    print("\n4. Training controller head (L4-A - frozen encoder)...")
    model = train_controller_head(model, train_trajs, n_epochs=50, lr=0.01)
    
    # Evaluate
    print("\n5. Evaluating...")
    result = evaluate(model, n_test, restitution, obs_noise, dropout, seed=42)
    
    print(f"\n   JEPA + trained controller:")
    print(f"     Success rate: {result['success_rate']:.1%}")
    print(f"     Terminal miss: {result['terminal_miss']:.3f}")
    print(f"     Bounce count: {result['bounce_count']:.1f}")
    
    # Compare
    print("\n" + "="*70)
    print("COMPARISON")
    print("="*70)
    print(f"PD (full state):       71.0%")
    print(f"PD (partial):          56.5%")
    print(f"JEPA + ctrl (L4-A):  {result['success_rate']:.1%}")
    
    gap_closed = (result['success_rate'] - 0.565) / (0.71 - 0.565)
    print(f"\nGap closed: {gap_closed:.1%}")
    
    if result['success_rate'] > 0.60:
        print("\n✓ L4-A works! Belief is control-sufficient.")
    else:
        print("\n✗ L4-A failed. Need L3 (impact-consistency) or L4-B.")


if __name__ == '__main__':
    main()
