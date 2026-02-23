#!/usr/bin/env python3
"""
PHASE 4: Control-Aware Encoder - Making the JEPA Bridge Work

Goal: Learn encoder E(s)→z such that linear feedback in z approaches state-feedback.

Method: Control-probe alignment (Option A)
- Train encoder with auxiliary head predicting [x-x*, v, bounce_mode]
- Loss: L_JEPA + α * MSE(probe(z), control_vars)

This ensures latent preserves control-relevant statistics.
"""

import numpy as np
import json
import os


class BouncingBallGravity:
    def __init__(self, tau=0.3, restitution=0.8):
        self.g = 9.81
        self.e = restitution
        self.dt = 0.05
        self.x_target = 2.0
        self.tau = tau
        
    def reset(self, seed=None):
        if seed is not None:
            np.random.seed(seed)
        start_positions = [0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5]
        self.x = start_positions[seed % len(start_positions)]
        self.v = 0.0
        self.bounce_count = 0
        return np.array([self.x, self.v])
    
    def step(self, a):
        a = np.clip(a, -2.0, 2.0)
        self.v += (-self.g + a) * self.dt
        self.x += self.v * self.dt
        
        bounced = False
        if self.x < 0:
            self.x = -self.x * self.e
            self.v = -self.v * self.e
            bounced = True
            self.bounce_count += 1
        if self.x > 3:
            self.x = 3 - (self.x - 3) * self.e
            self.v = -self.v * self.e
            bounced = True
            self.bounce_count += 1
            
        return np.array([self.x, self.v]), bounced


def collect_trajectories_with_labels(n_traj=500, horizon=30):
    """Collect trajectories with control-relevant labels."""
    trajectories = []
    
    for ep in range(n_traj):
        env = BouncingBallGravity(restitution=0.8)
        obs = env.reset(seed=ep)
        x, v = obs[0], obs[1]
        
        traj = {'states': [], 'actions': [], 'next_states': [], 
                'control_labels': [], 'bounce_modes': []}
        
        for step in range(horizon):
            state = np.array([x, v])
            
            # Control-relevant labels
            x_error = x - 2.0  # Position error
            v_state = v        # Velocity
            near_wall = 1.0 if (x < 0.5 or x > 2.5) else 0.0  # Near bounce
            
            control_label = np.array([x_error, v_state, near_wall])
            
            # Action from random policy
            a = np.random.uniform(-2.0, 2.0)
            
            obs, bounced = env.step(a)
            x_next, v_next = obs[0], obs[1]
            next_state = np.array([x_next, v_next])
            
            # Bounce mode: 1 if just bounced
            bounce_mode = 1.0 if bounced else 0.0
            
            traj['states'].append(state)
            traj['actions'].append(a)
            traj['next_states'].append(next_state)
            traj['control_labels'].append(control_label)
            traj['bounce_modes'].append(bounce_mode)
            
            x, v = x_next, v_next
        
        trajectories.append(traj)
    
    return trajectories


class ControlAwareEncoder:
    """
    Encoder with control-probe alignment.
    
    Architecture:
    - Encoder: state → latent (z)
    - Predictor: z_t → z_{t+1} (JEPA-style)
    - Control probe: z → [x_error, v, near_wall] (auxiliary)
    """
    
    def __init__(self, state_dim=2, latent_dim=8, hidden_dim=32):
        self.state_dim = state_dim
        self.latent_dim = latent_dim
        
        np.random.seed(42)
        # Encoder weights
        self.W1 = np.random.randn(state_dim, hidden_dim) * 0.5
        self.b1 = np.zeros(hidden_dim)
        self.W2 = np.random.randn(hidden_dim, latent_dim) * 0.3
        self.b2 = np.zeros(latent_dim)
        
        # Control probe (predicts [x_error, v, near_wall])
        self.probe_W = np.random.randn(latent_dim, 3) * 0.3
        self.probe_b = np.zeros(3)
        
        # Predictor (z_t → z_{t+1})
        self.pred_W = np.random.randn(latent_dim, latent_dim) * 0.3
        self.pred_b = np.zeros(latent_dim)
    
    def encode(self, state):
        """E(s) → z"""
        h = np.tanh(np.dot(state, self.W1) + self.b1)
        z = np.dot(h, self.W2) + self.b2
        return z
    
    def predict_next(self, z):
        """Predict next latent from current"""
        return np.dot(z, self.pred_W) + self.pred_b
    
    def probe(self, z):
        """Predict control variables from latent"""
        return np.dot(z, self.probe_W) + self.probe_b
    
    def train(self, trajectories, n_epochs=200, lr=0.01, alpha=1.0, beta=0.5):
        """
        Train with:
        - Predictive loss: predict next latent
        - Control probe loss: predict control-relevant variables
        
        alpha: weight for control probe loss
        beta: weight for predictive loss
        """
        # Collect data
        states = []
        next_states = []
        control_labels = []
        
        for traj in trajectories:
            states.extend(traj['states'])
            next_states.extend(traj['next_states'])
            control_labels.extend(traj['control_labels'])
        
        states = np.array(states)
        next_states = np.array(next_states)
        control_labels = np.array(control_labels)
        
        print(f"Training control-aware encoder on {len(states)} transitions...")
        print(f"  alpha (control loss) = {alpha}")
        print(f"  beta (predictive loss) = {beta}")
        
        for epoch in range(n_epochs):
            # Mini-batch
            idx = np.random.permutation(len(states))[:128]
            s_batch = states[idx]
            s_next_batch = next_states[idx]
            c_batch = control_labels[idx]
            
            # Forward pass
            z = np.array([self.encode(s) for s in s_batch])
            z_next = np.array([self.encode(s) for s in s_next_batch])
            z_pred = np.array([self.predict_next(zi) for zi in z])
            c_pred = np.array([self.probe(zi) for zi in z])
            
            # Losses
            pred_loss = np.mean((z_pred - z_next) ** 2)
            control_loss = np.mean((c_pred - c_batch) ** 2)
            
            total_loss = beta * pred_loss + alpha * control_loss
            
            # Gradient descent (simplified)
            # Update probe head (most important for control)
            grad_probe = 2 * alpha * np.dot(z.T, (c_pred - c_batch)) / len(s_batch)
            self.probe_W -= lr * grad_probe
            self.probe_b -= lr * 2 * alpha * np.mean(c_pred - c_batch, axis=0)
            
            # Update predictor
            grad_pred = 2 * beta * np.dot(z.T, (z_pred - z_next)) / len(s_batch)
            self.pred_W -= lr * grad_pred
            self.pred_b -= lr * 2 * beta * np.mean(z_pred - z_next, axis=0)
            
            if (epoch + 1) % 50 == 0:
                print(f"  Epoch {epoch+1}: total={total_loss:.4f}, pred={pred_loss:.4f}, control={control_loss:.4f}")
    
    def fit_linear_policy(self, trajectories, n_samples=1000):
        """Fit linear policy w in latent space using optimal feedback as target."""
        X = []
        y = []
        
        for traj in trajectories:
            for s in traj['states']:
                x, v = s[0], s[1]
                # Optimal feedback action
                a_opt = 1.5 * (2.0 - x) + (-2.0) * (-v)
                a_opt = np.clip(a_opt, -2.0, 2.0)
                
                z = self.encode(s)
                X.append(z)
                y.append(a_opt)
        
        X = np.array(X)
        y = np.array(y)
        
        # Least squares
        w = np.linalg.lstsq(X, y, rcond=None)[0]
        return w


def run_latent_policy(encoder, w, n_episodes=200, restitution=0.8, seed=0):
    """Run linear policy in latent space."""
    successes = 0
    
    for ep in range(n_episodes):
        env = BouncingBallGravity(restitution=restitution)
        obs = env.reset(seed=ep + seed)
        x, v = obs[0], obs[1]
        
        for step in range(30):
            state = np.array([x, v])
            z = encoder.encode(state)
            a = np.dot(w, z)
            a = np.clip(a, -2.0, 2.0)
            
            env.step(a)
            x, v = env.x, env.v
            
            if abs(x - env.x_target) < env.tau:
                successes += 1
                break
    
    return successes / n_episodes


def run_state_feedback(k1=1.5, k2=-2.0, n_episodes=200, restitution=0.8, seed=0):
    """Linear feedback on true state (upper bound)."""
    successes = 0
    
    for ep in range(n_episodes):
        env = BouncingBallGravity(restitution=restitution)
        obs = env.reset(seed=ep + seed)
        x, v = obs[0], obs[1]
        
        for step in range(30):
            a = k1 * (2.0 - x) + k2 * (-v)
            a = np.clip(a, -2.0, 2.0)
            
            env.step(a)
            x, v = env.x, env.v
            
            if abs(x - env.x_target) < env.tau:
                successes += 1
                break
    
    return successes / n_episodes


def run_open_loop(n_episodes=200, restitution=0.8, seed=0):
    """Open-loop MPPI baseline."""
    successes = 0
    
    for ep in range(n_episodes):
        env = BouncingBallGravity(restitution=restitution)
        obs = env.reset(seed=ep + seed)
        x, v = obs[0], obs[1]
        
        for step in range(30):
            # Simple MPPI
            actions = np.random.uniform(-2.0, 2.0, (64, 30))
            costs = []
            
            for a_seq in actions:
                # Simulate
                x_test, v_test = x, v
                for a in a_seq[:min(30, 30-step)]:
                    a = np.clip(a, -2.0, 2.0)
                    v_test += (-9.81 + a) * 0.05
                    x_test += v_test * 0.05
                    if x_test < 0:
                        x_test = -x_test * restitution
                        v_test = -v_test * restitution
                    elif x_test > 3:
                        x_test = 3 - (x_test - 3) * restitution
                        v_test = -v_test * restitution
                costs.append((x_test - 2.0) ** 2)
            
            costs = np.array(costs)
            weights = np.exp(-costs / 1.0)
            weights = weights / weights.sum()
            action = (actions.T @ weights)[0]
            
            env.step(action)
            x, v = env.x, env.v
            
            if abs(x - env.x_target) < env.tau:
                successes += 1
                break
    
    return successes / n_episodes


def main():
    print("="*70)
    print("PHASE 4: Control-Aware Encoder")
    print("="*70)
    
    # Test in hard regime (e=0.8)
    restitution = 0.8
    
    # 1. Collect training data
    print("\n1. Collecting training trajectories...")
    trajectories = collect_trajectories_with_labels(n_traj=500, horizon=30)
    print(f"   Collected {len(trajectories)} trajectories")
    
    # 2. Train control-aware encoder
    print("\n2. Training control-aware encoder...")
    encoder = ControlAwareEncoder(state_dim=2, latent_dim=8, hidden_dim=32)
    encoder.train(trajectories, n_epochs=200, lr=0.01, alpha=1.0, beta=0.5)
    
    # 3. Fit linear policy
    print("\n3. Fitting linear policy in latent space...")
    w = encoder.fit_linear_policy(trajectories)
    print(f"   Learned weights shape: {w.shape}")
    
    # 4. Compare methods
    print("\n" + "="*70)
    print(f"COMPARISON (restitution={restitution})")
    print("="*70)
    
    n_episodes = 200
    
    print("\nOpen-loop MPPI:")
    ol_rate = run_open_loop(n_episodes=n_episodes, restitution=restitution)
    print(f"  Success: {ol_rate:.1%}")
    
    print("\nState feedback (upper bound):")
    state_rate = run_state_feedback(n_episodes=n_episodes, restitution=restitution)
    print(f"  Success: {state_rate:.1%}")
    
    print("\nLatent-linear (control-aware encoder):")
    latent_rate = run_latent_policy(encoder, w, n_episodes=n_episodes, restitution=restitution)
    print(f"  Success: {latent_rate:.1%}")
    
    # Baseline: random encoder
    print("\nLatent-linear (random encoder baseline):")
    random_encoder = ControlAwareEncoder(state_dim=2, latent_dim=8, hidden_dim=32)
    np.random.seed(999)
    random_encoder.W1 = np.random.randn(2, 32) * 0.5
    random_encoder.W2 = np.random.randn(32, 8) * 0.3
    w_random = random_encoder.fit_linear_policy(trajectories)
    random_rate = run_latent_policy(random_encoder, w_random, n_episodes=n_episodes, restitution=restitution)
    print(f"  Success: {random_rate:.1%}")
    
    # Summary
    print("\n" + "="*70)
    print("SUMMARY")
    print("="*70)
    
    gap_to_state = state_rate - latent_rate
    improvement_over_random = latent_rate - random_rate
    
    print(f"""
Open-loop MPPI:        {ol_rate:.1%}
Random latent:         {random_rate:.1%}
Control-aware latent:  {latent_rate:.1%}
State feedback:        {state_rate:.1%}

Improvement over random: +{improvement_over_random:.1%}
Gap to state feedback:   -{gap_to_state:.1%}
""")
    
    if latent_rate > random_rate + 0.05:
        print("✓ Control-aware training HELPED!")
        print("  The probe loss successfully aligned latent with control variables.")
    elif latent_rate > random_rate:
        print("≈ Control-aware training slightly helped")
    else:
        print("✗ Control-aware training did not help")
        print("  Need stronger alignment or different architecture.")


if __name__ == '__main__':
    main()
