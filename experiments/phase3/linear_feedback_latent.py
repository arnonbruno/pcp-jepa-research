#!/usr/bin/env python3
"""
PHASE 3: Linear Feedback in Latent Space (JEPA Connection)

Test: Does a linear policy in latent space work as well as gains in raw space?

1. Raw state feedback: a = w^T * [x-x*, v] (dim=2)
2. Latent space feedback: a = w^T * z (dim=latent_dim)
"""

import numpy as np
import json
import os


class BouncingBallGravity:
    def __init__(self, tau=0.3):
        self.g = 9.81
        self.e = 0.8
        self.dt = 0.05
        self.x_target = 2.0
        self.tau = tau
        
    def reset(self, seed=None):
        if seed is not None:
            np.random.seed(seed)
        start_positions = [0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5]
        self.x = start_positions[seed % len(start_positions)]
        self.v = 0.0
        return np.array([self.x, self.v])
    
    def step(self, a):
        self.v += (-self.g + a) * self.dt
        self.x += self.v * self.dt
        
        if self.x < 0:
            self.x = -self.x * self.e
            self.v = -self.v * self.e
        if self.x > 3:
            self.x = 3 - (self.x - 3) * self.e
            self.v = -self.v * self.e
            
        return np.array([self.x, self.v])


def simulate_policy(x0, v0, w, phi_fn, horizon=30, action_scale=2.0):
    """Execute policy: a = clip(w^T * phi(s), bounds)."""
    x, v = x0, v0
    total_cost = 0
    
    for _ in range(horizon):
        phi = phi_fn(x, v)
        a = np.dot(w, phi)
        a = np.clip(a, -action_scale, action_scale)
        
        # Physics
        v += (-9.81 + a) * 0.05
        x += v * 0.05
        
        if x < 0:
            x = -x * 0.8
            v = -v * 0.8
        elif x > 3:
            x = 3 - (x - 3) * 0.8
            v = -v * 0.8
        
        total_cost += (x - 2.0) ** 2
    
    return x, total_cost


def run_linear_feedback(weights, phi_fn, n_samples=0, temperature=0, seed=0):
    """Run episode with linear feedback policy."""
    env = BouncingBallGravity()
    obs = env.reset(seed=seed)
    x, v = obs[0], obs[1]
    
    for step in range(30):
        phi = phi_fn(x, v)
        a = np.dot(weights, phi)
        a = np.clip(a, -2.0, 2.0)
        
        obs = env.step(a)
        x, v = obs[0], obs[1]
        
        if abs(x - env.x_target) < env.tau:
            return True
    return False


def run_linear_feedback_mppi(phi_fn, dim, n_samples=500, temperature=1.0, seed=0):
    """MPPI over linear feedback weights."""
    env = BouncingBallGravity()
    obs = env.reset(seed=seed)
    x, v = obs[0], obs[1]
    
    for step in range(30):
        # Sample weight vectors
        weights_samples = np.random.uniform(-3.0, 3.0, (n_samples, dim))
        
        costs = []
        for i in range(n_samples):
            final_x, cost = simulate_policy(x, v, weights_samples[i], phi_fn)
            costs.append(cost)
        
        costs = np.array(costs)
        min_cost = costs.min()
        weights = np.exp(-(costs - min_cost) / temperature)
        weights = weights / weights.sum()
        
        # Best weights
        w = (weights_samples.T @ weights)
        
        # Execute
        phi = phi_fn(x, v)
        a = np.clip(np.dot(w, phi), -2.0, 2.0)
        
        obs = env.step(a)
        x, v = obs[0], obs[1]
        
        if abs(x - env.x_target) < env.tau:
            return True
    return False


def main():
    print("="*70)
    print("LINEAR FEEDBACK IN RAW vs LATENT SPACE")
    print("="*70)
    
    n_episodes = 200
    
    # 1. Raw state feedback: phi(s) = [x-x*, v]
    print("\n1. Raw state feedback (dim=2): phi = [x-x*, v]")
    print("   This is exactly PD control")
    
    def phi_raw(x, v):
        return np.array([x - 2.0, v])
    
    # Fixed optimal gains
    w_raw = np.array([1.5, -2.0])  # k1=1.5, k2=-2.0 from earlier
    successes = sum(run_linear_feedback(w_raw, phi_raw, seed=ep) for ep in range(n_episodes))
    print(f"   Fixed weights (1.5, -2.0): {successes/n_episodes:.1%}")
    
    # 2. Enhanced raw features: phi(s) = [x-x*, v, sign(x-1.5), x^2]
    print("\n2. Enhanced raw features (dim=4): phi = [x-x*, v, x^2, sign(x-1.5)]")
    
    def phi_enhanced(x, v):
        return np.array([x - 2.0, v, x**2, np.sign(x - 1.5)])
    
    # MPPI over enhanced features
    successes = sum(run_linear_feedback_mppi(phi_enhanced, dim=4, seed=ep) for ep in range(n_episodes))
    print(f"   MPPI over dim=4: {successes/n_episodes:.1%}")
    
    # 3. Simulated latent space (random encoder)
    print("\n3. Simulated latent space (dim=8): phi = E(s) where E is random MLP")
    
    # Create random encoder
    np.random.seed(42)
    W_enc = np.random.randn(2, 8) * 0.5
    b_enc = np.random.randn(8) * 0.1
    
    def phi_latent(x, v):
        h = np.tanh(np.dot([x, v], W_enc) + b_enc)
        return h
    
    successes = sum(run_linear_feedback_mppi(phi_latent, dim=8, seed=ep) for ep in range(n_episodes))
    print(f"   MPPI over dim=8 latent: {successes/n_episodes:.1%}")
    
    # 4. Simulated JEPA-trained latent (structured encoder)
    print("\n4. Structured latent (dim=4): phi encodes bounce phase")
    
    # This simulates what JEPA might learn: phase + velocity
    def phi_structured(x, v):
        # Phase: 0 = going left, 1 = going right
        phase = 0.0 if v > 0 else 1.0
        # Distance to nearest wall
        dist_left = x
        dist_right = 3 - x
        # Return structured features
        return np.array([x - 2.0, v, phase, min(dist_left, dist_right)])
    
    successes = sum(run_linear_feedback_mppi(phi_structured, dim=4, seed=ep) for ep in range(n_episodes))
    print(f"   MPPI over dim=4 structured: {successes/n_episodes:.1%}")
    
    # 5. Just linear (no features)
    print("\n5. Linear in raw (dim=2): MPPI over weights")
    
    successes = sum(run_linear_feedback_mppi(phi_raw, dim=2, seed=ep) for ep in range(n_episodes))
    print(f"   MPPI over dim=2: {successes/n_episodes:.1%}")
    
    # Summary
    print("\n" + "="*70)
    print("SUMMARY")
    print("="*70)
    print("""
Key insight: The "gains" that work (k1=1.5, k2=-2.0) are optimal 
for the specific feature basis phi(s) = [x-x*, v].

If we change the feature basis, we need to re-optimize weights.
The question is: can a learned latent provide a better basis?
""")


if __name__ == '__main__':
    main()
