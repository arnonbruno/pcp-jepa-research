#!/usr/bin/env python3
"""
PHASE 3: Stick-Slip System - Replication Test

Goal: Test if "open-loop plateau vs feedback structure" generalizes
to another discontinuous system.

Stick-slip dynamics:
- State: [x, v]
- Action: force u ∈ [-u_max, u_max]
- Event: stick ↔ slip transition (velocity crosses zero with friction)
"""

import numpy as np
import json
import os


class StickSlipBlock:
    """
    Stick-slip friction dynamics.
    
    Stick: |v| < v_thresh and |u| < F_static → v = 0
    Slip: |v| >= v_thresh or |u| >= F_static → sliding with kinetic friction
    """
    
    def __init__(self, tau=0.2):
        self.dt = 0.02
        self.x_target = 2.0
        self.tau = tau
        
        # Friction parameters
        self.F_static = 1.5
        self.F_kinetic = 1.0
        self.v_thresh = 0.01
        self.mass = 1.0
        
    def reset(self, seed=None):
        if seed is not None:
            np.random.seed(seed)
        # Start positions
        start_positions = [0.0, 0.5, 1.0, 1.5, 2.5, 3.0]
        self.x = start_positions[seed % len(start_positions)]
        self.v = 0.0
        self.mode = 'stick'  # or 'slip'
        return np.array([self.x, self.v])
    
    def step(self, u):
        # Clip action
        u = np.clip(u, -3.0, 3.0)
        
        # Check for mode transition
        if self.mode == 'stick':
            # Check if force overcomes static friction
            if abs(u) > self.F_static:
                self.mode = 'slip'
                self.v = np.sign(u) * 0.01  # Small initial velocity
        else:
            # Check if velocity crosses zero (stick transition)
            if abs(self.v) < self.v_thresh and abs(u) < self.F_static:
                self.mode = 'stick'
                self.v = 0.0
        
        # Update dynamics
        if self.mode == 'stick':
            # No motion
            pass
        else:
            # Sliding with kinetic friction
            friction = -np.sign(self.v) * self.F_kinetic
            a = (u + friction) / self.mass
            self.v += a * self.dt
            
            # Position update
            self.x += self.v * self.dt
        
        # Boundaries
        self.x = np.clip(self.x, -1.0, 4.0)
        if self.x <= -1.0 or self.x >= 4.0:
            self.v = 0.0
        
        return np.array([self.x, self.v])


def simulate_rollout(x0, v0, actions, horizon=50):
    """Simulate action sequence."""
    env = StickSlipBlock()
    env.x = x0
    env.v = v0
    env.mode = 'stick' if abs(v0) < 0.01 else 'slip'
    
    total_cost = 0
    for a in actions:
        env.step(a)
        total_cost += (env.x - env.x_target) ** 2
    
    total_cost += 10 * (env.x - env.x_target) ** 2  # Terminal cost
    return env.x, total_cost


def run_action_sequence(horizon=50, n_samples=64, temperature=1.0, seed=0):
    """Open-loop action sequence planning."""
    env = StickSlipBlock()
    obs = env.reset(seed=seed)
    x, v = obs[0], obs[1]
    
    for step in range(50):
        # Sample action sequences
        actions = np.random.uniform(-3.0, 3.0, (n_samples, horizon))
        
        costs = []
        for i in range(n_samples):
            _, cost = simulate_rollout(x, v, actions[i], horizon=min(horizon, 50-step))
            costs.append(cost)
        
        costs = np.array(costs)
        min_cost = costs.min()
        weights = np.exp(-(costs - min_cost) / temperature)
        weights = weights / weights.sum()
        
        action = (actions.T @ weights)[0]
        env.step(action)
        x, v = env.x, env.v
        
        if abs(x - env.x_target) < env.tau:
            return True
    
    return False


def run_knots(k=5, horizon=50, n_samples=64, temperature=1.0, seed=0):
    """Knot parameterization."""
    env = StickSlipBlock()
    obs = env.reset(seed=seed)
    x, v = obs[0], obs[1]
    
    for step in range(50):
        # Sample knots
        knot_actions = np.random.uniform(-3.0, 3.0, (n_samples, k))
        
        costs = []
        for i in range(n_samples):
            # Interpolate to horizon
            full_actions = np.repeat(knot_actions[i], (50-step) // k + 1)[:50-step]
            _, cost = simulate_rollout(x, v, full_actions)
            costs.append(cost)
        
        costs = np.array(costs)
        min_cost = costs.min()
        weights = np.exp(-(costs - min_cost) / temperature)
        weights = weights / weights.sum()
        
        # Best action
        action = (knot_actions.T @ weights)[step % k]
        env.step(action)
        x, v = env.x, env.v
        
        if abs(x - env.x_target) < env.tau:
            return True
    
    return False


def run_feedback_gains(k1=2.0, k2=1.0, seed=0):
    """Fixed feedback gains: u = k1*(x*-x) + k2*(-v)."""
    env = StickSlipBlock()
    obs = env.reset(seed=seed)
    x, v = obs[0], obs[1]
    
    for step in range(50):
        # Feedback law
        u = k1 * (env.x_target - x) + k2 * (-v)
        u = np.clip(u, -3.0, 3.0)
        
        env.step(u)
        x, v = env.x, env.v
        
        if abs(x - env.x_target) < env.tau:
            return True
    
    return False


def grid_search_gains(n_episodes=50):
    """Find optimal gains via grid search."""
    k1_range = np.arange(0.5, 5.0, 0.5)
    k2_range = np.arange(-2.0, 3.0, 0.5)
    
    best_success = 0
    best_k1, best_k2 = 2.0, 1.0
    
    for k1 in k1_range:
        for k2 in k2_range:
            successes = sum(run_feedback_gains(k1, k2, ep) for ep in range(n_episodes))
            rate = successes / n_episodes
            if rate > best_success:
                best_success = rate
                best_k1, best_k2 = k1, k2
    
    return best_k1, best_k2, best_success


def main():
    print("="*70)
    print("STICK-SLIP SYSTEM: Open-Loop vs Feedback")
    print("="*70)
    
    n_episodes = 200
    
    # 1. Open-loop action sequences
    print("\n1. Action sequences (H=50)...")
    successes = sum(run_action_sequence(horizon=50, seed=ep) for ep in range(n_episodes))
    print(f"   Success: {successes/n_episodes:.1%}")
    
    # 2. Knots
    print("\n2. Knots (K=5)...")
    successes = sum(run_knots(k=5, seed=ep) for ep in range(n_episodes))
    print(f"   Success: {successes/n_episodes:.1%}")
    
    # 3. Grid search for optimal gains
    print("\n3. Grid search for optimal gains...")
    k1_opt, k2_opt, rate = grid_search_gains(n_episodes=50)
    print(f"   Best gains: k1={k1_opt:.1f}, k2={k2_opt:.1f}")
    print(f"   Success (50 eps): {rate:.1%}")
    
    # 4. Test optimal gains
    print(f"\n4. Feedback gains (k1={k1_opt:.1f}, k2={k2_opt:.1f})...")
    successes = sum(run_feedback_gains(k1_opt, k2_opt, ep) for ep in range(n_episodes))
    print(f"   Success: {successes/n_episodes:.1%}")
    
    # 5. Fixed gains baseline
    print("\n5. Fixed gains (k1=2.0, k2=1.0)...")
    successes = sum(run_feedback_gains(2.0, 1.0, ep) for ep in range(n_episodes))
    print(f"   Success: {successes/n_episodes:.1%}")
    
    # Summary
    print("\n" + "="*70)
    print("COMPARISON")
    print("="*70)
    print("""
If feedback gains > action sequences, the phenomenon generalizes!
This would confirm: representation structure (closed-loop) beats
open-loop across different hybrid systems.
""")


if __name__ == '__main__':
    main()
