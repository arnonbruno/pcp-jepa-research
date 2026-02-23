#!/usr/bin/env python3
"""
PHASE 3: Policy-Space Planning (Feedback Gains)

Instead of planning action sequences, plan feedback gains:
a_t = clip(k1*(target - x) + k2*(-v), bounds)

MPPI searches over (k1, k2) instead of H actions.
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


def true_physics_rollout_with_policy(x0, v0, k1, k2, target=2.0, g=9.81, e=0.8, dt=0.05, horizon=30):
    """Execute policy a = clip(k1*(target-x) + k2*(-v), bounds)."""
    x, v = x0, v0
    total_cost = 0
    
    for _ in range(horizon):
        # Feedback law
        a = k1 * (target - x) + k2 * (-v)
        a = np.clip(a, -2.0, 2.0)
        
        # Physics
        v += (-g + a) * dt
        x += v * dt
        
        if x < 0:
            x = -x * e
            v = -v * e
        elif x > 3:
            x = 3 - (x - 3) * e
            v = -v * e
        
        total_cost += (x - target) ** 2
    
    return x, total_cost


def mppi_feedback(x0, v0, n_samples=1000, temperature=1.0):
    """MPPI searching over feedback gains (k1, k2)."""
    # Sample gain pairs
    k1_samples = np.random.uniform(-5.0, 5.0, n_samples)
    k2_samples = np.random.uniform(-5.0, 5.0, n_samples)
    
    costs = []
    for i in range(n_samples):
        _, cost = true_physics_rollout_with_policy(x0, v0, k1_samples[i], k2_samples[i])
        costs.append(cost)
    
    costs = np.array(costs)
    
    min_cost = costs.min()
    costs_shifted = costs - min_cost
    weights = np.exp(-costs_shifted / temperature)
    weights = weights / weights.sum()
    
    # Weighted combination of gains
    k1_best = (k1_samples * weights).sum()
    k2_best = (k2_samples * weights).sum()
    
    return k1_best, k2_best, weights.max()


def run_feedback_planning(n_samples=1000, temperature=1.0, seed=0):
    """Run episode with feedback gain planning."""
    env = BouncingBallGravity()
    obs = env.reset(seed=seed)
    x, v = obs[0], obs[1]
    
    for step in range(30):
        # Plan feedback gains
        k1, k2, w_max = mppi_feedback(x, v, n_samples=n_samples, temperature=temperature)
        
        # Execute one step with planned gains
        a = k1 * (env.x_target - x) + k2 * (-v)
        a = np.clip(a, -2.0, 2.0)
        
        obs = env.step(a)
        x, v = obs[0], obs[1]
        
        if abs(x - env.x_target) < env.tau:
            return {'success': True, 'miss': abs(x - env.x_target)}
    
    return {'success': False, 'miss': abs(x - env.x_target)}


def run_p_controller(seed=0, k1=2.0, k2=0.0):
    """P controller baseline."""
    env = BouncingBallGravity()
    obs = env.reset(seed=seed)
    x, v = obs[0], obs[1]
    
    for step in range(30):
        a = k1 * (env.x_target - x) + k2 * (-v)
        a = np.clip(a, -2.0, 2.0)
        
        obs = env.step(a)
        x, v = obs[0], obs[1]
        
        if abs(x - env.x_target) < env.tau:
            return {'success': True, 'miss': abs(x - env.x_target)}
    
    return {'success': False, 'miss': abs(x - env.x_target)}


def main():
    print("="*70)
    print("FEEDBACK GAIN PLANNING")
    print("="*70)
    
    n_episodes = 200
    
    # Test 1: P controller (fixed gains)
    print("\n1. P controller (k1=2.0)...")
    successes = 0
    for ep in range(n_episodes):
        r = run_p_controller(seed=ep)
        successes += int(r['success'])
    p_success = successes / n_episodes
    print(f"   Success: {p_success:.1%}")
    
    # Test 2: PD controller (fixed gains)
    print("\n2. PD controller (k1=2.0, k2=0.5)...")
    successes = 0
    for ep in range(n_episodes):
        r = run_p_controller(seed=ep, k1=2.0, k2=0.5)
        successes += int(r['success'])
    pd_success = successes / n_episodes
    print(f"   Success: {pd_success:.1%}")
    
    # Test 3: MPPI over feedback gains
    print("\n3. MPPI over feedback gains (k1, k2)...")
    successes = 0
    for ep in range(n_episodes):
        r = run_feedback_planning(n_samples=1000, temperature=1.0, seed=ep)
        successes += int(r['success'])
    mppi_success = successes / n_episodes
    print(f"   Success: {mppi_success:.1%}")
    
    # Test 4: Grid search over gains
    print("\n4. Grid search over (k1, k2)...")
    k1_range = np.arange(-3.0, 5.0, 0.5)
    k2_range = np.arange(-2.0, 3.0, 0.5)
    
    best_success = 0
    best_k1, best_k2 = 2.0, 0.0
    
    for k1 in k1_range:
        for k2 in k2_range:
            successes = 0
            for ep in range(50):  # Quick eval
                r = run_p_controller(seed=ep, k1=k1, k2=k2)
                successes += int(r['success'])
            
            if successes > best_success:
                best_success = successes
                best_k1, best_k2 = k1, k2
    
    grid_success = best_success / 50
    print(f"   Best gains: k1={best_k1:.1f}, k2={best_k2:.1f}")
    print(f"   Success: {grid_success:.1%}")
    
    # Summary
    print("\n" + "="*70)
    print("SUMMARY")
    print("="*70)
    
    print(f"\n{'Method':<30} {'Success':<10}")
    print("-"*40)
    print(f"{'P controller (k1=2.0)':<30} {p_success:>7.1%}")
    print(f"{'PD controller (k1=2,k2=0.5)':<30} {pd_success:>7.1%}")
    print(f"{'MPPI over (k1,k2)':<30} {mppi_success:>7.1%}")
    print(f"{'Grid search best':<30} {grid_success:>7.1%}")
    
    # Save
    output_dir = '/home/ulluboz/pcp-jepa-research/results/phase3'
    os.makedirs(output_dir, exist_ok=True)
    
    results = {
        'p_controller': p_success,
        'pd_controller': pd_success,
        'mppi_feedback': mppi_success,
        'grid_search': grid_success,
        'best_k1': best_k1,
        'best_k2': best_k2,
    }
    
    with open(f'{output_dir}/feedback_gain_planning.json', 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"\nSaved to {output_dir}/feedback_gain_planning.json")


if __name__ == '__main__':
    main()
