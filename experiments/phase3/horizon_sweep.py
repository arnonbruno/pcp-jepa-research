#!/usr/bin/env python3
"""
PHASE 3: Horizon Sweep (Step A)

Test: Does success increase with horizon H?

H ∈ {10, 20, 30, 50, 100}
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


def true_physics_rollout(x, v, actions, g=9.81, e=0.8, dt=0.05):
    for a in actions:
        v += (-g + a) * dt
        x += v * dt
        
        if x < 0:
            x = -x * e
            v = -v * e
        elif x > 3:
            x = 3 - (x - 3) * e
            v = -v * e
            
    return x, (x - 2.0) ** 2


def mppi(x, v, horizon=30, n_samples=64, temperature=1.0, action_scale=2.0):
    actions = np.random.uniform(-action_scale, action_scale, (n_samples, horizon))
    
    costs = []
    for i in range(n_samples):
        _, total_cost = true_physics_rollout(x, v, actions[i])
        costs.append(total_cost)
        
    costs = np.array(costs)
    
    min_cost = costs.min()
    costs_shifted = costs - min_cost
    weights = np.exp(-costs_shifted / temperature)
    weights = weights / weights.sum()
    
    actions_weighted = actions.T @ weights
    return actions_weighted[0]


def run_episode(horizon, n_samples=64, temperature=1.0, seed=0):
    env = BouncingBallGravity()
    obs = env.reset(seed=seed)
    x, v = obs[0], obs[1]
    
    # Execute for min(horizon, 30) steps (max episode length)
    max_steps = min(horizon, 30)
    
    for step in range(max_steps):
        action = mppi(x, v, horizon=horizon, n_samples=n_samples, temperature=temperature)
        obs = env.step(action)
        x, v = obs[0], obs[1]
        
        if abs(x - env.x_target) < env.tau:
            return {'success': True, 'miss': abs(x - env.x_target), 'steps': step}
    
    return {'success': False, 'miss': abs(x - env.x_target), 'steps': max_steps}


def main():
    print("="*70)
    print("HORIZON SWEEP: MPPI Success vs Planning Horizon H")
    print("="*70)
    
    horizons = [10, 20, 30, 50, 100]
    n_episodes = 200
    
    results = {}
    
    for H in horizons:
        print(f"\nTesting H={H}...", end=" ", flush=True)
        
        successes = 0
        misses = []
        avg_steps = 0
        
        for ep in range(n_episodes):
            result = run_episode(horizon=H, n_samples=64, temperature=1.0, seed=ep)
            if result['success']:
                successes += 1
            misses.append(result['miss'])
            avg_steps += result['steps']
        
        results[H] = {
            'success_rate': successes / n_episodes,
            'miss_mean': np.mean(misses),
            'miss_median': np.median(misses),
            'miss_std': np.std(misses),
            'avg_steps': avg_steps / n_episodes,
        }
        
        print(f"Success: {results[H]['success_rate']:.1%}, Miss: {results[H]['miss_mean']:.3f}")
    
    # Summary table
    print("\n" + "="*70)
    print("SUMMARY")
    print("="*70)
    print(f"\n{'H':<6} {'Success':<10} {'Miss Mean':<12} {'Miss Median':<12} {'Avg Steps':<10}")
    print("-"*60)
    
    for H in horizons:
        r = results[H]
        print(f"{H:<6} {r['success_rate']:>7.1%}   {r['miss_mean']:>9.3f}   {r['miss_median']:>9.3f}   {r['avg_steps']:>7.1f}")
    
    # Save
    output_dir = '/home/ulluboz/pcp-jepa-research/results/phase3'
    os.makedirs(output_dir, exist_ok=True)
    
    with open(f'{output_dir}/horizon_sweep.json', 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"\nSaved to {output_dir}/horizon_sweep.json")
    
    # Interpretation
    print("\n" + "="*70)
    print("INTERPRETATION")
    print("="*70)
    
    # Check if success increases with H
    success_rates = [results[H]['success_rate'] for H in horizons]
    if success_rates[-1] > success_rates[0]:
        print(f"\n✓ Success INCREASES with horizon: {success_rates[0]:.1%} (H={horizons[0]}) → {success_rates[-1]:.1%} (H={horizons[-1]})")
        print("  → Horizon is a bottleneck; longer planning helps")
    else:
        print(f"\n✗ Success does NOT increase with horizon")
        print("  → Horizon is NOT the bottleneck")


if __name__ == '__main__':
    main()
