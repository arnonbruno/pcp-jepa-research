#!/usr/bin/env python3
"""
PHASE 3: Knot Parameterization (Step B)

Replace H=30 independent actions with K knots + interpolation

K ∈ {3, 5, 10, 15}
Compare: MPPI-knots vs MPPI-flat
"""

import numpy as np
import json
import os
from scipy import interpolate


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
    """Execute action sequence and return final state and cost."""
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


def knots_to_horizon(knot_actions, horizon):
    """Interpolate K knots to H actions using piecewise constant."""
    k = len(knot_actions)
    
    if k >= horizon:
        return np.array(knot_actions[:horizon])
    
    # Piecewise constant: each knot controls (horizon/k) steps
    actions = np.zeros(horizon)
    steps_per_knot = horizon // k
    
    for i in range(k):
        start = i * steps_per_knot
        end = min((i + 1) * steps_per_knot, horizon)
        actions[start:end] = knot_actions[i]
    
    return actions


def mppi_knots(x, v, horizon=30, n_samples=64, temperature=1.0, 
               n_knots=5, action_scale=2.0):
    """MPPI with knot parameterization."""
    
    # Sample knot actions
    knot_actions = np.random.uniform(-action_scale, action_scale, (n_samples, n_knots))
    
    costs = []
    for i in range(n_samples):
        # Convert knots to full horizon actions
        full_actions = knots_to_horizon(knot_actions[i], horizon)
        _, total_cost = true_physics_rollout(x, v, full_actions)
        costs.append(total_cost)
    
    costs = np.array(costs)
    
    min_cost = costs.min()
    costs_shifted = costs - min_cost
    weights = np.exp(-costs_shifted / temperature)
    weights = weights / weights.sum()
    
    # Weighted combination of knots
    knot_result = knot_actions.T @ weights
    first_action = knot_result[0]
    
    return first_action, knot_result


def mppi_flat(x, v, horizon=30, n_samples=64, temperature=1.0, action_scale=2.0):
    """Standard MPPI with flat action parameterization."""
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
    return actions_weighted[0], actions_weighted


def run_episode_knots(n_knots, horizon=30, n_samples=64, temperature=1.0, seed=0):
    env = BouncingBallGravity()
    obs = env.reset(seed=seed)
    x, v = obs[0], obs[1]
    
    max_steps = min(horizon, 30)
    
    for step in range(max_steps):
        action, _ = mppi_knots(x, v, horizon=horizon, n_samples=n_samples, 
                              temperature=temperature, n_knots=n_knots)
        obs = env.step(action)
        x, v = obs[0], obs[1]
        
        if abs(x - env.x_target) < env.tau:
            return {'success': True, 'miss': abs(x - env.x_target)}
    
    return {'success': False, 'miss': abs(x - env.x_target)}


def run_episode_flat(horizon=30, n_samples=64, temperature=1.0, seed=0):
    env = BouncingBallGravity()
    obs = env.reset(seed=seed)
    x, v = obs[0], obs[1]
    
    max_steps = min(horizon, 30)
    
    for step in range(max_steps):
        action, _ = mppi_flat(x, v, horizon=horizon, n_samples=n_samples, 
                             temperature=temperature)
        obs = env.step(action)
        x, v = obs[0], obs[1]
        
        if abs(x - env.x_target) < env.tau:
            return {'success': True, 'miss': abs(x - env.x_target)}
    
    return {'success': False, 'miss': abs(x - env.x_target)}


def main():
    print("="*70)
    print("KNOT PARAMETERIZATION: MPPI with K knots")
    print("="*70)
    
    n_episodes = 200
    horizon = 30
    n_samples = 64
    
    # Test different knot counts
    knot_values = [3, 5, 10, 15, 30]  # 30 = flat (baseline)
    
    results = {}
    
    # First: flat baseline
    print(f"\nFlat (K=30) baseline...", end=" ", flush=True)
    successes = 0
    misses = []
    
    for ep in range(n_episodes):
        result = run_episode_flat(horizon=horizon, n_samples=n_samples, seed=ep)
        if result['success']:
            successes += 1
        misses.append(result['miss'])
    
    results['flat'] = {
        'success_rate': successes / n_episodes,
        'miss_mean': np.mean(misses),
        'miss_median': np.median(misses),
    }
    print(f"Success: {results['flat']['success_rate']:.1%}")
    
    # Now test knot values
    for k in [3, 5, 10, 15]:
        print(f"\nK={k} knots...", end=" ", flush=True)
        
        successes = 0
        misses = []
        
        for ep in range(n_episodes):
            result = run_episode_knots(n_knots=k, horizon=horizon, n_samples=n_samples, seed=ep)
            if result['success']:
                successes += 1
            misses.append(result['miss'])
        
        results[f'k{k}'] = {
            'success_rate': successes / n_episodes,
            'miss_mean': np.mean(misses),
            'miss_median': np.median(misses),
        }
        
        print(f"Success: {results[f'k{k}']['success_rate']:.1%}, Miss: {results[f'k{k}']['miss_mean']:.3f}")
    
    # Summary
    print("\n" + "="*70)
    print("SUMMARY")
    print("="*70)
    print(f"\n{'K':<8} {'Success':<12} {'Miss Mean':<12} {'Miss Median':<12}")
    print("-"*45)
    
    print(f"{'Flat':<8} {results['flat']['success_rate']:>8.1%}   {results['flat']['miss_mean']:>9.3f}   {results['flat']['miss_median']:>9.3f}")
    for k in [3, 5, 10, 15]:
        r = results[f'k{k}']
        print(f"{k:<8} {r['success_rate']:>8.1%}   {r['miss_mean']:>9.3f}   {r['miss_median']:>9.3f}")
    
    # Save
    output_dir = '/home/ulluboz/pcp-jepa-research/results/phase3'
    os.makedirs(output_dir, exist_ok=True)
    
    with open(f'{output_dir}/knot_sweep.json', 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"\nSaved to {output_dir}/knot_sweep.json")
    
    # Interpretation
    print("\n" + "="*70)
    print("INTERPRETATION")
    print("="*70)
    
    flat_rate = results['flat']['success_rate']
    best_knot = max([k for k in [3, 5, 10, 15]], 
                   key=lambda x: results[f'k{x}']['success_rate'])
    best_rate = results[f'k{best_knot}']['success_rate']
    
    if best_rate > flat_rate + 0.05:
        print(f"\n✓ Knots IMPROVE success: {flat_rate:.1%} (flat) → {best_rate:.1%} (K={best_knot})")
        print("  → Action dimensionality IS the bottleneck; knots help!")
    elif best_rate < flat_rate - 0.05:
        print(f"\n✗ Knots REDUCE success: {flat_rate:.1%} → {best_rate:.1%}")
        print("  → Knot parameterization loses too much flexibility")
    else:
        print(f"\n≈ Knots have MINIMAL effect")
        print("  → Action parameterization is NOT the bottleneck")


if __name__ == '__main__':
    main()
