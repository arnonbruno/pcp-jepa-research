#!/usr/bin/env python3
"""
PHASE 3: CEM vs MPPI Comparison + Temperature × Sample Budget Sweep

1. CEM baseline: Cross-Entropy Method with identical rollout model
2. Temperature sweep: λ ∈ {0.1, 0.5, 1.0, 2.0, 5.0}
3. Sample budget sweep: N ∈ {64, 256, 1024}

Goal: Determine why MPPI has chronic low ESS - is it the weighting, sample budget, or temperature?
"""

import numpy as np
import json
import os
from itertools import product

# ============== ENVIRONMENT (WITH GRAVITY) ==============
class BouncingBallGravity:
    def __init__(self):
        self.g = 9.81
        self.e = 0.8
        self.dt = 0.05
        self.x_target = 2.0
        self.tau = 0.3
        
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


def true_physics_step(x, v, a, g=9.81, e=0.8, dt=0.05):
    v_new = v + (-g + a) * dt
    x_new = x + v_new * dt
    
    event = 0
    if x_new < 0:
        event = 1
        x_new = -x_new * e
        v_new = -v_new * e
    elif x_new > 3:
        event = 1
        x_new = 3 - (x_new - 3) * e
        v_new = -v_new * e
        
    return x_new, v_new, event


def true_physics_rollout(x, v, actions, g=9.81, e=0.8, dt=0.05):
    states = [(x, v)]
    costs = []
    
    for a in actions:
        x, v, _ = true_physics_step(x, v, a, g, e, dt)
        states.append((x, v))
        costs.append((x - 2.0) ** 2)
        
    final_x = states[-1][0]
    total_cost = sum(costs) + (final_x - 2.0) ** 2
    
    return states, total_cost


# ============== PLANNERS ==============

def mppi(x, v, horizon=30, n_samples=64, temperature=1.0, action_scale=2.0):
    """Standard MPPI."""
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
    
    # ESS
    ess = 1.0 / (weights ** 2).sum()
    ess_frac = ess / n_samples
    w_max = weights.max()
    
    actions_weighted = actions.T @ weights
    return actions_weighted[0], {'ess_frac': ess_frac, 'w_max': w_max}


def cem(x, v, horizon=30, n_samples=64, n_elites=10, action_scale=2.0):
    """Cross-Entropy Method."""
    # Initialize
    mean = np.zeros(horizon)
    std = action_scale * np.ones(horizon)
    
    for _ in range(5):  # CEM iterations
        # Sample
        actions = np.random.normal(mean, std, (n_samples, horizon))
        actions = np.clip(actions, -action_scale, action_scale)
        
        # Evaluate
        costs = []
        for i in range(n_samples):
            _, total_cost = true_physics_rollout(x, v, actions[i])
            costs.append(total_cost)
            
        costs = np.array(costs)
        
        # Elite selection
        elite_idx = np.argsort(costs)[:n_elites]
        elite_actions = actions[elite_idx]
        
        # Update distribution
        mean = elite_actions.mean(axis=0)
        std = elite_actions.std(axis=0) + 1e-6  # Prevent collapse
    
    # ESS-like metric: elite diversity
    elite_diversity = elite_actions.std(axis=0).mean()
    
    return mean[0], {'elite_diversity': elite_diversity, 'elite_cost_min': costs[elite_idx[0]]}


# ============== EXPERIMENTS ==============

def run_episode(env, planner, horizon=30, n_samples=64, temperature=1.0, **kwargs):
    """Run one episode."""
    obs = env.reset(seed=kwargs.get('seed', 0))
    x, v = obs[0], obs[1]
    
    success = False
    miss = 0
    
    for step in range(30):
        if planner == 'mppi':
            action, diag = mppi(x, v, horizon, n_samples, temperature)
        elif planner == 'cem':
            action, diag = cem(x, v, horizon, n_samples)
        else:
            action = np.random.uniform(-2.0, 2.0)
            diag = {}
            
        obs = env.step(action)
        x, v = obs[0], obs[1]
        
        if abs(x - env.x_target) < env.tau:
            success = True
            
    miss = abs(x - env.x_target)
    return {'success': success, 'miss': miss, 'final_x': x}


def experiment_1_cem_vs_mppi():
    """Compare CEM vs MPPI."""
    print("="*70)
    print("EXPERIMENT 1: CEM vs MPPI")
    print("="*70)
    
    n_episodes = 100
    horizon = 30
    n_samples = 64
    
    planners = ['mppi', 'cem']
    results = {p: [] for p in planners}
    
    for ep in range(n_episodes):
        for planner in planners:
            env = BouncingBallGravity()
            result = run_episode(env, planner, horizon=horizon, n_samples=n_samples, 
                               temperature=1.0, seed=ep)
            results[planner].append(result)
            
        if (ep + 1) % 20 == 0:
            print(f"Episode {ep+1}/{n_episodes}")
    
    # Summary
    summary = {}
    for planner in planners:
        successes = sum(r['success'] for r in results[planner])
        misses = [r['miss'] for r in results[planner]]
        summary[planner] = {
            'success_rate': successes / n_episodes,
            'miss_mean': np.mean(misses),
            'miss_std': np.std(misses),
        }
        print(f"\n{planner.upper()}:")
        print(f"  Success: {summary[planner]['success_rate']:.1%}")
        print(f"  Miss: {summary[planner]['miss_mean']:.3f} ± {summary[planner]['miss_std']:.3f}")
    
    return summary


def experiment_2_temp_sample_sweep():
    """Temperature × Sample budget sweep."""
    print("\n" + "="*70)
    print("EXPERIMENT 2: TEMPERATURE × SAMPLE SWEEP")
    print("="*70)
    
    n_episodes = 50
    horizon = 30
    
    temperatures = [0.1, 0.5, 1.0, 2.0, 5.0]
    sample_counts = [64, 256, 1024]
    
    results = {}
    
    for n_samples in sample_counts:
        for temp in temperatures:
            key = f"N{n_samples}_T{temp}"
            print(f"\nRunning {key}...", end=" ")
            
            successes = 0
            misses = []
            
            for ep in range(n_episodes):
                env = BouncingBallGravity()
                result = run_episode(env, 'mppi', horizon=horizon, 
                                   n_samples=n_samples, temperature=temp, seed=ep)
                if result['success']:
                    successes += 1
                misses.append(result['miss'])
            
            results[key] = {
                'n_samples': n_samples,
                'temperature': temp,
                'success_rate': successes / n_episodes,
                'miss_mean': np.mean(misses),
                'miss_std': np.std(misses),
            }
            
            print(f"Success: {results[key]['success_rate']:.1%}, Miss: {results[key]['miss_mean']:.3f}")
    
    return results


def main():
    # Experiment 1: CEM vs MPPI
    cem_mppi_results = experiment_1_cem_vs_mppi()
    
    # Experiment 2: Temperature × Sample sweep
    sweep_results = experiment_2_temp_sample_sweep()
    
    # Save
    output_dir = '/home/ulluboz/pcp-jepa-research/results/phase3'
    os.makedirs(output_dir, exist_ok=True)
    
    with open(f'{output_dir}/cem_vs_mppi.json', 'w') as f:
        json.dump(cem_mppi_results, f, indent=2)
        
    with open(f'{output_dir}/temp_sample_sweep.json', 'w') as f:
        json.dump(sweep_results, f, indent=2)
    
    print("\n" + "="*70)
    print("SUMMARY")
    print("="*70)
    
    # Find best config
    best_key = max(sweep_results.keys(), key=lambda k: sweep_results[k]['success_rate'])
    best = sweep_results[best_key]
    print(f"\nBest config: {best_key}")
    print(f"  Success: {best['success_rate']:.1%}, Miss: {best['miss_mean']:.3f}")
    
    # Print sweep as table
    print("\n--- Temperature × Sample Sweep ---")
    print(f"{'N':<8} {'T=0.1':<10} {'T=0.5':<10} {'T=1.0':<10} {'T=2.0':<10} {'T=5.0':<10}")
    print("-"*60)
    for n in [64, 256, 1024]:
        row = f"{n:<8}"
        for t in [0.1, 0.5, 1.0, 2.0, 5.0]:
            key = f"N{n}_T{t}"
            sr = sweep_results[key]['success_rate']
            row += f"{sr:>8.0%} "
        print(row)
    
    print(f"\nSaved to {output_dir}/")


if __name__ == '__main__':
    main()
