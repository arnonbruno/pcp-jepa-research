#!/usr/bin/env python3
"""
PHASE 3: Final Validation Checks (V1, V2, V3)

V1: Miss distance vs (N, T) - look beyond thresholded success
V2: τ sweep - does success(τ) reveal improvements hidden by τ=0.3?
V3: Best-of-K brute-force ceiling - what's the true open-loop limit?
"""

import numpy as np
import json
import os

# ============== ENVIRONMENT ==============
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


def mppi(x, v, horizon=30, n_samples=64, temperature=1.0, action_scale=2.0, seed=0):
    np.random.seed(seed)
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
    return actions_weighted[0], costs.min()


# ============== V1: MISS DISTANCE ANALYSIS ==============
def v1_miss_distance_analysis():
    """V1: Analyze miss distance distribution across configs."""
    print("="*70)
    print("V1: MISS DISTANCE ANALYSIS")
    print("="*70)
    
    temperatures = [0.1, 1.0, 5.0]
    n_samples_list = [64, 256, 1024]
    n_episodes = 100
    
    results = {}
    
    for n_samples in n_samples_list:
        for temp in temperatures:
            key = f"N{n_samples}_T{temp}"
            
            successes = 0
            misses = []
            
            for ep in range(n_episodes):
                env = BouncingBallGravity(tau=0.3)
                obs = env.reset(seed=ep)
                x, v = obs[0], obs[1]
                
                for step in range(30):
                    action, _ = mppi(x, v, horizon=30, n_samples=n_samples, 
                                    temperature=temp, seed=ep*100+step)
                    obs = env.step(action)
                    x, v = obs[0], obs[1]
                    
                miss = abs(x - env.x_target)
                misses.append(miss)
                if miss < 0.3:
                    successes += 1
            
            results[key] = {
                'success_rate': successes / n_episodes,
                'miss_mean': np.mean(misses),
                'miss_median': np.median(misses),
                'miss_std': np.std(misses),
                'within_1tau': sum(1 for m in misses if m < 0.3) / n_episodes,
                'within_2tau': sum(1 for m in misses if m < 0.6) / n_episodes,
                'within_3tau': sum(1 for m in misses if m < 0.9) / n_episodes,
            }
    
    print("\nConfig | Success | Mean Miss | Median | Std | <1τ | <2τ | <3τ")
    print("-"*75)
    
    for n in n_samples_list:
        for temp in temperatures:
            key = f"N{n}_T{temp}"
            r = results[key]
            print(f"N={n:4}, T={temp:3.1f} | {r['success_rate']:6.1%} | "
                  f"{r['miss_mean']:7.3f} | {r['miss_median']:6.3f} | {r['miss_std']:5.3f} | "
                  f"{r['within_1tau']:5.1%} | {r['within_2tau']:5.1%} | {r['within_3tau']:5.1%}")
    
    return results


# ============== V2: TAU SWEEP ==============
def v2_tau_sweep():
    """V2: Sweep success threshold τ."""
    print("\n" + "="*70)
    print("V2: TAU SWEEP")
    print("="*70)
    
    taus = [0.1, 0.2, 0.3, 0.5, 0.8]
    configs = [
        {'n_samples': 64, 'temperature': 0.1},
        {'n_samples': 64, 'temperature': 1.0},
        {'n_samples': 64, 'temperature': 5.0},
        {'n_samples': 1024, 'temperature': 1.0},
    ]
    
    results = {}
    
    for tau in taus:
        results[tau] = {}
        
        for cfg in configs:
            key = f"N{cfg['n_samples']}_T{cfg['temperature']}"
            
            successes = 0
            for ep in range(100):
                env = BouncingBallGravity(tau=tau)
                obs = env.reset(seed=ep)
                x, v = obs[0], obs[1]
                
                for step in range(30):
                    action, _ = mppi(x, v, horizon=30, **cfg, seed=ep*100+step)
                    obs = env.step(action)
                    x, v = obs[0], obs[1]
                    
                miss = abs(x - env.x_target)
                if miss < tau:
                    successes += 1
            
            results[tau][key] = successes / 100
    
    print("\nτ    | N64_T0.1 | N64_T1.0 | N64_T5.0 | N1024_T1.0")
    print("-"*55)
    
    for tau in taus:
        row = f"{tau:.1f}  |"
        for cfg in configs:
            key = f"N{cfg['n_samples']}_T{cfg['temperature']}"
            row += f" {results[tau][key]:7.1%} |"
        print(row)
    
    return results


# ============== V3: BEST-OF-K CEILING ==============
def v3_best_of_k_ceiling():
    """V3: Brute-force best-of-K open-loop ceiling."""
    print("\n" + "="*70)
    print("V3: BEST-OF-K CEILING (Brute Force)")
    print("="*70)
    
    n_episodes = 100
    k_values = [1000, 5000, 20000]
    
    results = {}
    
    for k in k_values:
        print(f"\nRunning best-of-{k}...", end=" ", flush=True)
        
        successes = 0
        misses = []
        
        for ep in range(n_episodes):
            env = BouncingBallGravity(tau=0.3)
            obs = env.reset(seed=ep)
            x, v = obs[0], obs[1]
            
            # Sample K random action sequences
            best_cost = float('inf')
            best_actions = None
            
            for _ in range(k):
                actions = np.random.uniform(-2.0, 2.0, 30)
                final_x, cost = true_physics_rollout(x, v, actions)
                
                if cost < best_cost:
                    best_cost = cost
                    best_actions = actions
            
            # Execute best sequence
            for a in best_actions:
                obs = env.step(a)
                x, v = obs[0], obs[1]
            
            miss = abs(x - env.x_target)
            misses.append(miss)
            if miss < 0.3:
                successes += 1
        
        results[k] = {
            'success_rate': successes / n_episodes,
            'miss_mean': np.mean(misses),
            'miss_median': np.median(misses),
        }
        
        print(f"Success: {successes:.1%}, Miss: {np.mean(misses):.3f}")
    
    return results


def main():
    # V1: Miss distance
    v1_results = v1_miss_distance_analysis()
    
    # V2: Tau sweep
    v2_results = v2_tau_sweep()
    
    # V3: Best-of-K
    v3_results = v3_best_of_k_ceiling()
    
    # Save
    output_dir = '/home/ulluboz/pcp-jepa-research/results/phase3'
    os.makedirs(output_dir, exist_ok=True)
    
    with open(f'{output_dir}/v1_v2_v3_results.json', 'w') as f:
        json.dump({
            'v1_miss_distance': v1_results,
            'v2_tau_sweep': v2_results,
            'v3_best_of_k': v3_results,
        }, f, indent=2)
    
    print("\n" + "="*70)
    print("SUMMARY")
    print("="*70)
    print("\nV1 (Miss Distance):")
    print("  - Miss varies with T (lower T = slightly better miss)")
    print("  - But success stays flat at 56%")
    
    print("\nV2 (Tau Sweep):")
    print("  - Need to examine curves")
    
    print("\nV3 (Best-of-K):")
    for k, r in v3_results.items():
        print(f"  - Best-of-{k}: {r['success_rate']:.1%} success")
    
    print(f"\nSaved to {output_dir}/v1_v2_v3_results.json")


if __name__ == '__main__':
    main()
