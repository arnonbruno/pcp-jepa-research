#!/usr/bin/env python3
"""
PHASE 3: Validation Checks for Sweep Results

Check 1: PRNG diversity - verify actions differ across configs
Check 2: Behavioral hash - verify first actions differ across configs
"""

import numpy as np
import json
import os

# ============== ENVIRONMENT ==============
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


# ============== MPPI WITH PROPER SEEDING ==============
def mppi_with_logging(x, v, horizon=30, n_samples=64, temperature=1.0, 
                      action_scale=2.0, seed=0):
    """MPPI with detailed logging of randomness."""
    
    # IMPORTANT: Seed each config differently to ensure diversity
    np.random.seed(seed * 10000 + n_samples * 100 + int(temperature * 10))
    
    actions = np.random.uniform(-action_scale, action_scale, (n_samples, horizon))
    
    # Log action statistics
    action_mean = actions.mean()
    action_std = actions.std()
    
    costs = []
    for i in range(n_samples):
        _, total_cost = true_physics_rollout(x, v, actions[i])
        costs.append(total_cost)
        
    costs = np.array(costs)
    
    min_cost = costs.min()
    costs_shifted = costs - min_cost
    weights = np.exp(-costs_shifted / temperature)
    weights = weights / weights.sum()
    
    ess = 1.0 / (weights ** 2).sum()
    ess_frac = ess / n_samples
    w_max = weights.max()
    
    actions_weighted = actions.T @ weights
    first_action = actions_weighted[0]
    
    return first_action, {
        'action_mean': action_mean,
        'action_std': action_std,
        'ess_frac': ess_frac,
        'w_max': w_max,
        'cost_mean': costs.mean(),
        'cost_std': costs.std(),
    }


# ============== CHECK 1: PRNG DIVERSITY ==============
def check_prng_diversity():
    """Verify that different configs produce different action distributions."""
    print("="*70)
    print("CHECK 1: PRNG DIVERSITY")
    print("="*70)
    
    env = BouncingBallGravity()
    obs = env.reset(seed=0)
    x, v = obs[0], obs[1]
    
    configs = [
        {'n_samples': 64, 'temperature': 0.1, 'seed': 0},
        {'n_samples': 64, 'temperature': 1.0, 'seed': 0},
        {'n_samples': 64, 'temperature': 5.0, 'seed': 0},
        {'n_samples': 256, 'temperature': 1.0, 'seed': 0},
        {'n_samples': 1024, 'temperature': 1.0, 'seed': 0},
    ]
    
    print("\nConfig | Action Mean | Action Std | ESS_frac | w_max")
    print("-"*65)
    
    for cfg in configs:
        action, stats = mppi_with_logging(x, v, horizon=30, **cfg)
        print(f"N={cfg['n_samples']:4}, T={cfg['temperature']:3.1f} | "
              f"{stats['action_mean']:8.4f} | {stats['action_std']:8.4f} | "
              f"{stats['ess_frac']:7.3f} | {stats['w_max']:6.3f}")
    
    # Now check with DIFFERENT seeds for SAME config
    print("\n--- Same config, different seeds ---")
    cfg = {'n_samples': 64, 'temperature': 1.0}
    for seed in range(5):
        action, stats = mppi_with_logging(x, v, horizon=30, seed=seed, **cfg)
        print(f"Seed {seed}: action_mean={stats['action_mean']:.4f}, action_std={stats['action_std']:.4f}")


# ============== CHECK 2: BEHAVIORAL HASH ==============
def check_behavioral_hash():
    """Verify first actions differ across configs for same initial state."""
    print("\n" + "="*70)
    print("CHECK 2: BEHAVIORAL HASH")
    print("="*70)
    
    # Test on 20 fixed initial states
    n_tests = 20
    temperatures = [0.1, 0.5, 1.0, 2.0, 5.0]
    
    results = {t: [] for t in temperatures}
    
    for seed in range(n_tests):
        env = BouncingBallGravity()
        obs = env.reset(seed=seed)
        x, v = obs[0], obs[1]
        
        for temp in temperatures:
            action, _ = mppi_with_logging(x, v, horizon=30, n_samples=64, 
                                          temperature=temp, seed=seed)
            results[temp].append(action)
    
    # Compare first actions across temperatures
    print("\nFirst action by temperature (first 5 episodes):")
    print("Episode | T=0.1 | T=0.5 | T=1.0 | T=2.0 | T=5.0")
    print("-"*55)
    
    for ep in range(5):
        row = f"{ep:7} |"
        for temp in temperatures:
            row += f" {results[temp][ep]:6.3f} |"
        print(row)
    
    # Check if actions are identical across temperatures
    identical_count = 0
    for ep in range(n_tests):
        actions = [results[temp][ep] for temp in temperatures]
        if max(actions) - min(actions) < 0.01:
            identical_count += 1
    
    print(f"\nEpisodes with nearly identical actions across all T: {identical_count}/{n_tests}")
    
    # Compute variance
    for temp in temperatures:
        arr = np.array(results[temp])
        print(f"T={temp}: mean={arr.mean():.3f}, std={arr.std():.3f}")


# ============== CHECK 3: SUCCESS RATE WITH PROPER SEEDING ==============
def check_success_rate_variance():
    """Verify success rate actually varies with more episodes."""
    print("\n" + "="*70)
    print("CHECK 3: SUCCESS RATE VARIANCE")
    print("="*70)
    
    def run_episode(seed, n_samples=64, temperature=1.0):
        env = BouncingBallGravity()
        obs = env.reset(seed=seed)
        x, v = obs[0], obs[1]
        
        for step in range(30):
            action, _ = mppi_with_logging(x, v, horizon=30, 
                                         n_samples=n_samples, temperature=temperature, seed=seed*100+step)
            obs = env.step(action)
            x, v = obs[0], obs[1]
            
            if abs(x - env.x_target) < env.tau:
                return True
        return False
    
    # Run multiple trials to see variance
    print("\nRunning 3 trials of 100 episodes each...")
    
    for trial in range(3):
        np.random.seed(trial * 12345)
        
        configs = [
            {'n_samples': 64, 'temperature': 0.1},
            {'n_samples': 64, 'temperature': 5.0},
            {'n_samples': 1024, 'temperature': 0.1},
        ]
        
        for cfg in configs:
            successes = 0
            for ep in range(100):
                # Different seed for each episode AND each step
                seed = trial * 10000 + ep
                if run_episode(seed, **cfg):
                    successes += 1
            
            print(f"Trial {trial+1}, N={cfg['n_samples']}, T={cfg['temperature']}: {successes}% success")


def main():
    check_prng_diversity()
    check_behavioral_hash()
    check_success_rate_variance()
    
    print("\n" + "="*70)
    print("VALIDATION COMPLETE")
    print("="*70)


if __name__ == '__main__':
    main()
