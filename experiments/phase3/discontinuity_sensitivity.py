#!/usr/bin/env python3
"""
PHASE 3: Discontinuity Sensitivity Law

Test: Does the open-loop vs feedback gap widen as "hybridness" increases?

Vary restitution ∈ {0.3, 0.6, 0.8, 0.95} and plot success for:
- Open-loop MPPI
- Feedback gains
- Fixed PD

Expected: Gap widens as restitution → 0.95 (more energetic bounces = harder to predict)
"""

import numpy as np
import json
import os


class BouncingBallVariableRestitution:
    def __init__(self, restitution=0.8, tau=0.3):
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
        return np.array([self.x, self.v])
    
    def step(self, a):
        a = np.clip(a, -2.0, 2.0)
        self.v += (-self.g + a) * self.dt
        self.x += self.v * self.dt
        
        if self.x < 0:
            self.x = -self.x * self.e
            self.v = -self.v * self.e
        if self.x > 3:
            self.x = 3 - (self.x - 3) * self.e
            self.v = -self.v * self.e
            
        return np.array([self.x, self.v])


def simulate_rollout(x0, v0, actions, e):
    """Simulate action sequence with given restitution."""
    x, v = x0, v0
    g = 9.81
    dt = 0.05
    
    for a in actions:
        a = np.clip(a, -2.0, 2.0)
        v += (-g + a) * dt
        x += v * dt
        
        if x < 0:
            x = -x * e
            v = -v * e
        elif x > 3:
            x = 3 - (x - 3) * e
            v = -v * e
    
    return x, (x - 2.0) ** 2


def run_open_loop(restitution, n_episodes=200, seed=0):
    """Open-loop MPPI with variable restitution."""
    successes = 0
    
    for ep in range(n_episodes):
        env = BouncingBallVariableRestitution(restitution=restitution)
        obs = env.reset(seed=ep + seed)
        x, v = obs[0], obs[1]
        
        for step in range(30):
            # Sample actions
            actions = np.random.uniform(-2.0, 2.0, (64, 30))
            
            costs = []
            for i in range(64):
                _, cost = simulate_rollout(x, v, actions[i], restitution)
                costs.append(cost)
            
            costs = np.array(costs)
            min_cost = costs.min()
            weights = np.exp(-(costs - min_cost) / 1.0)
            weights = weights / weights.sum()
            
            action = (actions.T @ weights)[0]
            env.step(action)
            x, v = env.x, env.v
            
            if abs(x - env.x_target) < env.tau:
                successes += 1
                break
    
    return successes / n_episodes


def run_feedback_gains(restitution, k1=1.5, k2=-2.0, n_episodes=200, seed=0):
    """Feedback gains with variable restitution."""
    successes = 0
    
    for ep in range(n_episodes):
        env = BouncingBallVariableRestitution(restitution=restitution)
        obs = env.reset(seed=ep + seed)
        x, v = obs[0], obs[1]
        
        for step in range(30):
            a = k1 * (env.x_target - x) + k2 * (-v)
            a = np.clip(a, -2.0, 2.0)
            
            env.step(a)
            x, v = env.x, env.v
            
            if abs(x - env.x_target) < env.tau:
                successes += 1
                break
    
    return successes / n_episodes


def run_fixed_pd(restitution, k1=2.0, k2=0.0, n_episodes=200, seed=0):
    """Fixed PD (no damping) - baseline."""
    return run_feedback_gains(restitution, k1=k1, k2=k2, n_episodes=n_episodes, seed=seed)


def main():
    print("="*70)
    print("DISCONTINUITY SENSITIVITY: Restitution Sweep")
    print("="*70)
    
    restitutions = [0.3, 0.5, 0.7, 0.8, 0.9, 0.95]
    n_episodes = 200
    
    results = {'open_loop': [], 'feedback_opt': [], 'fixed_pd': []}
    
    for e in restitutions:
        print(f"\nRestitution e={e:.2f}")
        
        # Open-loop
        ol = run_open_loop(e, n_episodes=n_episodes)
        results['open_loop'].append(ol)
        print(f"  Open-loop: {ol:.1%}")
        
        # Optimal feedback
        fb = run_feedback_gains(e, k1=1.5, k2=-2.0, n_episodes=n_episodes)
        results['feedback_opt'].append(fb)
        print(f"  Feedback (k1=1.5, k2=-2.0): {fb:.1%}")
        
        # Fixed PD
        pd = run_fixed_pd(e, k1=2.0, k2=0.0, n_episodes=n_episodes)
        results['fixed_pd'].append(pd)
        print(f"  Fixed PD (k1=2.0, k2=0): {pd:.1%}")
        
        # Gap
        gap = fb - ol
        print(f"  Gap: {gap:.1%}")
    
    # Summary table
    print("\n" + "="*70)
    print("SUMMARY: Success vs Restitution")
    print("="*70)
    print(f"\n{'e':<8} {'Open-loop':<12} {'Feedback':<12} {'Fixed PD':<12} {'Gap':<10}")
    print("-"*55)
    
    for i, e in enumerate(restitutions):
        ol = results['open_loop'][i]
        fb = results['feedback_opt'][i]
        pd = results['fixed_pd'][i]
        gap = fb - ol
        print(f"{e:.2f}     {ol:>8.1%}     {fb:>8.1%}     {pd:>8.1%}     {gap:>6.1%}")
    
    # Save
    output_dir = '/home/ulluboz/pcp-jepa-research/results/phase3'
    os.makedirs(output_dir, exist_ok=True)
    
    save_data = {
        'restitutions': restitutions,
        'open_loop': results['open_loop'],
        'feedback_opt': results['feedback_opt'],
        'fixed_pd': results['fixed_pd'],
    }
    
    with open(f'{output_dir}/discontinuity_sensitivity.json', 'w') as f:
        json.dump(save_data, f, indent=2)
    
    print(f"\nSaved to {output_dir}/discontinuity_sensitivity.json")
    
    # Interpretation
    print("\n" + "="*70)
    print("INTERPRETATION")
    print("="*70)
    
    # Check if gap increases with restitution
    gaps = [results['feedback_opt'][i] - results['open_loop'][i] for i in range(len(restitutions))]
    
    if gaps[-1] > gaps[0]:
        print(f"\n✓ Gap WIDENS with hybridness: {gaps[0]:.1%} → {gaps[-1]:.1%}")
        print("  Higher restitution = more energetic bounces = harder to predict")
        print("  Feedback adapts to bounce outcomes, open-loop cannot")
    else:
        print(f"\n? Gap does NOT widen: {gaps[0]:.1%} → {gaps[-1]:.1%}")


if __name__ == '__main__':
    main()
