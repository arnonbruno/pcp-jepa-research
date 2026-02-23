#!/usr/bin/env python3
"""
PHASE 3: Open-Loop vs MPC (Closed-Loop) Comparison

Test 1: Open-loop - plan once, execute all actions without replanning
Test 2: MPC (closed-loop) - replan every step

This tests whether the ceiling is due to open-loop brittleness.
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
    
    actions_weighted = actions.T @ weights
    return actions_weighted


# ============== OPEN-LOOP: Plan once, execute all ==============
def run_open_loop(horizon=30, n_samples=64, temperature=1.0, seed=0):
    """Open-loop: plan once for H steps, execute without replanning."""
    env = BouncingBallGravity()
    obs = env.reset(seed=seed)
    x, v = obs[0], obs[1]
    
    # Plan once
    actions = mppi(x, v, horizon=horizon, n_samples=n_samples, temperature=temperature)
    
    # Execute all actions without replanning
    for step in range(min(horizon, 30)):
        a = actions[step]
        obs = env.step(a)
        x, v = obs[0], obs[1]
        
        if abs(x - env.x_target) < env.tau:
            return {'success': True, 'miss': abs(x - env.x_target), 'mode': 'open-loop'}
    
    return {'success': False, 'miss': abs(x - env.x_target), 'mode': 'open-loop'}


# ============== MPC: Replan every step ==============
def run_mpc(horizon=30, n_samples=64, temperature=1.0, seed=0):
    """MPC closed-loop: replan every step."""
    env = BouncingBallGravity()
    obs = env.reset(seed=seed)
    x, v = obs[0], obs[1]
    
    for step in range(30):
        # Replan from current state
        actions = mppi(x, v, horizon=horizon, n_samples=n_samples, temperature=temperature)
        
        # Execute first action
        a = actions[0]
        obs = env.step(a)
        x, v = obs[0], obs[1]
        
        if abs(x - env.x_target) < env.tau:
            return {'success': True, 'miss': abs(x - env.x_target), 'mode': 'MPC'}
    
    return {'success': False, 'miss': abs(x - env.x_target), 'mode': 'MPC'}


# ============== P CONTROLLER (BASELINE) ==============
def run_p_controller(kp=2.0, seed=0):
    """P controller as baseline."""
    env = BouncingBallGravity()
    obs = env.reset(seed=seed)
    x, v = obs[0], obs[1]
    
    for step in range(30):
        # a = kp * (target - x)
        a = kp * (env.x_target - x)
        a = np.clip(a, -2.0, 2.0)
        
        obs = env.step(a)
        x, v = obs[0], obs[1]
        
        if abs(x - env.x_target) < env.tau:
            return {'success': True, 'miss': abs(x - env.x_target), 'mode': 'P'}
    
    return {'success': False, 'miss': abs(x - env.x_target), 'mode': 'P'}


def main():
    print("="*70)
    print("OPEN-LOOP vs MPC COMPARISON")
    print("="*70)
    
    n_episodes = 200
    horizon = 30
    n_samples = 64
    temperature = 1.0
    
    modes = ['open-loop', 'MPC', 'P']
    results = {m: {'successes': 0, 'misses': []} for m in modes}
    
    print(f"\nRunning {n_episodes} episodes...")
    
    for ep in range(n_episodes):
        # Open-loop
        r = run_open_loop(horizon=horizon, n_samples=n_samples, temperature=temperature, seed=ep)
        results['open-loop']['successes'] += int(r['success'])
        results['open-loop']['misses'].append(r['miss'])
        
        # MPC
        r = run_mpc(horizon=horizon, n_samples=n_samples, temperature=temperature, seed=ep)
        results['MPC']['successes'] += int(r['success'])
        results['MPC']['misses'].append(r['miss'])
        
        # P controller
        r = run_p_controller(seed=ep)
        results['P']['successes'] += int(r['success'])
        results['P']['misses'].append(r['miss'])
        
        if (ep + 1) % 50 == 0:
            print(f"  Episode {ep+1}/{n_episodes}")
    
    # Summary
    print("\n" + "="*70)
    print("RESULTS")
    print("="*70)
    
    print(f"\n{'Mode':<12} {'Success':<10} {'Miss Mean':<12} {'Miss Median':<12}")
    print("-"*50)
    
    for mode in modes:
        s = results[mode]['successes'] / n_episodes
        m = np.mean(results[mode]['misses'])
        med = np.median(results[mode]['misses'])
        print(f"{mode:<12} {s:>7.1%}   {m:>9.3f}   {med:>9.3f}")
    
    # Save
    output_dir = '/home/ulluboz/pcp-jepa-research/results/phase3'
    os.makedirs(output_dir, exist_ok=True)
    
    save_results = {}
    for mode in modes:
        save_results[mode] = {
            'success_rate': results[mode]['successes'] / n_episodes,
            'miss_mean': float(np.mean(results[mode]['misses'])),
            'miss_median': float(np.median(results[mode]['misses'])),
        }
    
    with open(f'{output_dir}/openloop_vs_mpc.json', 'w') as f:
        json.dump(save_results, f, indent=2)
    
    print(f"\nSaved to {output_dir}/openloop_vs_mpc.json")
    
    # Interpretation
    print("\n" + "="*70)
    print("INTERPRETATION")
    print("="*70)
    
    ol = save_results['open-loop']['success_rate']
    mpc = save_results['MPC']['success_rate']
    p = save_results['P']['success_rate']
    
    print(f"\nOpen-loop: {ol:.1%}")
    print(f"MPC:       {mpc:.1%}")
    print(f"P:         {p:.1%}")
    
    if mpc > ol + 0.1:
        print(f"\n✓ MPC significantly outperforms open-loop (+{mpc-ol:.1%})")
        print("  → The ceiling IS due to open-loop brittleness!")
    elif mpc > ol:
        print(f"\n✓ MPC slightly outperforms open-loop (+{mpc-ol:.1%})")
    else:
        print(f"\n✗ MPC does NOT improve over open-loop")
        print("  → The problem is NOT replanning; it's deeper")


if __name__ == '__main__':
    main()
