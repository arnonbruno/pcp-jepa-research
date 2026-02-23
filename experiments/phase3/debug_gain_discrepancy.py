#!/usr/bin/env python3
"""
PHASE 3: Debug 56% vs 71% Discrepancy

Run signature verification:
- Initial-state seed set
- Action bounds/scale
- Horizon H / episode length
- Success threshold τ
- Gravity/walls/restitution
- Replanning on/off
- Clipping order
"""

import numpy as np
import json
import os

# ============== RUN SIGNATURE ==============
SIGNATURE = {
    'env': 'BouncingBallGravity',
    'g': 9.81,
    'e': 0.8,
    'dt': 0.05,
    'x_target': 2.0,
    'tau': 0.3,
    'action_bounds': [-2.0, 2.0],
    'H': 30,
    'episode_length': 30,
    'start_positions': [0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5],
    'replanning': True,
}


class BouncingBallGravity:
    def __init__(self):
        self.g = SIGNATURE['g']
        self.e = SIGNATURE['e']
        self.dt = SIGNATURE['dt']
        self.x_target = SIGNATURE['x_target']
        self.tau = SIGNATURE['tau']
        
    def reset(self, seed=None):
        if seed is not None:
            np.random.seed(seed)
        self.x = SIGNATURE['start_positions'][seed % len(SIGNATURE['start_positions'])]
        self.v = 0.0
        return np.array([self.x, self.v])
    
    def step(self, a):
        # Clip BEFORE dynamics (standard)
        a = np.clip(a, SIGNATURE['action_bounds'][0], SIGNATURE['action_bounds'][1])
        
        self.v += (-self.g + a) * self.dt
        self.x += self.v * self.dt
        
        if self.x < 0:
            self.x = -self.x * self.e
            self.v = -self.v * self.e
        if self.x > 3:
            self.x = 3 - (self.x - 3) * self.e
            self.v = -self.v * self.e
            
        return np.array([self.x, self.v])


def run_gain_controller(k1, k2, n_episodes=1000, seed_offset=0):
    """
    Gain controller: a = clip(k1*(x* - x) + k2*(-v), bounds)
    
    This is the CORRECT implementation.
    """
    successes = 0
    misses = []
    
    for ep in range(n_episodes):
        env = BouncingBallGravity()
        obs = env.reset(seed=ep + seed_offset)
        x, v = obs[0], obs[1]
        
        for step in range(SIGNATURE['episode_length']):
            # Feedback law: a = k1*(target - x) + k2*(-v)
            a = k1 * (env.x_target - x) + k2 * (-v)
            
            # Clip to bounds
            a = np.clip(a, SIGNATURE['action_bounds'][0], SIGNATURE['action_bounds'][1])
            
            obs = env.step(a)
            x, v = obs[0], obs[1]
            
            if abs(x - env.x_target) < env.tau:
                successes += 1
                break
        
        misses.append(abs(x - env.x_target))
    
    return {
        'success_rate': successes / n_episodes,
        'miss_mean': np.mean(misses),
        'miss_std': np.std(misses),
        'n_episodes': n_episodes,
    }


def main():
    print("="*70)
    print("DEBUG: 56% vs 71% DISCREPANCY")
    print("="*70)
    
    print("\nRun signature:")
    for k, v in SIGNATURE.items():
        print(f"  {k}: {v}")
    
    print("\n" + "="*70)
    print("TEST 1: Different gain configurations (1000 episodes each)")
    print("="*70)
    
    configs = [
        {'k1': 1.5, 'k2': -2.0},
        {'k1': 2.0, 'k2': -2.0},
        {'k1': 2.0, 'k2': 0.0},
        {'k1': 1.0, 'k2': -1.0},
        {'k1': 3.0, 'k2': -1.0},
    ]
    
    for cfg in configs:
        result = run_gain_controller(cfg['k1'], cfg['k2'], n_episodes=1000)
        print(f"k1={cfg['k1']:.1f}, k2={cfg['k2']:.1f}: {result['success_rate']:.1%} (miss: {result['miss_mean']:.3f}±{result['miss_std']:.3f})")
    
    print("\n" + "="*70)
    print("TEST 2: Grid search for optimal gains")
    print("="*70)
    
    best_rate = 0
    best_k1, best_k2 = 2.0, 0.0
    
    for k1 in np.arange(0.5, 4.0, 0.5):
        for k2 in np.arange(-3.0, 1.0, 0.5):
            result = run_gain_controller(k1, k2, n_episodes=200)
            if result['success_rate'] > best_rate:
                best_rate = result['success_rate']
                best_k1, best_k2 = k1, k2
    
    print(f"Best gains: k1={best_k1:.1f}, k2={best_k2:.1f}")
    print(f"Best success: {best_rate:.1%}")
    
    # Verify with more episodes
    print("\nVerifying best gains with 1000 episodes...")
    result = run_gain_controller(best_k1, best_k2, n_episodes=1000)
    print(f"k1={best_k1:.1f}, k2={best_k2:.1f}: {result['success_rate']:.1%}")
    
    print("\n" + "="*70)
    print("CONCLUSION")
    print("="*70)
    
    # Check if we can reproduce 71%
    if best_rate >= 0.65:
        print(f"\n✓ Can achieve {best_rate:.1%} with optimized gains")
        print("  The 71% from earlier was likely with different gains.")
    else:
        print(f"\n? Maximum achievable: {best_rate:.1%}")
        print("  Cannot reproduce 71% with this run signature.")


if __name__ == '__main__':
    main()
