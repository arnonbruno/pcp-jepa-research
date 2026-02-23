#!/usr/bin/env python3
"""
PHASE 4: Mode-Event Conditioned Controller (FIXED)

Bug fix: k2 sign was wrong. Correct formula from Phase 3:
  a = k1*(x* - x) + k2*(-v) where k2 = -2.0
  This gives: 1.5*(2-x) + (-2)*(-v) = 1.5*(2-x) + 2v = 71% success

Goal: Test if piecewise gains improve beyond 71%
"""

import numpy as np


class BouncingBallGravity:
    def __init__(self, tau=0.3, restitution=0.8):
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
        
        bounced = False
        if self.x < 0:
            self.x = -self.x * self.e
            self.v = -self.v * self.e
            bounced = True
        if self.x > 3:
            self.x = 3 - (self.x - 3) * self.e
            self.v = -self.v * self.e
            bounced = True
            
        return np.array([self.x, self.v]), bounced


def run_single_pd(k1, k2, n_episodes=200, restitution=0.8, seed=0):
    """
    Run single PD controller: a = clip(k1*(x* - x) + k2*(-v))
    """
    successes = 0
    
    for ep in range(n_episodes):
        env = BouncingBallGravity(restitution=restitution)
        obs = env.reset(seed=ep + seed)
        x, v = obs[0], obs[1]
        
        for step in range(30):
            a = k1 * (env.x_target - x) + k2 * (-v)
            a = np.clip(a, -2.0, 2.0)
            
            obs, _ = env.step(a)
            x, v = obs[0], obs[1]
            
            if abs(x - env.x_target) < env.tau:
                successes += 1
                break
    
    return successes / n_episodes


def run_mode_controller(k1, k2, k1_b, k2_b, n_episodes=200, restitution=0.8, seed=0):
    """
    Mode-conditioned controller:
    - If near wall (x < 0.5 or x > 2.5): use bounce gains
    - Else: use flight gains
    """
    successes = 0
    
    for ep in range(n_episodes):
        env = BouncingBallGravity(restitution=restitution)
        obs = env.reset(seed=ep + seed)
        x, v = obs[0], obs[1]
        
        for step in range(30):
            # Mode gate: near wall/bounce?
            near_wall = 1.0 if (x < 0.5 or x > 2.5) else 0.0
            
            # Select gains based on mode
            if near_wall > 0.5:
                a = k1_b * (env.x_target - x) + k2_b * (-v)
            else:
                a = k1 * (env.x_target - x) + k2 * (-v)
            
            a = np.clip(a, -2.0, 2.0)
            
            obs, _ = env.step(a)
            x, v = obs[0], obs[1]
            
            if abs(x - env.x_target) < env.tau:
                successes += 1
                break
    
    return successes / n_episodes


def cem_optimize_mode_controller(n_iter=50, n_samples=64, n_elite=10, restitution=0.8):
    """
    CEM over 4 parameters: [k1, k2, k1_b, k2_b]
    """
    param_bounds = np.array([
        [-5.0, 5.0],   # k1 (flight)
        [-5.0, 5.0],   # k2 (flight)  
        [-5.0, 5.0],   # k1_b (bounce)
        [-5.0, 5.0],   # k2_b (bounce)
    ])
    
    # Initialize with single PD optimal (k1=1.5, k2=-2.0)
    mean = np.array([1.5, -2.0, 1.5, -2.0])
    std = np.array([1.0, 1.0, 1.0, 1.0])
    
    best_score = -1
    best_params = None
    
    print("CEM optimization over mode-controller gains...")
    
    for iteration in range(n_iter):
        samples = np.random.randn(n_samples, 4) * std + mean
        samples = np.clip(samples, param_bounds[:, 0], param_bounds[:, 1])
        
        scores = []
        for params in samples:
            k1, k2, k1_b, k2_b = params
            score = run_mode_controller(k1, k2, k1_b, k2_b, n_episodes=50, 
                                         restitution=restitution, seed=iteration*100)
            scores.append(score)
        
        scores = np.array(scores)
        
        elite_idx = np.argsort(scores)[-n_elite:]
        elite_params = samples[elite_idx]
        elite_scores = scores[elite_idx]
        
        mean = elite_params.mean(axis=0)
        std = elite_params.std(axis=0) + 0.01
        
        best_iter_idx = np.argmax(scores)
        if scores[best_iter_idx] > best_score:
            best_score = scores[best_iter_idx]
            best_params = samples[best_iter_idx]
        
        if (iteration + 1) % 10 == 0:
            print(f"  Iter {iteration+1}: best={best_score:.1%}, elite_mean={elite_scores.mean():.1%}")
    
    return best_params, best_score


def sweep_restitution(k1, k2, k1_b, k2_b):
    """Success rate vs restitution for single vs mode controller."""
    print("\n" + "="*70)
    print("SWEEP: Restitution (e)")
    print("="*70)
    
    results = {"single": [], "mode": [], "restitution": []}
    
    for e in [0.3, 0.5, 0.7, 0.8, 0.9, 0.95]:
        single_rate = run_single_pd(k1, k2, n_episodes=200, restitution=e)
        mode_rate = run_mode_controller(k1, k2, k1_b, k2_b, n_episodes=200, restitution=e)
        
        results["single"].append(single_rate)
        results["mode"].append(mode_rate)
        results["restitution"].append(e)
        
        print(f"  e={e}: single={single_rate:.1%}, mode={mode_rate:.1%}, gain={mode_rate-single_rate:+.1%}")
    
    return results


def main():
    print("="*70)
    print("PHASE 4: Mode-Event Conditioned Controller (FIXED)")
    print("="*70)
    
    # Verify single PD works (sanity check - should be 71%)
    print("\n1. Verify single PD (sanity check)")
    pd_rate = run_single_pd(1.5, -2.0, n_episodes=200, restitution=0.8)
    print(f"   Single PD (k1=1.5, k2=-2.0): {pd_rate:.1%}")
    
    if abs(pd_rate - 0.71) > 0.05:
        print(f"   WARNING: Expected ~71%, got {pd_rate:.1%}")
    
    # CEM optimize mode controller
    print("\n2. CEM optimization over 4 gain parameters...")
    best_params, best_score = cem_optimize_mode_controller(n_iter=50, n_samples=64)
    k1, k2, k1_b, k2_b = best_params
    print(f"\n   Best gains: k1={k1:.2f}, k2={k2:.2f}, k1_b={k1_b:.2f}, k2_b={k2_b:.2f}")
    print(f"   Best score (50 ep): {best_score:.1%}")
    
    # Full evaluation
    print("\n3. Full evaluation (200 episodes)")
    mode_rate = run_mode_controller(k1, k2, k1_b, k2_b, n_episodes=200, restitution=0.8)
    single_rate = run_single_pd(1.5, -2.0, n_episodes=200, restitution=0.8)
    
    print(f"   Single PD: {single_rate:.1%}")
    print(f"   Mode controller: {mode_rate:.1%}")
    print(f"   Improvement: {mode_rate - single_rate:+.1%}")
    
    # Sweep restitution
    results = sweep_restitution(k1, k2, k1_b, k2_b)
    
    # Summary
    print("\n" + "="*70)
    print("SUMMARY")
    print("="*70)
    print(f"Single PD (e=0.8):        {single_rate:.1%}")
    print(f"Mode controller (e=0.8):  {mode_rate:.1%}")
    print(f"Improvement:              {mode_rate - single_rate:+.1%}")
    
    if mode_rate > single_rate + 0.05:
        print("\n✓ Mode-conditioned gains IMPROVE over single PD!")
        print("  Hybrid dynamics requires mode-dependent feedback.")
    elif mode_rate > single_rate:
        print("\n≈ Mode controller slightly better")
    else:
        print("\n✗ Mode controller did not help")
        print("  Single PD is near-optimal for this task.")


if __name__ == '__main__':
    main()
