#!/usr/bin/env python3
"""
PHASE 4: Nonlinear Controller Experiments

Check A: Tanh-PD
  a = clip(tanh(gamma * (k1*(x* - x) + k2*(-v))), [-2, 2])

Check B: PID
  a = clip(k1*e + k2*de + k3*sum(e), [-2, 2])
  where e = x* - x

Goal: Beat 71% ceiling with tiny nonlinearity or integral term.
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
        
        if self.x < 0:
            self.x = -self.x * self.e
            self.v = -self.v * self.e
        if self.x > 3:
            self.x = 3 - (self.x - 3) * self.e
            self.v = -self.v * self.e
            
        return np.array([self.x, self.v])


# ============================================================================
# CONTROLLERS
# ============================================================================

def run_single_pd(k1, k2, n_episodes=200, restitution=0.8, seed=0):
    """Baseline PD: a = clip(k1*(x* - x) + k2*(-v))"""
    successes = 0
    
    for ep in range(n_episodes):
        env = BouncingBallGravity(restitution=restitution)
        obs = env.reset(seed=ep + seed)
        x, v = obs[0], obs[1]
        
        for step in range(30):
            a = k1 * (env.x_target - x) + k2 * (-v)
            a = np.clip(a, -2.0, 2.0)
            
            obs = env.step(a)
            x, v = obs[0], obs[1]
            
            if abs(x - env.x_target) < env.tau:
                successes += 1
                break
    
    return successes / n_episodes


def run_tanh_pd(k1, k2, gamma, n_episodes=200, restitution=0.8, seed=0):
    """
    Tanh-PD: a = clip(tanh(gamma * (k1*e + k2*(-v))), [-2, 2])
    """
    successes = 0
    
    for ep in range(n_episodes):
        env = BouncingBallGravity(restitution=restitution)
        obs = env.reset(seed=ep + seed)
        x, v = obs[0], obs[1]
        
        for step in range(30):
            e = env.x_target - x  # error
            a_raw = k1 * e + k2 * (-v)
            a = np.tanh(gamma * a_raw) * 2.0  # scale to [-2, 2]
            a = np.clip(a, -2.0, 2.0)
            
            obs = env.step(a)
            x, v = obs[0], obs[1]
            
            if abs(x - env.x_target) < env.tau:
                successes += 1
                break
    
    return successes / n_episodes


def run_pid(k1, k2, k3, n_episodes=200, restitution=0.8, seed=0):
    """
    PID: a = clip(k1*e + k2*de + k3*sum(e))
    """
    successes = 0
    
    for ep in range(n_episodes):
        env = BouncingBallGravity(restitution=restitution)
        obs = env.reset(seed=ep + seed)
        x, v = obs[0], obs[1]
        
        integral = 0.0
        prev_e = env.x_target - x
        
        for step in range(30):
            e = env.x_target - x
            de = e - prev_e  # derivative of error
            prev_e = e
            
            integral += e
            
            a = k1 * e + k2 * de + k3 * integral
            a = np.clip(a, -2.0, 2.0)
            
            obs = env.step(a)
            x, v = obs[0], obs[1]
            
            if abs(x - env.x_target) < env.tau:
                successes += 1
                break
    
    return successes / n_episodes


def run_deadzone_pd(k1, k2, delta, n_episodes=200, restitution=0.8, seed=0):
    """
    Deadzone-PD: reduce gain when error is small
    """
    successes = 0
    
    for ep in range(n_episodes):
        env = BouncingBallGravity(restitution=restitution)
        obs = env.reset(seed=ep + seed)
        x, v = obs[0], obs[1]
        
        for step in range(30):
            e = env.x_target - x
            
            # Deadzone: if |e| < delta, reduce gain
            if abs(e) < delta:
                k1_eff = k1 * 0.1  # reduced gain in deadzone
            else:
                k1_eff = k1
            
            a = k1_eff * e + k2 * (-v)
            a = np.clip(a, -2.0, 2.0)
            
            obs = env.step(a)
            x, v = obs[0], obs[1]
            
            if abs(x - env.x_target) < env.tau:
                successes += 1
                break
    
    return successes / n_episodes


# ============================================================================
# CEM OPTIMIZATION
# ============================================================================

def cem_optimize(controller_type, n_iter=50, n_samples=64, n_elite=10, restitution=0.8):
    """
    CEM over controller parameters.
    """
    if controller_type == 'tanh_pd':
        # [k1, k2, gamma]
        bounds = np.array([[0, 5], [-5, 0], [0.1, 5.0]])
        initial = np.array([1.5, -2.0, 1.0])
        def run_fn(params): return run_tanh_pd(*params, restitution=restitution)
    elif controller_type == 'pid':
        # [k1, k2, k3]
        bounds = np.array([[0, 5], [-5, 0], [-1, 1]])
        initial = np.array([1.5, -2.0, 0.0])
        def run_fn(params): return run_pid(*params, restitution=restitution)
    elif controller_type == 'deadzone':
        # [k1, k2, delta]
        bounds = np.array([[0, 5], [-5, 0], [0, 1.0]])
        initial = np.array([1.5, -2.0, 0.1])
        def run_fn(params): return run_deadzone_pd(*params, restitution=restitution)
    
    mean = initial.copy()
    std = np.array([1.0, 1.0, 0.5])
    
    best_score = -1
    best_params = None
    
    print(f"CEM over {controller_type}...")
    
    for iteration in range(n_iter):
        samples = np.random.randn(n_samples, 3) * std + mean
        samples = np.clip(samples, bounds[:, 0], bounds[:, 1])
        
        scores = []
        for params in samples:
            score = run_fn(params)
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


def main():
    print("="*70)
    print("PHASE 4: Nonlinear Controller Experiments")
    print("="*70)
    
    restitution = 0.8
    n_test = 500  # larger test set for confidence
    
    # Baseline
    print("\n0. Baseline (single PD)")
    baseline = run_single_pd(1.5, -2.0, n_episodes=n_test, restitution=restitution)
    print(f"   PD (k1=1.5, k2=-2.0): {baseline:.1%}")
    
    # Check A: Tanh-PD
    print("\n" + "="*70)
    print("CHECK A: Tanh-PD")
    print("="*70)
    best_tanh, score_tanh = cem_optimize('tanh_pd', n_iter=50, restitution=restitution)
    print(f"   Best params: k1={best_tanh[0]:.2f}, k2={best_tanh[1]:.2f}, gamma={best_tanh[2]:.2f}")
    
    # Full eval
    tanh_rate = run_tanh_pd(best_tanh[0], best_tanh[1], best_tanh[2], n_episodes=n_test, restitution=restitution)
    print(f"   Tanh-PD ({n_test} eps): {tanh_rate:.1%}")
    
    # Check B: PID
    print("\n" + "="*70)
    print("CHECK B: PID")
    print("="*70)
    best_pid, score_pid = cem_optimize('pid', n_iter=50, restitution=restitution)
    print(f"   Best params: k1={best_pid[0]:.2f}, k2={best_pid[1]:.2f}, k3={best_pid[2]:.2f}")
    
    pid_rate = run_pid(best_pid[0], best_pid[1], best_pid[2], n_episodes=n_test, restitution=restitution)
    print(f"   PID ({n_test} eps): {pid_rate:.1%}")
    
    # Check C: Deadzone (bonus)
    print("\n" + "="*70)
    print("CHECK C: Deadzone-PD")
    print("="*70)
    best_dz, score_dz = cem_optimize('deadzone', n_iter=50, restitution=restitution)
    print(f"   Best params: k1={best_dz[0]:.2f}, k2={best_dz[1]:.2f}, delta={best_dz[2]:.2f}")
    
    dz_rate = run_deadzone_pd(best_dz[0], best_dz[1], best_dz[2], n_episodes=n_test, restitution=restitution)
    print(f"   Deadzone-PD ({n_test} eps): {dz_rate:.1%}")
    
    # Summary
    print("\n" + "="*70)
    print("SUMMARY")
    print("="*70)
    print(f"Baseline PD:          {baseline:.1%}")
    print(f"Tanh-PD:              {tanh_rate:.1%} (delta: {tanh_rate-baseline:+.1%})")
    print(f"PID:                  {pid_rate:.1%} (delta: {pid_rate-baseline:+.1%})")
    print(f"Deadzone-PD:          {dz_rate:.1%} (delta: {dz_rate-baseline:+.1%})")
    
    # Determine ceiling
    max_rate = max(baseline, tanh_rate, pid_rate, dz_rate)
    if max_rate > baseline + 0.03:
        print(f"\n✓ {max_rate:.1%} BEATS baseline! 71% is not the ceiling.")
    else:
        print(f"\n✓ Baseline is optimal: {baseline:.1%} ≈ {max_rate:.1%}")
        print("  71% is the ceiling for this controller family.")


if __name__ == '__main__':
    main()
