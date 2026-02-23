#!/usr/bin/env python3
"""
GO/NO-GO TESTS for Phase 4

Test 1: Imitation-only upper bound (no JEPA)
- Train encoder + linear head to directly predict optimal feedback actions
- No predictive loss, no dynamics model
- If this reaches ~70% → Option B will work

Test 2: Linearity check on ground truth
- Fit best linear controller on true state
- If can't reach ~70% → linear controllability insufficient
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
        self.bounce_count = 0
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
            self.bounce_count += 1
        if self.x > 3:
            self.x = 3 - (self.x - 3) * self.e
            self.v = -self.v * self.e
            bounced = True
            self.bounce_count += 1
            
        return np.array([self.x, self.v]), bounced


# ============================================================================
# TEST 1: Imitation-Only Upper Bound
# ============================================================================

def test_imitation_only():
    """
    Train simple encoder + linear head to predict optimal actions.
    No JEPA, no dynamics - just state → action mapping.
    """
    print("\n" + "="*70)
    print("TEST 1: Imitation-Only Upper Bound (No JEPA)")
    print("="*70)
    
    # Collect training data with optimal feedback actions
    n_traj = 500
    horizon = 30
    states = []
    actions = []
    
    for ep in range(n_traj):
        env = BouncingBallGravity(restitution=0.8)
        obs = env.reset(seed=ep)
        x, v = obs[0], obs[1]
        
        for step in range(horizon):
            # Optimal feedback action (PD controller)
            a_opt = 1.5 * (2.0 - x) + (-2.0) * (-v)
            a_opt = np.clip(a_opt, -2.0, 2.0)
            
            states.append([x, v])
            actions.append(a_opt)
            
            # Random step to collect diverse states
            a = np.random.uniform(-2.0, 2.0)
            obs, _ = env.step(a)
            x, v = obs[0], obs[1]
    
    states = np.array(states)
    actions = np.array(actions)
    
    print(f"Collected {len(states)} state-action pairs")
    
    # Simple feature encoder: [x, v, x^2, v^2, x*v, near_wall]
    def encode_features(s):
        x, v = s
        near_wall = 1.0 if (x < 0.5 or x > 2.5) else 0.0
        return np.array([x, v, x**2, v**2, x*v, near_wall, 1.0])  # + bias
    
    X = np.array([encode_features(s) for s in states])
    
    # Linear regression: action = w^T * features
    w = np.linalg.lstsq(X, actions, rcond=None)[0]
    
    # Evaluate
    successes = 0
    n_episodes = 200
    
    for ep in range(n_episodes):
        env = BouncingBallGravity(restitution=0.8)
        obs = env.reset(seed=ep + 1000)
        x, v = obs[0], obs[1]
        
        for step in range(30):
            features = encode_features([x, v])
            a = np.dot(w, features)
            a = np.clip(a, -2.0, 2.0)
            
            obs, _ = env.step(a)
            x, v = obs[0], obs[1]
            
            if abs(x - env.x_target) < env.tau:
                successes += 1
                break
    
    rate = successes / n_episodes
    print(f"\nImitation-only success rate: {rate:.1%}")
    return rate


# ============================================================================
# TEST 2: Linearity Check on Ground Truth State
# ============================================================================

def test_linear_on_ground_truth():
    """
    Fit best linear controller on true state.
    a = w^T * [x-x*, v, near_wall]
    Compare to hand-designed PD.
    """
    print("\n" + "="*70)
    print("TEST 2: Linearity Check on Ground Truth State")
    print("="*70)
    
    # Collect training data
    n_traj = 500
    horizon = 30
    states = []
    actions = []
    
    for ep in range(n_traj):
        env = BouncingBallGravity(restitution=0.8)
        obs = env.reset(seed=ep)
        x, v = obs[0], obs[1]
        
        for step in range(horizon):
            # Optimal feedback action
            a_opt = 1.5 * (2.0 - x) + (-2.0) * (-v)
            a_opt = np.clip(a_opt, -2.0, 2.0)
            
            # Feature vector: [x-x*, v, x^2, v^2, near_wall]
            x_err = x - 2.0
            near_wall = 1.0 if (x < 0.5 or x > 2.5) else 0.0
            features = np.array([x_err, v, x**2, v**2, near_wall, 1.0])
            
            states.append(features)
            actions.append(a_opt)
            
            a = np.random.uniform(-2.0, 2.0)
            obs, _ = env.step(a)
            x, v = obs[0], obs[1]
    
    X = np.array(states)
    y = np.array(actions)
    
    # Linear regression
    w = np.linalg.lstsq(X, y, rcond=None)[0]
    print(f"Learned weights: {w[:-1]}")  # exclude bias
    
    # Compare to hand-designed PD
    print("\nHand-designed PD: a = 1.5*(x* - x) + 2*v")
    
    # Evaluate learned linear controller
    successes = 0
    n_episodes = 200
    
    for ep in range(n_episodes):
        env = BouncingBallGravity(restitution=0.8)
        obs = env.reset(seed=ep + 1000)
        x, v = obs[0], obs[1]
        
        for step in range(30):
            x_err = x - 2.0
            near_wall = 1.0 if (x < 0.5 or x > 2.5) else 0.0
            features = np.array([x_err, v, x**2, v**2, near_wall, 1.0])
            
            a = np.dot(w, features)
            a = np.clip(a, -2.0, 2.0)
            
            obs, _ = env.step(a)
            x, v = obs[0], obs[1]
            
            if abs(x - env.x_target) < env.tau:
                successes += 1
                break
    
    rate_linear = successes / n_episodes
    
    # Also test hand-designed PD
    successes_pd = 0
    for ep in range(n_episodes):
        env = BouncingBallGravity(restitution=0.8)
        obs = env.reset(seed=ep + 1000)
        x, v = obs[0], obs[1]
        
        for step in range(30):
            a = 1.5 * (2.0 - x) + (-2.0) * (-v)
            a = np.clip(a, -2.0, 2.0)
            
            obs, _ = env.step(a)
            x, v = obs[0], obs[1]
            
            if abs(x - env.x_target) < env.tau:
                successes_pd += 1
                break
    
    rate_pd = successes_pd / n_episodes
    
    print(f"\nHand-designed PD success: {rate_pd:.1%}")
    print(f"Learned linear on state: {rate_linear:.1%}")
    
    return rate_linear, rate_pd


# ============================================================================
# MAIN
# ============================================================================

if __name__ == '__main__':
    # Test 1: Imitation only
    imitation_rate = test_imitation_only()
    
    # Test 2: Linearity on ground truth
    linear_gt_rate, pd_rate = test_linear_on_ground_truth()
    
    # Summary
    print("\n" + "="*70)
    print("GO/NO-GO SUMMARY")
    print("="*70)
    print(f"Test 1 - Imitation-only (no JEPA):    {imitation_rate:.1%}")
    print(f"Test 2 - Linear on true state:         {linear_gt_rate:.1%}")
    print(f"Test 2 - Hand-designed PD:            {pd_rate:.1%}")
    print(f"Target (state feedback upper bound):   71.0%")
    print()
    
    if imitation_rate >= 0.65 and linear_gt_rate >= 0.65:
        print("✓ GO: Both tests pass! → Implement Option B")
    elif imitation_rate < 0.60:
        print("✗ NO-GO: Test 1 failed → Linear latent controller insufficient")
        print("  Need: mode/event conditioning or piecewise policy")
    elif linear_gt_rate < 0.60:
        print("✗ NO-GO: Test 2 failed → Linear on state can't reach 70%")
        print("  Need: nonlinear/piecewise controller")
    else:
        print("? MARGINAL: Close but not quite → Try Option B anyway")
