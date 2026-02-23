#!/usr/bin/env python3
"""
PHASE 3: Sanity Check + Dimension vs Success Curve

1. Verify tuned gains generalize across different initial-state sets
2. Success vs parameter dimension (action sequence → knots → gains → neural)
"""

import numpy as np
import json
import os


class BouncingBallGravity:
    def __init__(self, tau=0.3, seed_offset=0):
        self.g = 9.81
        self.e = 0.8
        self.dt = 0.05
        self.x_target = 2.0
        self.tau = tau
        self.seed_offset = seed_offset
        
    def reset(self, seed=None):
        if seed is not None:
            np.random.seed(seed + self.seed_offset)
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


# ============== DIMENSION vs SUCCESS ==============

def run_action_sequence(dim=30, n_samples=64, temperature=1.0, seed=0):
    """Action sequence: dim = horizon H"""
    env = BouncingBallGravity(seed_offset=seed*1000)
    obs = env.reset(seed=seed)
    x, v = obs[0], obs[1]
    
    for step in range(30):
        actions = np.random.uniform(-2.0, 2.0, (n_samples, dim))
        
        costs = []
        for i in range(n_samples):
            _, cost = true_physics_rollout(x, v, actions[i])
            costs.append(cost)
        
        costs = np.array(costs)
        min_cost = costs.min()
        weights = np.exp(-(costs - min_cost) / temperature)
        weights = weights / weights.sum()
        
        action = (actions.T @ weights)[0]
        
        obs = env.step(action)
        x, v = obs[0], obs[1]
        
        if abs(x - env.x_target) < env.tau:
            return True
    
    return False


def run_knots(k=5, n_samples=64, temperature=1.0, seed=0):
    """Knots: dim = K"""
    horizon = 30
    
    env = BouncingBallGravity(seed_offset=seed*1000)
    obs = env.reset(seed=seed)
    x, v = obs[0], obs[1]
    
    for step in range(30):
        # Sample knot actions
        knot_actions = np.random.uniform(-2.0, 2.0, (n_samples, k))
        
        costs = []
        for i in range(n_samples):
            # Interpolate knots to horizon
            full_actions = np.repeat(knot_actions[i], horizon // k)
            if len(full_actions) < horizon:
                full_actions = np.concatenate([full_actions, np.full(horizon - len(full_actions), knot_actions[i][-1])])
            
            _, cost = true_physics_rollout(x, v, full_actions)
            costs.append(cost)
        
        costs = np.array(costs)
        min_cost = costs.min()
        weights = np.exp(-(costs - min_cost) / temperature)
        weights = weights / weights.sum()
        
        knot_result = knot_actions.T @ weights
        action = knot_result[step % k]
        
        obs = env.step(action)
        x, v = obs[0], obs[1]
        
        if abs(x - env.x_target) < env.tau:
            return True
    
    return False


def run_gains(k1=1.5, k2=-2.0, n_samples=0, temperature=0, seed=0):
    """Fixed feedback gains: dim = 2"""
    env = BouncingBallGravity(seed_offset=seed*1000)
    obs = env.reset(seed=seed)
    x, v = obs[0], obs[1]
    
    for step in range(30):
        a = k1 * (2.0 - x) + k2 * (-v)
        a = np.clip(a, -2.0, 2.0)
        
        obs = env.step(a)
        x, v = obs[0], obs[1]
        
        if abs(x - env.x_target) < env.tau:
            return True
    
    return False


def run_gains_mppi(n_samples=500, temperature=1.0, seed=0):
    """MPPI over gains: dim = 2"""
    env = BouncingBallGravity(seed_offset=seed*1000)
    obs = env.reset(seed=seed)
    x, v = obs[0], obs[1]
    
    for step in range(30):
        # Sample gain pairs
        k1_samples = np.random.uniform(-3.0, 5.0, n_samples)
        k2_samples = np.random.uniform(-3.0, 3.0, n_samples)
        
        costs = []
        for i in range(n_samples):
            # Execute policy
            x_test, v_test = x, v
            for _ in range(30):
                a = np.clip(k1_samples[i] * (2.0 - x_test) + k2_samples[i] * (-v_test), -2.0, 2.0)
                v_test += (-9.81 + a) * 0.05
                x_test += v_test * 0.05
                if x_test < 0:
                    x_test = -x_test * 0.8
                    v_test = -v_test * 0.8
                elif x_test > 3:
                    x_test = 3 - (x_test - 3) * 0.8
                    v_test = -v_test * 0.8
            
            cost = (x_test - 2.0) ** 2
            costs.append(cost)
        
        costs = np.array(costs)
        min_cost = costs.min()
        weights = np.exp(-(costs - min_cost) / temperature)
        weights = weights / weights.sum()
        
        k1 = (k1_samples * weights).sum()
        k2 = (k2_samples * weights).sum()
        
        a = np.clip(k1 * (2.0 - x) + k2 * (-v), -2.0, 2.0)
        
        obs = env.step(a)
        x, v = obs[0], obs[1]
        
        if abs(x - env.x_target) < env.tau:
            return True
    
    return False


def run_neural_policy(n_params=20, n_samples=200, temperature=1.0, seed=0):
    """Simple neural policy: dim = n_params"""
    # Simple MLP: 2 inputs (x, v) → 2 hidden → 1 output
    # Represent as flat parameter vector
    
    horizon = 30
    env = BouncingBallGravity(seed_offset=seed*1000)
    obs = env.reset(seed=seed)
    x, v = obs[0], obs[1]
    
    for step in range(30):
        # Sample neural policies
        policies = []
        for _ in range(n_samples):
            # Random weights
            W1 = np.random.randn(2, 4) * 0.5
            b1 = np.random.randn(4) * 0.5
            W2 = np.random.randn(4, 1) * 0.5
            b2 = np.random.randn(1) * 0.5
            params = np.concatenate([W1.flatten(), b1, W2.flatten(), b2])
            policies.append(params)
        
        policies = np.array(policies)
        
        costs = []
        for i in range(n_samples):
            # Execute policy
            x_test, v_test = x, v
            for _ in range(horizon):
                # Forward pass
                h = np.tanh(np.dot([x_test, v_test], policies[i][:8].reshape(2,4)) + policies[i][8:12])
                a = np.dot(h, policies[i][12:16].reshape(4,1)).item() + policies[i][16]
                a = np.clip(a, -2.0, 2.0)
                
                v_test += (-9.81 + a) * 0.05
                x_test += v_test * 0.05
                if x_test < 0:
                    x_test = -x_test * 0.8
                    v_test = -v_test * 0.8
                elif x_test > 3:
                    x_test = 3 - (x_test - 3) * 0.8
                    v_test = -v_test * 0.8
            
            cost = (x_test - 2.0) ** 2
            costs.append(cost)
        
        costs = np.array(costs)
        min_cost = costs.min()
        weights = np.exp(-(costs - min_cost) / temperature)
        weights = weights / weights.sum()
        
        # Get best policy
        best_idx = np.argmax(weights)
        params = policies[best_idx]
        
        h = np.tanh(np.dot([x, v], params[:8].reshape(2,4)) + params[8:12])
        a = np.dot(h, params[12:16].reshape(4,1)).item() + params[16]
        a = np.clip(a, -2.0, 2.0)
        
        obs = env.step(a)
        x, v = obs[0], obs[1]
        
        if abs(x - env.x_target) < env.tau:
            return True
    
    return False


def main():
    print("="*70)
    print("DIMENSION vs SUCCESS CURVE")
    print("="*70)
    
    n_episodes = 200
    methods = [
        ('Action seq (H=30)', lambda s: run_action_sequence(dim=30, seed=s)),
        ('Knots K=10', lambda s: run_knots(k=10, seed=s)),
        ('Knots K=5', lambda s: run_knots(k=5, seed=s)),
        ('Fixed gains', lambda s: run_gains(k1=1.5, k2=-2.0, seed=s)),
        ('MPPI gains', lambda s: run_gains_mppi(seed=s)),
        ('Neural 20p', lambda s: run_neural_policy(n_params=20, seed=s)),
    ]
    
    results = []
    
    for name, fn in methods:
        print(f"\n{name}...", end=" ", flush=True)
        
        successes = 0
        for ep in range(n_episodes):
            if fn(ep):
                successes += 1
        
        rate = successes / n_episodes
        print(f"Success: {rate:.1%}")
        results.append({'method': name, 'success': rate})
    
    # Sanity check: fixed gains across 3 seed batches
    print("\n" + "="*70)
    print("SANITY CHECK: Fixed gains across seed batches")
    print("="*70)
    
    for offset in [0, 10000, 20000]:
        env = BouncingBallGravity(seed_offset=offset)
        successes = 0
        for ep in range(100):
            x, v = env.reset(ep)[0], env.reset(ep)[1]
            for _ in range(30):
                a = np.clip(1.5 * (2.0 - x) + (-2.0) * (-v), -2.0, 2.0)
                env.step(a)
                x, v = env.x, env.v
                if abs(x - 2.0) < 0.3:
                    successes += 1
                    break
        
        print(f"Seed offset {offset}: {successes/100:.1%}")
    
    # Save
    output_dir = '/home/ulluboz/pcp-jepa-research/results/phase3'
    os.makedirs(output_dir, exist_ok=True)
    
    with open(f'{output_dir}/dimension_vs_success.json', 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"\nSaved to {output_dir}/dimension_vs_success.json")
    
    # Summary table
    print("\n" + "="*70)
    print("SUMMARY")
    print("="*70)
    print(f"\n{'Method':<20} {'Dim':<8} {'Success':<10}")
    print("-"*40)
    
    dim_map = {'Action seq (H=30)': 30, 'Knots K=10': 10, 'Knots K=5': 5, 
               'Fixed gains': 2, 'MPPI gains': 2, 'Neural 20p': 20}
    
    for r in results:
        dim = dim_map.get(r['method'], '?')
        print(f"{r['method']:<20} {dim:<8} {r['success']:>7.1%}")


if __name__ == '__main__':
    main()
