#!/usr/bin/env python3
"""
E2: DAgger-style Closed-Loop Training

Train on mixture of expert trajectories and model rollouts.
"""

import torch
import torch.nn as nn
import numpy as np

device = torch.device('cuda')
print(f"Using: {device}")


class Ball:
    def __init__(self, e=0.8):
        self.e = e
    def reset(self, seed):
        np.random.seed(seed)
        self.x = [0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5][seed % 7]
        self.v = 0.0
    def step(self, a):
        a = np.clip(a, -2, 2)
        self.v += (-9.81 + a) * 0.05
        self.x += self.v * 0.05
        if self.x < 0:
            self.x = -self.x * self.e
            self.v = -self.v * self.e
        if self.x > 3:
            self.x = 3 - (self.x - 3) * self.e
            self.v = -self.v * self.e


class Policy(nn.Module):
    """Simple MLP policy: [x, delta] -> a"""
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(2, 32),
            nn.Tanh(),
            nn.Linear(32, 32),
            nn.Tanh(),
            nn.Linear(32, 1)
        )
    
    def forward(self, x, delta):
        """x, delta: (B,)"""
        inp = torch.stack([x, delta], dim=1).unsqueeze(0)  # (1, B, 2)
        return torch.tanh(self.net(inp)).squeeze(0)


def expert_action(x, v):
    """PD controller on full state"""
    return np.clip(1.5 * (2.0 - x) + (-2.0) * (-v), -2, 2)


def collect_expert_data(n_episodes, seed):
    """Collect expert trajectories"""
    data = {'x': [], 'delta': [], 'a_expert': []}
    
    for ep in range(n_episodes):
        ball = Ball(0.8)
        ball.reset(seed=ep + seed)
        
        x_prev = ball.x
        for step in range(30):
            x = ball.x
            v = ball.v
            
            # Expert action
            a = expert_action(x, v)
            
            # Delta
            d = x - x_prev
            
            data['x'].append(x)
            data['delta'].append(d)
            data['a_expert'].append(a)
            
            ball.step(a)
            x_prev = x
    
    return data


def dagger_iteration(policy, expert_data, model_rollouts, beta, seed):
    """
    One DAgger iteration:
    - Roll out policy (with beta mixing)
    - Query expert at visited states
    - Return augmented dataset
    """
    # Combine expert and model rollouts
    x_all = list(expert_data['x']) + list(model_rollouts['x'])
    delta_all = list(expert_data['delta']) + list(model_rollouts['delta'])
    a_all = list(expert_data['a_expert']) + list(model_rollouts['a_expert'])
    
    return {'x': x_all, 'delta': delta_all, 'a_expert': a_all}


def train_policy(data, epochs=20):
    """Train policy on data"""
    n = len(data['x'])
    
    # Convert to tensors
    X = torch.tensor(np.array([data['x'], data['delta']]).T, dtype=torch.float32).to(device)
    Y = torch.tensor(data['a_expert'], dtype=torch.float32).to(device)
    
    policy = Policy().to(device)
    opt = torch.optim.Adam(policy.parameters(), lr=0.01)
    
    for epoch in range(epochs):
        idx = torch.randperm(n)
        for i in idx:
            x = X[i, 0]
            d = X[i, 1]
            a = Y[i]
            
            pred = policy(x, d)
            loss = (pred - a) ** 2
            
            opt.zero_grad()
            loss.backward()
            opt.step()
    
    return policy


def evaluate(policy, n_test, seed):
    """Evaluate policy"""
    successes = 0
    
    for ep in range(n_test):
        ball = Ball(0.8)
        ball.reset(seed=ep + seed)
        
        x_prev = ball.x
        
        for step in range(30):
            x = ball.x
            d = x - x_prev
            
            # Policy action
            x_t = torch.tensor([x], dtype=torch.float32, device=device)
            d_t = torch.tensor([d], dtype=torch.float32, device=device)
            with torch.no_grad():
                a = policy(x_t, d_t).item()
            a = float(np.clip(a, -2, 2))
            
            ball.step(a)
            x_prev = x
            
            if abs(ball.x - 2.0) < 0.3:
                successes += 1
                break
    
    return successes / n_test


def run_dagger(K=5, M=100, seed=42):
    """Run DAgger"""
    print(f"\nRunning DAgger: K={K} iterations, M={M} episodes/iter")
    
    # Initial expert data
    print("Collecting initial expert data...")
    expert_data = collect_expert_data(M, seed)
    
    # Train initial policy
    print("Training initial policy (expert only)...")
    policy = train_policy(expert_data, epochs=30)
    
    # Evaluate initial
    rate = evaluate(policy, 200, seed)
    print(f"  Iter 0 (expert only): {rate:.1%}")
    
    results = [rate]
    
    # DAgger iterations
    for i in range(1, K + 1):
        print(f"\nIteration {i}:")
        
        # Collect model rollouts
        print(f"  Collecting {M} model rollouts...")
        rollouts = {'x': [], 'delta': [], 'a_expert': []}
        
        beta = 1.0 / (i + 1)  # Decaying beta
        
        for ep in range(M):
            ball = Ball(0.8)
            ball.reset(seed=(i * 1000 + ep + seed))
            
            x_prev = ball.x
            
            for step in range(30):
                x = ball.x
                d = x - x_prev
                
                # Expert action
                a_expert = expert_action(x, ball.v)
                
                # Policy action
                with torch.no_grad():
                    x_t = torch.tensor([x], dtype=torch.float32, device=device)
                    d_t = torch.tensor([d], dtype=torch.float32, device=device)
                    a_policy = policy(x_t, d_t).item()
                
                # Mix: beta * expert + (1-beta) * policy
                if np.random.rand() < beta:
                    a = a_expert
                else:
                    a = float(np.clip(a_policy, -2, 2))
                
                # Record
                rollouts['x'].append(x)
                rollouts['delta'].append(d)
                rollouts['a_expert'].append(a_expert)  # Always label with expert!
                
                ball.step(a)
                x_prev = x
        
        # Aggregate data
        aug_data = dagger_iteration(policy, expert_data, rollouts, beta, seed)
        
        # Retrain
        print(f"  Training on {len(aug_data['x'])} samples...")
        policy = train_policy(aug_data, epochs=20)
        
        # Evaluate
        rate = evaluate(policy, 200, seed)
        print(f"  Iter {i}: {rate:.1%}")
        results.append(rate)
    
    return results


print("="*70)
print("E2: DAgger Closed-Loop Training")
print("="*70)

# Baseline: expert only
print("\n0. Baseline: Expert trajectories only (no DAgger)")
expert_only = collect_expert_data(100, 42)
policy = train_policy(expert_only, epochs=30)
baseline = evaluate(policy, 200, 42)
print(f"   BC baseline: {baseline:.1%}")

# Run DAgger
results = run_dagger(K=5, M=100, seed=42)

print("\n" + "="*70)
print("RESULTS")
print("="*70)
print(f"PD full state:   71.2%")
print(f"PD partial:      56.8%")
print(f"BC (expert):    {baseline:.1%}")
for i, r in enumerate(results):
    print(f"DAgger iter {i}: {r:.1%}")
