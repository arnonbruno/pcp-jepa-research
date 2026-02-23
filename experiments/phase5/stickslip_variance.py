#!/usr/bin/env python3
"""
StickSlip System: Observer Training Variance

Test if physics-informed architecture (F3) improves training stability
in StickSlip hybrid dynamics, similar to BouncingBall.

Protocol:
- Observe position x only (no velocity)
- Estimate velocity v from (x, x_prev)
- Use PD controller with estimated velocity
- Measure success rate over 10 seeds
"""

import torch
import torch.nn as nn
import numpy as np

device = torch.device('cuda')

# =============================================================================
# STICK-SLIP DYNAMICS
# =============================================================================

class StickSlipBlock:
    """Stick-slip friction dynamics."""
    
    def __init__(self, tau=0.2):
        self.dt = 0.02
        self.x_target = 2.0
        self.tau = tau
        self.F_static = 1.5
        self.F_kinetic = 1.0
        self.v_thresh = 0.01
        self.mass = 1.0
        self.mode = 'stick'
    
    def reset(self, x0):
        self.x = x0
        self.v = 0.0
        self.mode = 'stick'
    
    def step(self, u):
        u = np.clip(u, -3.0, 3.0)
        
        # Mode transition
        if self.mode == 'stick':
            if abs(u) > self.F_static:
                self.mode = 'slip'
                self.v = np.sign(u) * 0.01
        else:
            if abs(self.v) < self.v_thresh and abs(u) < self.F_static:
                self.mode = 'stick'
                self.v = 0.0
        
        # Dynamics
        if self.mode == 'slip':
            friction = -np.sign(self.v) * self.F_kinetic
            a = (u + friction) / self.mass
            self.v += a * self.dt
            self.x += self.v * self.dt
        
        # Boundaries
        self.x = np.clip(self.x, -1.0, 4.0)
        if self.x <= -1.0 or self.x >= 4.0:
            self.v = 0.0

# =============================================================================
# OBSERVER MODELS
# =============================================================================

class BaselineObserver(nn.Module):
    """Raw (x, x_prev) input."""
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(2, 64), nn.ReLU(),
            nn.Linear(64, 64), nn.ReLU(),
            nn.Linear(64, 1)
        )
    
    def forward(self, x, x_prev):
        return self.net(torch.cat([x, x_prev], dim=-1))


class F3Observer(nn.Module):
    """Physics-normalized: v = Δx/dt + correction."""
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(2, 64), nn.ReLU(),
            nn.Linear(64, 64), nn.ReLU(),
            nn.Linear(64, 1)
        )
    
    def forward(self, x, x_prev):
        delta_v = (x - x_prev) / 0.02  # dt = 0.02
        return delta_v + self.net(torch.cat([x, delta_v], dim=-1))

# =============================================================================
# TRAINING
# =============================================================================

def generate_expert_data(n_episodes=300):
    """Generate training data from PD controller with full state."""
    X, Xp, V = [], [], []
    
    start_positions = [0.0, 0.5, 1.0, 1.5, 2.5, 3.0, 3.5]
    
    for ep in range(n_episodes):
        x0 = start_positions[ep % len(start_positions)]
        env = StickSlipBlock()
        env.reset(x0)
        xp = x0
        
        for _ in range(50):
            # PD controller with true velocity
            u = 2.0 * (2.0 - env.x) + 1.0 * (-env.v)
            u = np.clip(u, -3.0, 3.0)
            
            X.append(env.x)
            Xp.append(xp)
            V.append(env.v)
            
            xp = env.x
            env.step(u)
    
    return np.array(X), np.array(Xp), np.array(V)

def train_baseline(seed, epochs=30):
    torch.manual_seed(seed)
    np.random.seed(seed)
    
    X, Xp, V = generate_expert_data(300)
    X_t = torch.tensor(X, dtype=torch.float32, device=device).unsqueeze(-1)
    Xp_t = torch.tensor(Xp, dtype=torch.float32, device=device).unsqueeze(-1)
    V_t = torch.tensor(V, dtype=torch.float32, device=device).unsqueeze(-1)
    
    obs = BaselineObserver().to(device)
    opt = torch.optim.Adam(obs.parameters(), lr=0.01)
    
    for _ in range(epochs):
        idx = torch.randperm(len(X_t))
        for i in range(0, len(X_t), 64):
            b = idx[i:i+64]
            loss = ((obs(X_t[b], Xp_t[b]) - V_t[b])**2).mean()
            opt.zero_grad()
            loss.backward()
            opt.step()
    
    return obs

def train_f3(seed, epochs=30):
    torch.manual_seed(seed)
    np.random.seed(seed)
    
    X, Xp, V = generate_expert_data(300)
    X_t = torch.tensor(X, dtype=torch.float32, device=device).unsqueeze(-1)
    Xp_t = torch.tensor(Xp, dtype=torch.float32, device=device).unsqueeze(-1)
    V_t = torch.tensor(V, dtype=torch.float32, device=device).unsqueeze(-1)
    
    obs = F3Observer().to(device)
    opt = torch.optim.Adam(obs.parameters(), lr=0.01)
    
    for _ in range(epochs):
        idx = torch.randperm(len(X_t))
        for i in range(0, len(X_t), 64):
            b = idx[i:i+64]
            loss = ((obs(X_t[b], Xp_t[b]) - V_t[b])**2).mean()
            opt.zero_grad()
            loss.backward()
            opt.step()
    
    return obs

# =============================================================================
# EVALUATION
# =============================================================================

def evaluate(obs, n_trials=50):
    """Evaluate observer with PD controller."""
    success = 0
    
    start_positions = [0.0, 0.5, 1.0, 1.5, 2.5, 3.0, 3.5]
    
    for x0 in start_positions:
        for _ in range(n_trials):
            env = StickSlipBlock()
            env.reset(x0)
            xp = x0
            
            for _ in range(50):
                with torch.no_grad():
                    v_est = obs(
                        torch.tensor([[env.x]], dtype=torch.float32, device=device),
                        torch.tensor([[xp]], dtype=torch.float32, device=device)
                    ).item()
                
                u = 2.0 * (2.0 - env.x) + 1.0 * (-v_est)
                u = np.clip(u, -3.0, 3.0)
                
                xp = env.x
                env.step(u)
                
                if abs(env.x - 2.0) < 0.2:
                    success += 1
                    break
    
    return success / (len(start_positions) * n_trials)

def evaluate_fd():
    """FD baseline: v = (x - x_prev) / dt."""
    success = 0
    
    start_positions = [0.0, 0.5, 1.0, 1.5, 2.5, 3.0, 3.5]
    
    for x0 in start_positions:
        for _ in range(50):
            env = StickSlipBlock()
            env.reset(x0)
            xp = x0
            
            for _ in range(50):
                v_fd = (env.x - xp) / 0.02
                u = 2.0 * (2.0 - env.x) + 1.0 * (-v_fd)
                u = np.clip(u, -3.0, 3.0)
                
                xp = env.x
                env.step(u)
                
                if abs(env.x - 2.0) < 0.2:
                    success += 1
                    break
    
    return success / (len(start_positions) * 50)

# =============================================================================
# MAIN
# =============================================================================

if __name__ == '__main__':
    print('='*70)
    print('StickSlip: Observer Training Variance')
    print('='*70)
    
    # FD baseline
    fd_rate = evaluate_fd()
    print(f'\nFD baseline: {fd_rate:.1%}')
    
    # Train and evaluate
    print('\nSeed | Baseline | F3')
    print('-'*40)
    
    baseline_results = []
    f3_results = []
    
    for seed in range(10):
        obs_b = train_baseline(seed)
        obs_f3 = train_f3(seed)
        
        r_b = evaluate(obs_b)
        r_f3 = evaluate(obs_f3)
        
        baseline_results.append(r_b)
        f3_results.append(r_f3)
        
        print(f'{seed:4d} | {r_b:.1%}     | {r_f3:.1%}')
    
    print('='*40)
    print(f'Mean: {np.mean(baseline_results):.1%}     | {np.mean(f3_results):.1%}')
    print(f'Std:  {np.std(baseline_results):.1%}      | {np.std(f3_results):.1%}')
    print(f'FD:   {fd_rate:.1%}')
    
    # Count seeds near FD
    n_b = sum(1 for r in baseline_results if r > fd_rate - 0.05)
    n_f3 = sum(1 for r in f3_results if r > fd_rate - 0.05)
    print(f'\nSeeds near FD: Baseline={n_b}/10, F3={n_f3}/10')
