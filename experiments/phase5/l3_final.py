#!/usr/bin/env python3
"""
PHASE 5: L3 - Impact Consistency (GPU, Batched)
"""

import torch
import torch.nn as nn
import numpy as np

device = torch.device('cuda')
print(f"Using: {device}")


class BouncingBall:
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
        bounced = False
        if self.x < 0:
            self.x = -self.x * self.e
            self.v = -self.v * self.e
            bounced = True
        elif self.x > 3:
            self.x = 3 - (self.x - 3) * self.e
            self.v = -self.v * self.e
            bounced = True
        return bounced


def generate_batched_data(n_ep, e, seed):
    np.random.seed(seed)
    env = BouncingBall(e)
    
    obs, acts, bounces = [], [], []
    
    for ep in range(n_ep):
        env.reset(seed=ep + seed)
        x = env.x
        v = env.v
        
        ep_obs, ep_acts, ep_bounces = [], [], []
        
        for _ in range(30):
            ep_obs.append([float(x)])
            a = float(np.clip(1.5 * (2.0 - x) + (-2.0) * (-v), -2, 2))
            ep_acts.append([a])
            bounced = env.step(a)
            ep_bounces.append([1.0 if bounced else 0.0])
            x = env.x
            v = env.v
        
        obs.append(ep_obs)
        acts.append(ep_acts)
        bounces.append(ep_bounces)
    
    return np.array(obs, dtype=np.float32), np.array(acts, dtype=np.float32), np.array(bounces, dtype=np.float32)


class ImpactModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.main = nn.Sequential(
            nn.Linear(3, 32),
            nn.Tanh(),
            nn.Linear(32, 32),
            nn.Tanh(),
            nn.Linear(32, 1)
        )
        
    def forward(self, x, a_prev, bounce):
        inp = torch.stack([x, a_prev, bounce], dim=1)
        return torch.tanh(self.main(inp))


def train_l3(n_ep=500, epochs=100):
    print("Generating data...")
    obs, acts, bounces = generate_batched_data(n_ep, 0.8, 42)
    print(f"  Shape: {obs.shape}, bounces: {bounces.sum()}")
    
    obs_t = torch.tensor(obs).to(device)
    acts_t = torch.tensor(acts).to(device)
    bounces_t = torch.tensor(bounces).to(device)
    
    model = ImpactModel().to(device)
    opt = torch.optim.Adam(model.parameters(), lr=0.01)
    
    print(f"Training L3 ({epochs} epochs)...")
    for epoch in range(epochs):
        for t in range(1, 30):
            x = obs_t[:, t, 0]
            a_prev = acts_t[:, t-1, 0]
            bounce = bounces_t[:, t, 0]
            target = acts_t[:, t, 0]
            
            pred = model(x, a_prev, bounce)
            ctrl_loss = ((pred - target) ** 2).mean()
            
            impact_mask = (bounce > 0.5).float()
            if impact_mask.sum() > 0:
                delta = model(x * impact_mask, a_prev * impact_mask, torch.ones_like(bounce))
                actual_delta = (target - pred) * impact_mask
                impact_loss = ((delta - actual_delta) ** 2).mean()
            else:
                impact_loss = 0
            
            loss = ctrl_loss + 0.5 * impact_loss
            
            opt.zero_grad()
            loss.backward()
            opt.step()
        
        if (epoch + 1) % 20 == 0:
            print(f"  Epoch {epoch+1}")
    
    return model


def evaluate(model, n_test=200, e=0.8, seed=0):
    env = BouncingBall(e)
    successes = 0
    
    model.eval()
    with torch.no_grad():
        for ep in range(n_test):
            env.reset(seed=ep + seed + 1000)
            x = env.x
            v = env.v
            last_a = 0.0
            last_bounce = 0.0
            
            for step in range(30):
                x_t = torch.tensor([x], device=device, dtype=torch.float32)
                a_t = torch.tensor([last_a], device=device, dtype=torch.float32)
                b_t = torch.tensor([last_bounce], device=device, dtype=torch.float32)
                
                a = model(x_t, a_t, b_t).item()
                a = float(np.clip(a, -2, 2))
                
                bounced = env.step(a)
                x = env.x
                v = env.v
                
                last_a = a
                last_bounce = 1.0 if bounced else 0.0
                
                if abs(x - 2.0) < 0.3:
                    successes += 1
                    break
    
    return successes / n_test


print("="*70)
print("PHASE 5: L3 - Impact Consistency (GPU)")
print("="*70)

model = train_l3(n_ep=500, epochs=100)

print("\nEvaluating...")
rate = evaluate(model, 200, 0.8, 42)

print("\n" + "="*70)
print("RESULTS")
print("="*70)
print(f"PD full:     71.0%")
print(f"PD partial:  56.5%")
print(f"L3 impact:  {rate:.1%}")

gap = (rate - 0.565) / (0.71 - 0.565)
print(f"\nGap closed: {gap:.1%}")

if rate > 0.60:
    print("\n✓ L3 works!")
else:
    print("\n✗ L3 didn't close gap.")
