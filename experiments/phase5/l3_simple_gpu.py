#!/usr/bin/env python3
"""
PHASE 5: L3 - Simple Impact Model (GPU)

Simplified version: direct mapping from (obs, prev_a) -> action, with impact loss.
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


def generate_data(n_ep, e, seed):
    np.random.seed(seed)
    env = BouncingBall(e)
    
    obs = []
    acts = []
    impacts = []
    
    for ep in range(n_ep):
        env.reset(seed=ep + seed)
        x = env.x
        v = env.v
        
        ep_obs = []
        ep_acts = []
        ep_impacts = []
        
        for step in range(30):
            # Partial obs: x only
            ep_obs.append([float(x)])
            
            # Expert action
            a = 1.5 * (2.0 - x) + (-2.0) * (-v)
            a = float(np.clip(a, -2, 2))
            ep_acts.append([a])
            
            # Step
            bounced = env.step(a)
            ep_impacts.append(1.0 if bounced else 0.0)
            
            x = env.x
            v = env.v
        
        obs.append(ep_obs)
        acts.append(ep_acts)
        impacts.append(ep_impacts)
    
    return np.array(obs, dtype=np.float32), np.array(acts, dtype=np.float32), np.array(impacts, dtype=np.float32)


class SimpleImpactModel(nn.Module):
    """Simple model: (x, prev_a, bounce_flag) -> action"""
    def __init__(self):
        super().__init__()
        # Main network
        self.net = nn.Sequential(
            nn.Linear(3, 32),  # x, prev_a, bounce_flag
            nn.Tanh(),
            nn.Linear(32, 32),
            nn.Tanh(),
            nn.Linear(32, 1)
        )
        
        # Impact model: predicts delta_action at impact
        self.impact_net = nn.Sequential(
            nn.Linear(3, 16),  # x, prev_a, bounce
            nn.Tanh(),
            nn.Linear(16, 1)
        )
    
    def forward(self, x, prev_a, bounce):
        # Ensure all inputs are 1D
        x = x.squeeze() if x.dim() > 1 else x
        prev_a = prev_a.squeeze() if prev_a.dim() > 1 else prev_a
        bounce = bounce.squeeze() if bounce.dim() > 1 else bounce
        inp = torch.stack([x, prev_a, bounce])
        return torch.tanh(self.net(inp.unsqueeze(0))).squeeze()
    
    def impact_correction(self, x, prev_a):
        """Predict action correction at impact."""
        x = x.squeeze() if x.dim() > 1 else x
        prev_a = prev_a.squeeze() if prev_a.dim() > 1 else prev_a
        bounce = torch.ones_like(x)
        inp = torch.stack([x, prev_a, bounce])
        return torch.tanh(self.impact_net(inp.unsqueeze(0))).squeeze()


def train_l3(obs, acts, impacts, epochs=100):
    model = SimpleImpactModel().to(device)
    opt = torch.optim.Adam(model.parameters(), lr=0.001)
    
    n_ep, T, _ = obs.shape
    
    # Move to GPU
    obs_t = torch.tensor(obs).to(device)
    acts_t = torch.tensor(acts).to(device)
    impacts_t = torch.tensor(impacts).to(device)
    
    print(f"Training L3 ({epochs} epochs)...")
    
    for epoch in range(epochs):
        total_loss = 0
        
        for ep in range(n_ep):
            ep_loss = 0
            
            for t in range(T):
                x = obs_t[ep, t]
                prev_a = acts_t[ep, t-1] if t > 0 else torch.zeros(1, device=device)
                bounce = impacts_t[ep, t]
                target = acts_t[ep, t]
                
                # Main prediction
                pred = model(x, prev_a, bounce)
                
                # Controller loss
                ctrl_loss = (pred - target.squeeze()).pow(2).mean()
                
                # Impact loss: when bounce=1, predict correction
                if bounce > 0.5 and t > 0:
                    delta = model.impact_correction(x, prev_a)
                    impact_loss = (delta - (target.squeeze() - pred)).pow(2)
                else:
                    impact_loss = 0
                
                loss = ctrl_loss + 0.5 * impact_loss
                ep_loss += loss
                
                opt.zero_grad()
                loss.backward()
                opt.step()
            
            total_loss += ep_loss.item()
        
        if (epoch + 1) % 20 == 0:
            print(f"  Epoch {epoch+1}: loss={total_loss/n_ep:.4f}")
    
    return model


def evaluate(model, n_test, e, seed):
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
                # Input
                x_t = torch.tensor([[[x]]], dtype=torch.float32).to(device)
                a_t = torch.tensor([[[last_a]]], dtype=torch.float32).to(device)
                b_t = torch.tensor([[[last_bounce]]], dtype=torch.float32).to(device)
                
                # Predict
                pred = model(x_t.squeeze(0), a_t.squeeze(0), b_t.squeeze(0))
                a = float(np.clip(pred.item(), -2, 2))
                
                # Step
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
print("PHASE 5: L3 - Simple Impact Model (GPU)")
print("="*70)

# Generate data
print("\n1. Generating data...")
obs, acts, impacts = generate_data(500, 0.8, 42)
print(f"   Shape: obs={obs.shape}, acts={acts.shape}, impacts={impacts.shape}")
print(f"   Impacts: {impacts.sum()}")

# Train
print("\n2. Training...")
model = train_l3(obs, acts, impacts, epochs=100)

# Evaluate
print("\n3. Evaluating...")
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
