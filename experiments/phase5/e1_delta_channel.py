#!/usr/bin/env python3
"""
E1: Add explicit delta channel to JEPA belief

Feed: x_t AND delta_x = x_t - x_{t-1}

This gives the model explicit temporal differencing signal.
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
        self.x = [0.5,1,1.5,2,2.5,3,3.5][seed % 7]
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


def generate_data(n_ep, e, seed):
    np.random.seed(seed)
    ball = Ball(e)
    
    obs = []      # [x]
    delta = []    # [x - x_prev]
    acts = []
    bounces = []
    
    for ep in range(n_ep):
        ball.reset(seed=ep + seed)
        x_prev = ball.x
        x_hist = [ball.x]
        
        for _ in range(30):
            x = ball.x
            
            # Delta channel: x - x_prev
            d = x - x_prev
            
            # Expert action
            a = 1.5 * (2.0 - x) + (-2.0) * (-ball.v)
            a = float(np.clip(a, -2, 2))
            
            # Step
            bounced = ball.step(a)
            
            obs.append([x])
            delta.append([d])
            acts.append([a])
            bounces.append(1.0 if bounced else 0.0)
            
            x_prev = x
    
    return (np.array(obs, dtype=np.float32), 
            np.array(delta, dtype=np.float32),
            np.array(acts, dtype=np.float32),
            np.array(bounces, dtype=np.float32))


class DeltaModel(nn.Module):
    """
    JEPA with explicit delta channel:
    - Input: x, delta_x (concatenated)
    - Encoder processes both
    - GRU for temporal context
    """
    def __init__(self):
        super().__init__()
        # Input: x + delta = 2
        self.encoder = nn.Sequential(
            nn.Linear(2, 16),  # process [x, delta]
            nn.Tanh()
        )
        # GRU: latent + action
        self.gru = nn.GRUCell(16 + 1, 24)  # z + a_prev -> belief
        # Controller
        self.ctrl = nn.Linear(24, 1)
        
    def forward(self, x, delta, prev_a):
        """
        x: (B,) position
        delta: (B,) position delta  
        prev_a: (B,) previous action
        """
        # Concatenate [x, delta]
        inp = torch.stack([x, delta], dim=1)  # (B, 2)
        z = self.encoder(inp)  # (B, 16)
        
        # GRU step
        gru_in = torch.cat([z, prev_a], dim=1)  # (B, 17)
        # Need hidden state - for simplicity, use zero init
        # In practice would maintain hidden state
        return gru_in
    
    def predict_action(self, gru_out):
        """Linear controller on belief"""
        return torch.tanh(self.ctrl(gru_out))


def train_delta_model(n_ep=500, epochs=100):
    print("Generating data...")
    obs, delta, acts, bounces = generate_data(n_ep, 0.8, 42)
    
    # Convert to tensors
    obs_t = torch.tensor(obs, dtype=torch.float32).to(device)  # (N, 1)
    delta_t = torch.tensor(delta, dtype=torch.float32).to(device)  # (N, 1)
    acts_t = torch.tensor(acts, dtype=torch.float32).to(device)  # (N, 1)
    bounces_t = torch.tensor(bounces, dtype=torch.float32).to(device)  # (N,)
    
    print(f"  Data: {len(obs)} samples")
    print(f"  Delta stats: mean={delta.mean():.3f}, std={delta.std():.3f}")
    
    model = DeltaModel().to(device)
    opt = torch.optim.Adam(model.parameters(), lr=0.01)
    
    print(f"Training E1: Delta channel ({epochs} epochs)...")
    
    # Process per episode (maintain hidden state)
    for epoch in range(epochs):
        total_loss = 0
        
        for ep in range(n_ep):
            ep_start = ep * 30
            h = torch.zeros(24, device=device)  # init hidden
            
            for t in range(30):
                idx = ep_start + t
                
                x = obs_t[idx, 0]
                d = delta_t[idx, 0]
                a_prev = acts_t[idx-1, 0] if t > 0 else torch.zeros(1, device=device)
                target = acts_t[idx, 0]
                
                # Forward
                gru_in = model(x, d, a_prev)
                h = torch.tanh(gru_in @ torch.randn(17, 24, device=device) + torch.zeros(24, device=device))
                
                # Use a simpler forward for training
                inp = torch.stack([x, d]).unsqueeze(0)
                z = model.encoder(inp)
                gru_in = torch.cat([z.squeeze(0), a_prev], dim=0)
                
                # Simple MLP controller
                a_pred = torch.tanh(model.ctrl(gru_in.unsqueeze(0)).squeeze())
                
                loss = ((a_pred - target) ** 2).mean()
                
                opt.zero_grad()
                loss.backward()
                opt.step()
                
                total_loss += loss.item()
        
        if (epoch + 1) % 20 == 0:
            print(f"  Epoch {epoch+1}: loss={total_loss/n_ep/30:.4f}")
    
    return model


# Simpler version - just MLP with delta
class SimpleDeltaModel(nn.Module):
    """MLP with explicit delta: [x, delta] -> action"""
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
        """x, delta: (B,) tensors"""
        # Stack to (B, 2)
        if x.dim() == 0:
            x = x.unsqueeze(0)
        if delta.dim() == 0:
            delta = delta.unsqueeze(0)
        inp = torch.stack([x, delta], dim=-1)
        return torch.tanh(self.net(inp))


def train_simple_delta(n_ep=500, epochs=100):
    print("Generating data...")
    obs, delta, acts, bounces = generate_data(n_ep, 0.8, 42)
    
    obs_t = torch.tensor(obs.squeeze(), dtype=torch.float32).to(device)
    delta_t = torch.tensor(delta.squeeze(), dtype=torch.float32).to(device)
    acts_t = torch.tensor(acts.squeeze(), dtype=torch.float32).to(device)
    
    model = SimpleDeltaModel().to(device)
    opt = torch.optim.Adam(model.parameters(), lr=0.01)
    
    print(f"Training Simple Delta Model ({epochs} epochs)...")
    for epoch in range(epochs):
        # Shuffle
        idx = torch.randperm(len(obs_t))
        
        total_loss = 0
        for i in idx:
            x = obs_t[i]
            d = delta_t[i]
            a = acts_t[i]
            
            pred = model(x, d)
            loss = (pred - a) ** 2
            
            opt.zero_grad()
            loss.backward()
            opt.step()
            total_loss += loss.item()
        
        if (epoch + 1) % 20 == 0:
            print(f"  Epoch {epoch+1}: loss={total_loss/len(obs_t):.4f}")
    
    return model


def evaluate(model, n_test=200, use_model=True):
    ball = Ball(0.8)
    successes = 0
    
    model.eval()
    with torch.no_grad():
        for ep in range(n_test):
            ball.reset(seed=ep + 1000)
            x_prev = ball.x
            
            for step in range(30):
                x = ball.x
                d = x - x_prev
                
                # Get action
                if use_model:
                    x_t = torch.tensor([x], device=device, dtype=torch.float32)
                    d_t = torch.tensor([d], device=device, dtype=torch.float32)
                    a = model(x_t, d_t).item()
                else:
                    # Fallback: finite diff
                    v_est = d / 0.05 if step > 0 else 0
                    a = 1.5*(2-x) + (-2)*(-v_est)
                
                a = np.clip(a, -2, 2)
                ball.step(a)
                
                x_prev = x
                
                if abs(ball.x - 2.0) < 0.3:
                    successes += 1
                    break
    
    return successes / n_test


print("="*70)
print("E1: Delta Channel Experiment")
print("="*70)

print("\n1. Training Simple Delta Model...")
model = train_simple_delta(n_ep=500, epochs=100)

print("\n2. Evaluating...")
rate = evaluate(model, n_test=200, use_model=True)

print("\n" + "="*70)
print("RESULTS")
print("="*70)
print(f"PD full state:     71.0%")
print(f"PD partial:        56.5%")
print(f"E1 Delta Model:    {rate:.1%}")

gap = (rate - 0.565) / (0.71 - 0.565)
print(f"\nGap closed: {gap:.1%}")

if rate > 0.60:
    print("\n✓ E1 works! Delta channel helps.")
else:
    print("\n✗ E1 didn't close gap significantly.")
