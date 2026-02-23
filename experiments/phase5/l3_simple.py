#!/usr/bin/env python3
"""
PHASE 5: L3 - Simplified Impact Model (GPU)
"""

import torch
import torch.nn as nn
import numpy as np

device = torch.device('cuda')
print(f"Using device: {device}")


class BouncingBall:
    def __init__(self, restitution=0.8):
        self.g = 9.81
        self.e = restitution
        self.dt = 0.05
        self.x_target = 2.0
        self.tau = 0.3
        self.x_bounds = (0, 3)
        
    def reset(self, seed):
        np.random.seed(seed)
        self.x = [0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5][seed % 7]
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
        elif self.x > 3:
            self.x = 3 - (self.x - 3) * self.e
            self.v = -self.v * self.e
            bounced = True
        return np.array([float(self.x), float(self.v)]), bounced


def generate_data(n_episodes, restitution, seed):
    np.random.seed(seed)
    env = BouncingBall(restitution)
    trajs = []
    for ep in range(n_episodes):
        state = env.reset(seed=ep + seed)
        x, v = float(state[0]), float(state[1])
        traj = {'obs': [], 'acts': [], 'bounces': []}
        for _ in range(30):
            obs = np.array([x])
            a = 1.5 * (2.0 - x) + (-2.0) * (-v)
            a = float(np.clip(a, -2.0, 2.0))
            state, bounced = env.step(a)
            x, v = float(state[0]), float(state[1])
            traj['obs'].append(obs)
            traj['acts'].append(a)
            traj['bounces'].append(1.0 if bounced else 0.0)
        trajs.append(traj)
    return trajs


class SimpleModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.enc = nn.Linear(1, 8)
        self.gru = nn.GRUCell(9, 16)  # 8 + 1
        self.ctrl = nn.Linear(16, 1)
        self.impact = nn.Linear(17, 16)  # belief + action
    
    def forward(self, obs, acts, bounces):
        B = len(obs)
        beliefs = []
        b = torch.zeros(16, device=device)
        for t in range(B):
            z = torch.tanh(self.enc(obs[t]))
            a_prev = acts[t-1] if t > 0 else torch.zeros(1, device=device)
            b_new = self.gru(torch.cat([z, a_prev]), b)
            if bounces[t] == 1.0 and t > 0:
                delta = torch.tanh(self.impact(torch.cat([b, acts[t-1]])))
                b = b + delta
            else:
                b = b_new
            beliefs.append(b)
        return torch.stack(beliefs)


def train(model, trajs, epochs=50):
    opt = torch.optim.Adam(model.parameters(), lr=0.01)
    for e in range(epochs):
        loss_sum = 0
        for traj in trajs:
            obs = torch.tensor([np.array(o) for o in traj['obs']], dtype=torch.float32).to(device)
            acts = torch.tensor(traj['acts'], dtype=torch.float32).unsqueeze(-1).to(device)
            bounces = torch.tensor(traj['bounces'], dtype=torch.float32).to(device)
            
            beliefs = model(obs, acts, bounces)
            
            # Controller loss
            loss = 0
            for t in range(len(beliefs)):
                a_pred = torch.tanh(model.ctrl(beliefs[t]))
                loss += ((a_pred - acts[t]) ** 2).mean()
            
            # Impact loss
            for t in range(1, len(beliefs)):
                if bounces[t] == 1:
                    delta = torch.tanh(model.impact(torch.cat([beliefs[t-1], acts[t-1]])))
                    loss += 0.1 * ((beliefs[t] - beliefs[t-1] - delta) ** 2).mean()
            
            opt.zero_grad()
            loss.backward()
            opt.step()
            loss_sum += loss.item()
        if (e+1) % 10 == 0:
            print(f"  Epoch {e+1}: {loss_sum/len(trajs):.4f}")
    return model


def eval_model(model, restitution, seed):
    model.eval()
    env = BouncingBall(restitution)
    successes = 0
    with torch.no_grad():
        for ep in range(200):
            state = env.reset(seed=ep + seed + 1000)
            x, v = float(state[0]), float(state[1])
            b = torch.zeros(16, device=device)
            last_a = 0.0
            for _ in range(30):
                obs = torch.tensor([[x]], dtype=torch.float32).to(device)
                a_t = torch.tensor([[last_a]], dtype=torch.float32).to(device)
                z = torch.tanh(model.enc(obs))
                b = model.gru(torch.cat([z, a_t.squeeze()]), b)
                a = torch.tanh(model.ctrl(b)).item()
                a = float(np.clip(a, -2, 2))
                last_a = a
                state, _ = env.step(a)
                x, v = float(state[0]), float(state[1])
                if abs(x - 2.0) < 0.3:
                    successes += 1
                    break
    return successes / 200


print("="*70)
print("L3 - Impact Consistency (GPU)")
print("="*70)

print("\n1. Generating data...")
trajs = generate_data(500, 0.8, 42)
print(f"   {len(trajs)} trajectories")

print("\n2. Training...")
model = SimpleModel().to(device)
model = train(model, trajs, epochs=50)

print("\n3. Evaluating...")
rate = eval_model(model, 0.8, 42)
print(f"   Success: {rate:.1%}")

print("\n" + "="*70)
print(f"PD full:     71.0%")
print(f"PD partial:  56.5%")
print(f"L3 impact:  {rate:.1%}")
gap = (rate - 0.565) / (0.71 - 0.565)
print(f"Gap closed: {gap:.1%}")
