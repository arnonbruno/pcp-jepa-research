#!/usr/bin/env python3
"""
PHASE 5: L4 with PyTorch + GPU
"""

import torch
import torch.nn as nn
import numpy as np

# Use GPU
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")


# ============================================================================
# ENVIRONMENT
# ============================================================================

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
        start_positions = [0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5]
        self.x = start_positions[seed % len(start_positions)]
        self.v = 0.0
        return np.array([self.x, self.v])
    
    def step(self, a):
        a = np.clip(a, -2.0, 2.0)
        self.v += (-self.g + a) * self.dt
        self.x += self.v * self.dt
        
        if self.x < self.x_bounds[0]:
            self.x = -self.x * self.e
            self.v = -self.v * self.e
        elif self.x > self.x_bounds[1]:
            self.x = self.x_bounds[1] - (self.x - self.x_bounds[1]) * self.e
            self.v = -self.v * self.e
            
        return np.array([float(self.x), float(self.v)])


def generate_data(n_episodes, horizon, restitution, seed, obs_noise=0.05, dropout=0.1):
    np.random.seed(seed)
    env = BouncingBall(restitution=restitution)
    
    trajectories = []
    for ep in range(n_episodes):
        state = env.reset(seed=ep + seed)
        x, v = float(state[0]), float(state[1])
        
        traj = {'obs': [], 'acts': []}
        
        for step in range(horizon):
            obs = np.array([x])
            if obs_noise > 0:
                obs = obs + np.random.randn() * obs_noise
            if dropout > 0 and np.random.rand() < dropout:
                obs = obs * 0
            
            a_expert = float(np.clip(1.5 * (2.0 - x) + (-2.0) * (-v), -2.0, 2.0))
            
            next_state = env.step(a_expert)
            x_next, v_next = float(next_state[0]), float(next_state[1])
            
            traj['obs'].append(obs)
            traj['acts'].append(a_expert)
            
            x, v = x_next, v_next
        
        trajectories.append(traj)
    
    return trajectories


# ============================================================================
# MODEL (PyTorch)
# ============================================================================

class JEPAModel(nn.Module):
    def __init__(self, obs_dim=1, action_dim=1, latent_dim=8, belief_dim=16):
        super().__init__()
        
        # Encoder
        self.encoder = nn.Sequential(
            nn.Linear(obs_dim, latent_dim),
            nn.Tanh()
        )
        
        # GRU cell
        self.gru = nn.GRUCell(latent_dim + action_dim, belief_dim)
        
        # Controller
        self.controller = nn.Linear(belief_dim, action_dim)
    
    def forward(self, obs_seq, act_seq):
        """Forward pass through sequence."""
        T = len(obs_seq)
        
        # obs_seq and act_seq are already tensors on GPU
        obs_tensor = obs_seq
        act_tensor = act_seq
        
        beliefs = []
        belief = torch.zeros(self.gru.hidden_size, device=device)
        
        for t in range(T):
            z = self.encoder(obs_tensor[t])
            a_prev = act_tensor[t-1] if t > 0 else torch.zeros(self.controller.out_features, device=device)
            
            x = torch.cat([z, a_prev])
            belief = self.gru(x, belief)
            beliefs.append(belief)
        
        return torch.stack(beliefs)
    
    def act(self, belief):
        return torch.tanh(self.controller(belief))


def train_model(model, trajectories, n_epochs=100, lr=0.01, freeze_encoder=False):
    """Train model."""
    model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.MSELoss()
    
    # Convert trajectories to tensors ONCE
    all_obs = [torch.tensor(np.array(t['obs']), dtype=torch.float32).to(device) for t in trajectories]
    all_acts = [torch.tensor(np.array(t['acts']), dtype=torch.float32).unsqueeze(-1).to(device) for t in trajectories]
    
    for epoch in range(n_epochs):
        total_loss = 0.0
        
        for obs, acts in zip(all_obs, all_acts):
            obs = obs.to(device)
            acts = acts.to(device)
            
            beliefs = model(obs, acts)
            
            # Imitation loss
            preds = model.controller(beliefs)
            preds = torch.tanh(preds)  # clip via tanh
            loss = criterion(preds, acts)
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
        
        if (epoch + 1) % 20 == 0:
            print(f"  Epoch {epoch+1}: loss={total_loss/len(trajectories):.4f}")
    
    return model


def evaluate(model, n_episodes, restitution, obs_noise, dropout, seed):
    model.eval()
    env = BouncingBall(restitution=restitution)
    
    successes = 0
    
    with torch.no_grad():
        for ep in range(n_episodes):
            state = env.reset(seed=ep + seed + 1000)
            x, v = float(state[0]), float(state[1])
            
            belief = torch.zeros(16).to(device)
            last_a = 0.0
            
            for step in range(30):
                obs = np.array([x])
                if obs_noise > 0 and np.random.rand() < obs_noise:
                    obs = obs + np.random.randn() * obs_noise
                if dropout > 0 and np.random.rand() < dropout:
                    obs = obs * 0
                
                obs_t = torch.tensor([obs], dtype=torch.float32).to(device)
                z = model.encoder(obs_t)
                a_prev = torch.tensor([[last_a]]).to(device)
                x_cat = torch.cat([z, a_prev.squeeze()])
                belief = model.gru(x_cat, belief)
                
                a = model.controller(belief)
                a = torch.tanh(a).item()
                a = float(np.clip(a, -2.0, 2.0))
                last_a = a
                
                state = env.step(a)
                x, v = float(state[0]), float(state[1])
                
                if abs(x - env.x_target) < env.tau:
                    successes += 1
                    break
    
    return successes / n_episodes


def main():
    print("="*70)
    print("PHASE 5: L4 with PyTorch + GPU")
    print("="*70)
    
    restitution = 0.8
    obs_noise = 0.05
    dropout = 0.1
    
    print("\n1. Generating data...")
    train_trajs = generate_data(500, 30, restitution, seed=42, obs_noise=obs_noise, dropout=dropout)
    print(f"   Generated {len(train_trajs)} trajectories")
    
    # Model
    model = JEPAModel(obs_dim=1, action_dim=1, latent_dim=8, belief_dim=16)
    print(f"\n2. Model: {sum(p.numel() for p in model.parameters())} parameters")
    
    # Train
    print("\n3. Training...")
    model = train_model(model, train_trajs, n_epochs=100, lr=0.01)
    
    # Evaluate
    print("\n4. Evaluating...")
    rate = evaluate(model, 200, restitution, obs_noise, dropout, seed=42)
    print(f"   Success rate: {rate:.1%}")
    
    print("\n" + "="*70)
    print("COMPARISON")
    print("="*70)
    print(f"PD (full state):   71.0%")
    print(f"PD (partial):      56.5%")
    print(f"JEPA + ctrl:      {rate:.1%}")
    
    gap_closed = (rate - 0.565) / (0.71 - 0.565)
    print(f"\nGap closed: {gap_closed:.1%}")


if __name__ == '__main__':
    main()
