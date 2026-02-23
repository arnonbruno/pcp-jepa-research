#!/usr/bin/env python3
"""
PHASE 5: L3 - Impact-Consistency with PyTorch + GPU

Goal: Make belief transitions around impacts physics-consistent.
This should make the belief control-sufficient.
"""

import torch
import torch.nn as nn
import numpy as np

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
        
        bounced = False
        if self.x < self.x_bounds[0]:
            self.x = -self.x * self.e
            self.v = -self.v * self.e
            bounced = True
        elif self.x > self.x_bounds[1]:
            self.x = self.x_bounds[1] - (self.x - self.x_bounds[1]) * self.e
            self.v = -self.v * self.e
            bounced = True
            
        return np.array([float(self.x), float(self.v)]), bounced


def generate_data(n_episodes, horizon, restitution, seed, obs_noise=0.05, dropout=0.1):
    np.random.seed(seed)
    env = BouncingBall(restitution=restitution)
    
    trajectories = []
    for ep in range(n_episodes):
        state = env.reset(seed=ep + seed)
        x, v = float(state[0]), float(state[1])
        
        traj = {'obs': [], 'acts': [], 'bounces': []}
        
        for step in range(horizon):
            obs = np.array([x])
            if obs_noise > 0:
                obs = obs + np.random.randn() * obs_noise
            if dropout > 0 and np.random.rand() < dropout:
                obs = obs * 0
            
            # Expert action
            a_expert = 1.5 * (2.0 - x) + (-2.0) * (-v)
            a_expert = float(np.clip(a_expert, -2.0, 2.0))
            
            # Step
            next_state, bounced = env.step(a_expert)
            x_next, v_next = float(next_state[0]), float(next_state[1])
            
            traj['obs'].append(obs)
            traj['acts'].append(a_expert)
            traj['bounces'].append(1.0 if bounced else 0.0)
            
            x, v = x_next, v_next
        
        trajectories.append(traj)
    
    return trajectories


# ============================================================================
# MODEL with IMPACT MODEL
# ============================================================================

class JEPAImpactModel(nn.Module):
    """
    JEPA with Impact-Consistency:
    - Encoder: obs -> latent
    - GRU: belief update
    - Impact Model: predicts belief change at impact
    - Controller: linear feedback
    """
    def __init__(self, obs_dim=1, action_dim=1, latent_dim=16, belief_dim=32):
        super().__init__()
        
        # Encoder
        self.encoder = nn.Sequential(
            nn.Linear(obs_dim, latent_dim),
            nn.Tanh()
        )
        
        # GRU
        self.gru = nn.GRUCell(latent_dim + action_dim, belief_dim)
        
        # Impact model: predicts belief change at impact
        self.impact_model = nn.Sequential(
            nn.Linear(belief_dim + 1, 16),  # belief + action
            nn.Tanh(),
            nn.Linear(16, belief_dim)
        )
        
        # Controller
        self.controller = nn.Linear(belief_dim, action_dim)
    
    def encode(self, obs):
        return self.encoder(obs)
    
    def gru_step(self, belief, z, a):
        x = torch.cat([z, a])
        return self.gru(x, belief)
    
    def impact_transition(self, belief, a):
        """Predict belief change at impact."""
        return self.impact_model(torch.cat([belief, a]))
    
    def forward(self, obs_seq, act_seq, bounce_seq=None):
        """Forward pass with impact modeling."""
        T = len(obs_seq)
        
        obs_tensor = obs_seq
        act_tensor = act_seq
        
        beliefs = []
        belief = torch.zeros(self.gru.hidden_size, device=device)
        
        for t in range(T):
            z = self.encode(obs_tensor[t])
            a_prev = act_tensor[t-1] if t > 0 else torch.zeros(1, device=device)
            
            # Normal GRU step
            belief_normal = self.gru_step(belief, z, a_prev)
            
            # If impact, apply impact model
            if bounce_seq is not None and t > 0:
                if bounce_seq[t] == 1.0:
                    # Impact: modify belief
                    delta = self.impact_transition(belief, act_tensor[t])
                    belief = belief + delta
                else:
                    belief = belief_normal
            else:
                belief = belief_normal
            
            beliefs.append(belief)
        
        return torch.stack(beliefs)
    
    def act(self, belief):
        return torch.tanh(self.controller(belief))


def train_l3(trajectories, n_epochs=100, lr=0.001):
    """Train with L1 (prediction) + L2 (event) + L3 (impact consistency)."""
    
    model = JEPAImpactModel(obs_dim=1, action_dim=1, latent_dim=16, belief_dim=32).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    
    print(f"  Model parameters: {sum(p.numel() for p in model.parameters())}")
    
    # Prepare data
    all_obs = [torch.tensor(np.array(t['obs']), dtype=torch.float32).to(device) for t in trajectories]
    all_acts = [torch.tensor(np.array(t['acts']), dtype=torch.float32).unsqueeze(-1).to(device) for t in trajectories]
    all_bounces = [torch.tensor(t['bounces'], dtype=torch.float32).to(device) for t in trajectories]
    
    for epoch in range(n_epochs):
        total_loss = 0.0
        total_impact_loss = 0.0
        
        for obs, acts, bounces in zip(all_obs, all_acts, all_bounces):
            # Forward pass
            beliefs = model(obs, acts, bounces)
            
            # L1: Prediction loss (predict next belief)
            pred_loss = 0.0
            for t in range(len(beliefs) - 1):
                pred_loss += torch.mean((beliefs[t+1] - beliefs[t]) ** 2)
            
            # L3: Impact consistency
            # At impacts, the belief should change predictably
            impact_loss = 0.0
            for t in range(1, len(beliefs)):
                if bounces[t] == 1.0:  # Impact occurred
                    # The impact model should predict the change
                    delta_pred = model.impact_transition(beliefs[t-1], acts[t-1])
                    delta_true = beliefs[t] - beliefs[t-1]
                    impact_loss += torch.mean((delta_pred - delta_true) ** 2)
            
            # L4: Controller imitation
            ctrl_loss = 0.0
            for t in range(len(beliefs)):
                a_pred = model.act(beliefs[t])
                ctrl_loss += torch.mean((a_pred - acts[t]) ** 2)
            
            # Total loss: balance all objectives
            loss = 0.3 * pred_loss + 1.0 * impact_loss + 0.5 * ctrl_loss
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            total_impact_loss += impact_loss.item()
        
        if (epoch + 1) % 20 == 0:
            print(f"    Epoch {epoch+1}: loss={total_loss/len(trajectories):.4f}, impact={total_impact_loss/len(trajectories):.4f}")
    
    return model


def evaluate(model, n_episodes, restitution, obs_noise, dropout, seed):
    model.eval()
    env = BouncingBall(restitution=restitution)
    
    successes = 0
    
    with torch.no_grad():
        for ep in range(n_episodes):
            state = env.reset(seed=ep + seed + 1000)
            x, v = float(state[0]), float(state[1])
            
            belief = torch.zeros(32, device=device)
            last_a = 0.0
            
            for step in range(30):
                obs = np.array([x])
                if obs_noise > 0 and np.random.rand() < obs_noise:
                    obs = obs + np.random.randn() * obs_noise
                if dropout > 0 and np.random.rand() < dropout:
                    obs = obs * 0
                
                obs_t = torch.tensor([obs], dtype=torch.float32).to(device)
                a_t = torch.tensor([[last_a]], dtype=torch.float32).to(device)
                
                z = model.encode(obs_t)
                belief = model.gru_step(belief, z, a_t.squeeze())
                
                a = model.act(belief).item()
                a = float(np.clip(a, -2.0, 2.0))
                last_a = a
                
                state, _ = env.step(a)
                x, v = float(state[0]), float(state[1])
                
                if abs(x - env.x_target) < env.tau:
                    successes += 1
                    break
    
    return successes / n_episodes


def main():
    print("="*70)
    print("PHASE 5: L3 - Impact-Consistency (PyTorch + GPU)")
    print("="*70)
    
    restitution = 0.8
    obs_noise = 0.05
    dropout = 0.1
    
    print("\n1. Generating data...")
    train_trajs = generate_data(500, 30, restitution, seed=42, obs_noise=obs_noise, dropout=dropout)
    print(f"   Generated {len(train_trajs)} trajectories")
    
    # Count bounces
    n_bounces = sum(sum(t['bounces']) for t in train_trajs)
    print(f"   Total bounces: {n_bounces}")
    
    # Train L3
    print("\n2. Training with L3 (impact consistency)...")
    model = train_l3(train_trajs, n_epochs=100, lr=0.001)
    
    # Evaluate
    print("\n3. Evaluating...")
    rate = evaluate(model, 200, restitution, obs_noise, dropout, seed=42)
    print(f"   Success rate: {rate:.1%}")
    
    print("\n" + "="*70)
    print("COMPARISON")
    print("="*70)
    print(f"PD (full state):   71.0%")
    print(f"PD (partial):      56.5%")
    print(f"L3 (impact):      {rate:.1%}")
    
    gap_closed = (rate - 0.565) / (0.71 - 0.565)
    print(f"\nGap closed: {gap_closed:.1%}")
    
    if rate > 0.60:
        print("\n✓ L3 works! Impact-consistency makes belief control-sufficient.")
    else:
        print("\n✗ L3 didn't close gap. Need different approach.")


if __name__ == '__main__':
    main()
