#!/usr/bin/env python3
"""
Belief Model for Velocity Estimation Under Dropout

Simpler approach:
1. Encode (x, x_prev) → z
2. Decode z → v
3. Predict z_next from (z, a) for dropout recovery
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

device = torch.device('cuda')

# =============================================================================
# ENVIRONMENT
# =============================================================================

class Ball:
    def __init__(self, dt=0.05, restitution=0.8):
        self.dt = dt
        self.restitution = restitution
        self.g = 9.81
        self.x = 0
        self.v = 0
        self.impact = False
    
    def reset(self, x0):
        self.x = x0
        self.v = 0
        self.impact = False
        return self.x, self.v
    
    def step(self, a):
        a = np.clip(a, -2, 2)
        self.v += (-self.g + a) * self.dt
        self.x += self.v * self.dt
        self.impact = False
        if self.x < 0:
            self.x = -self.x * self.restitution
            self.v = -self.v * self.restitution
            self.impact = True
        if self.x > 3:
            self.x = 3 - (self.x - 3) * self.restitution
            self.v = -self.v * self.restitution
            self.impact = True
        return self.x, self.v

# =============================================================================
# BELIEF MODEL
# =============================================================================

class BeliefModel(nn.Module):
    """Encoder-Decoder with dynamics prediction"""
    
    def __init__(self, latent_dim=32):
        super().__init__()
        
        # Encoder: (x, x_prev) → z
        self.encoder = nn.Sequential(
            nn.Linear(2, 64), nn.ReLU(),
            nn.Linear(64, 64), nn.ReLU(),
            nn.Linear(64, latent_dim)
        )
        
        # Dynamics: (z, a) → z_next
        self.dynamics = nn.Sequential(
            nn.Linear(latent_dim + 1, 64), nn.ReLU(),
            nn.Linear(64, 64), nn.ReLU(),
            nn.Linear(64, latent_dim)
        )
        
        # Decoder: z → v
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, 32), nn.ReLU(),
            nn.Linear(32, 1)
        )
    
    def encode(self, x, x_prev):
        """Encode observation to latent"""
        inp = torch.cat([x, x_prev], dim=-1)
        return self.encoder(inp)
    
    def predict(self, z, a):
        """Predict next latent state"""
        inp = torch.cat([z, a], dim=-1)
        return self.dynamics(inp)
    
    def decode(self, z):
        """Decode latent to velocity"""
        return self.decoder(z)

# =============================================================================
# TRAINING WITH TRAJECTORY PREDICTION
# =============================================================================

def generate_trajectories(n_episodes=500, dt=0.05, horizon=5):
    """Generate trajectories for multi-step prediction training"""
    trajectories = []
    init_positions = [0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5]
    
    for ep in range(n_episodes):
        env = Ball(dt=dt)
        x0 = init_positions[ep % len(init_positions)]
        env.reset(x0)
        xp = x0
        
        traj = {'x': [], 'xp': [], 'v': [], 'a': []}
        
        for t in range(30):
            a = np.clip(1.5 * (2 - env.x) + (-2) * (-env.v), -2, 2)
            
            traj['x'].append(env.x)
            traj['xp'].append(xp)
            traj['v'].append(env.v)
            traj['a'].append(a)
            
            xp = env.x
            env.step(a)
        
        trajectories.append(traj)
    
    return trajectories


def train_belief_model(n_epochs=100, latent_dim=32, lr=1e-3, pred_horizon=3):
    print("Generating trajectories...")
    trajectories = generate_trajectories(n_episodes=500)
    
    model = BeliefModel(latent_dim=latent_dim).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    
    print(f"Training for {n_epochs} epochs...")
    
    for epoch in range(n_epochs):
        total_loss = 0
        v_loss = 0
        pred_loss = 0
        
        for traj in trajectories:
            # Convert to tensors
            x = torch.tensor(traj['x'], dtype=torch.float32, device=device).unsqueeze(-1)
            xp = torch.tensor(traj['xp'], dtype=torch.float32, device=device).unsqueeze(-1)
            v = torch.tensor(traj['v'], dtype=torch.float32, device=device).unsqueeze(-1)
            a = torch.tensor(traj['a'], dtype=torch.float32, device=device).unsqueeze(-1)
            
            # Encode all states
            z = model.encode(x, xp)
            
            # Decode velocity
            v_pred = model.decode(z)
            loss_v = F.mse_loss(v_pred, v)
            
            # Multi-step prediction
            loss_pred = 0
            z_cur = z[0]
            for t in range(min(pred_horizon, len(z) - 1)):
                z_next_pred = model.predict(z_cur, a[t])
                z_next_true = z[t + 1]
                loss_pred += F.mse_loss(z_next_pred, z_next_true.detach())
                z_cur = z_next_pred
            
            loss = loss_v + 0.1 * loss_pred
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            v_loss += loss_v.item()
            pred_loss += loss_pred.item()
        
        if (epoch + 1) % 20 == 0:
            print(f"Epoch {epoch+1}: total={total_loss:.2f}, v_loss={v_loss:.4f}, pred_loss={pred_loss:.4f}")
    
    return model


# =============================================================================
# EVALUATION
# =============================================================================

class BeliefController:
    def __init__(self, model, dt=0.05):
        self.model = model
        self.dt = dt
        self.z = None
        self.last_x = None
        self.last_a = None
    
    def reset(self, x0):
        self.z = None
        self.last_x = x0
        self.last_a = 0.0
    
    def get_velocity(self, x, observation_available=True):
        if observation_available:
            # Encode fresh observation
            x_t = torch.tensor([[x]], dtype=torch.float32, device=device)
            xp_t = torch.tensor([[self.last_x]], dtype=torch.float32, device=device)
            
            with torch.no_grad():
                self.z = self.model.encode(x_t, xp_t)
                v = self.model.decode(self.z).item()
            
            self.last_x = x
        else:
            # Predict using dynamics model
            if self.z is not None and self.last_a is not None:
                a_t = torch.tensor([[self.last_a]], dtype=torch.float32, device=device)
                with torch.no_grad():
                    self.z = self.model.predict(self.z, a_t)
                    v = self.model.decode(self.z).item()
            else:
                v = 0.0
        
        return v


def evaluate_belief(model, n_trials=100, dropout_steps=0):
    init_positions = [1.5, 2.0, 2.5, 3.0, 3.5]
    success = 0
    
    for x0 in init_positions:
        for _ in range(n_trials):
            env = Ball()
            env.reset(x0)
            controller = BeliefController(model)
            controller.reset(x0)
            
            freeze_countdown = 0
            
            for t in range(30):
                if env.impact:
                    freeze_countdown = dropout_steps
                
                obs_available = freeze_countdown <= 0
                if freeze_countdown > 0:
                    freeze_countdown -= 1
                
                v_est = controller.get_velocity(env.x, obs_available)
                a = np.clip(1.5 * (2 - env.x) + (-2) * (-v_est), -2, 2)
                
                controller.last_a = a
                env.step(a)
                
                if abs(env.x - 2.0) < 0.3:
                    success += 1
                    break
    
    return success / (len(init_positions) * n_trials)


def evaluate_fd(n_trials=100, dropout_steps=0):
    init_positions = [1.5, 2.0, 2.5, 3.0, 3.5]
    success = 0
    
    for x0 in init_positions:
        for _ in range(n_trials):
            env = Ball()
            env.reset(x0)
            xp = x0
            freeze_countdown = 0
            v_est = 0.0
            
            for t in range(30):
                if env.impact:
                    freeze_countdown = dropout_steps
                
                if freeze_countdown > 0:
                    freeze_countdown -= 1
                else:
                    v_est = (env.x - xp) / 0.05
                    xp = env.x
                
                a = np.clip(1.5 * (2 - env.x) + (-2) * (-v_est), -2, 2)
                env.step(a)
                
                if abs(env.x - 2.0) < 0.3:
                    success += 1
                    break
    
    return success / (len(init_positions) * n_trials)


# =============================================================================
# MAIN
# =============================================================================

if __name__ == '__main__':
    print("="*70)
    print("Belief Model for Velocity Estimation Under Dropout")
    print("="*70)
    
    model = train_belief_model(n_epochs=100, latent_dim=32, lr=1e-3, pred_horizon=3)
    
    print("\n" + "="*70)
    print("Evaluation: Post-Impact Dropout")
    print("="*70)
    print(f"{'Dropout':>8} | {'Belief':>8} | {'FD':>8}")
    print("-"*30)
    
    for dropout in [0, 1, 2, 3, 5]:
        belief_rate = evaluate_belief(model, dropout_steps=dropout)
        fd_rate = evaluate_fd(dropout_steps=dropout)
        print(f"{dropout:>8} | {belief_rate:>8.1%} | {fd_rate:>8.1%}")