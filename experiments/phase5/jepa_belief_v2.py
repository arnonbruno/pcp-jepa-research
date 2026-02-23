#!/usr/bin/env python3
"""
JEPA with Multi-Step Prediction for Velocity Estimation

Key idea: When observation is missing, use JEPA predictor to estimate
the latent state, then decode velocity from the latent.

Training:
1. Learn embedding z = f_c(o, Δo/dt)
2. Learn predictor z_{t+1} = g(z_t, a_t)
3. Learn decoder v = d(z) to recover velocity

Evaluation:
- When observation available: z = f_c(o, Δo/dt)
- When observation missing: z_{t+1} = g(z_t, a_t)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from collections import deque

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
# JEPA ARCHITECTURE WITH DECODER
# =============================================================================

class Encoder(nn.Module):
    def __init__(self, latent_dim=32):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(2, 64), nn.ReLU(),
            nn.Linear(64, 64), nn.ReLU(),
            nn.Linear(64, latent_dim)
        )
    
    def forward(self, o, delta_o):
        return self.net(torch.cat([o, delta_o], dim=-1))


class TargetEncoder(nn.Module):
    def __init__(self, latent_dim=32):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(2, 64), nn.ReLU(),
            nn.Linear(64, 64), nn.ReLU(),
            nn.Linear(64, latent_dim)
        )
    
    def forward(self, o, delta_o):
        return self.net(torch.cat([o, delta_o], dim=-1))
    
    @torch.no_grad()
    def update_ema(self, encoder, tau=0.996):
        for tp, ep in zip(self.parameters(), encoder.parameters()):
            tp.data.mul_(tau).add_(ep.data, alpha=1 - tau)


class Predictor(nn.Module):
    def __init__(self, latent_dim=32):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(latent_dim + 1, 64), nn.ReLU(),
            nn.Linear(64, 64), nn.ReLU(),
            nn.Linear(64, latent_dim)
        )
    
    def forward(self, z, a):
        return self.net(torch.cat([z, a], dim=-1))


class VelocityDecoder(nn.Module):
    """Decode velocity from latent state"""
    def __init__(self, latent_dim=32):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(latent_dim, 32), nn.ReLU(),
            nn.Linear(32, 1)
        )
    
    def forward(self, z):
        return self.net(z)


class JEPABeliefModel(nn.Module):
    def __init__(self, latent_dim=32):
        super().__init__()
        self.encoder = Encoder(latent_dim)
        self.target_encoder = TargetEncoder(latent_dim)
        self.predictor = Predictor(latent_dim)
        self.velocity_decoder = VelocityDecoder(latent_dim)
        
        self.target_encoder.load_state_dict(self.encoder.state_dict())
    
    def encode(self, o, delta_o):
        return self.encoder(o, delta_o)
    
    def predict(self, z, a):
        return self.predictor(z, a)
    
    def decode_velocity(self, z):
        return self.velocity_decoder(z)
    
    def forward(self, o_t, delta_o_t, a_t, o_tp1, delta_o_tp1, v_tp1):
        # Context
        z_t = self.encoder(o_t, delta_o_t)
        
        # Target
        with torch.no_grad():
            z_target = self.target_encoder(o_tp1, delta_o_tp1)
        
        # Prediction
        z_pred = self.predictor(z_t, a_t)
        
        # Velocity decode
        v_pred = self.velocity_decoder(z_t)
        v_pred_next = self.velocity_decoder(z_pred)
        
        return z_t, z_pred, z_target, v_pred, v_pred_next, v_tp1
    
    def update_target(self, tau=0.996):
        self.target_encoder.update_ema(self.encoder, tau)


# =============================================================================
# TRAINING
# =============================================================================

def generate_data(n_episodes=500, dt=0.05):
    data = {'o': [], 'delta_o': [], 'a': [], 'o_next': [], 'delta_o_next': [], 'v': [], 'v_next': []}
    
    init_positions = [0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5]
    
    for ep in range(n_episodes):
        env = Ball(dt=dt)
        x0 = init_positions[ep % len(init_positions)]
        env.reset(x0)
        xp = x0
        
        for t in range(30):
            a = np.clip(1.5 * (2 - env.x) + (-2) * (-env.v), -2, 2)
            
            o = env.x
            delta_o = (env.x - xp) / dt
            v = env.v
            
            env.step(a)
            
            o_next = env.x
            delta_o_next = (env.x - o) / dt
            v_next = env.v
            
            data['o'].append(o)
            data['delta_o'].append(delta_o)
            data['a'].append(a)
            data['o_next'].append(o_next)
            data['delta_o_next'].append(delta_o_next)
            data['v'].append(v)
            data['v_next'].append(v_next)
            
            xp = env.x
    
    for k in data:
        data[k] = torch.tensor(np.array(data[k]), dtype=torch.float32, device=device)
    
    return data


def train(n_epochs=100, latent_dim=32, lr=1e-3, tau=0.996, lambda_v=1.0, lambda_pred=1.0):
    print("Generating data...")
    data = generate_data(n_episodes=500)
    
    model = JEPABeliefModel(latent_dim=latent_dim).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    
    batch_size = 64
    n_samples = len(data['o'])
    
    print(f"Training for {n_epochs} epochs...")
    
    for epoch in range(n_epochs):
        idx = torch.randperm(n_samples)
        total_loss = 0
        
        for i in range(0, n_samples, batch_size):
            b = idx[i:i+batch_size]
            
            z_t, z_pred, z_target, v_pred, v_pred_next, v_true_next = model(
                data['o'][b].unsqueeze(-1),
                data['delta_o'][b].unsqueeze(-1),
                data['a'][b].unsqueeze(-1),
                data['o_next'][b].unsqueeze(-1),
                data['delta_o_next'][b].unsqueeze(-1),
                data['v_next'][b].unsqueeze(-1)
            )
            
            # Embedding prediction loss
            loss_pred = F.mse_loss(z_pred, z_target.detach())
            
            # Velocity prediction loss (current and next)
            loss_v = F.mse_loss(v_pred.squeeze(), data['v'][b]) + \
                     F.mse_loss(v_pred_next.squeeze(), v_true_next.squeeze())
            
            loss = lambda_pred * loss_pred + lambda_v * loss_v
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            model.update_target(tau)
            
            total_loss += loss.item()
        
        if (epoch + 1) % 20 == 0:
            print(f"Epoch {epoch+1}: loss={total_loss:.4f}")
    
    return model


# =============================================================================
# EVALUATION
# =============================================================================

class JEPABeliefController:
    def __init__(self, model, dt=0.05):
        self.model = model
        self.dt = dt
        self.z = None
        self.last_x = None
    
    def reset(self, x0):
        self.z = None
        self.last_x = x0
    
    def estimate_velocity(self, x, a_last, observation_available=True):
        """Estimate velocity, using prediction if observation unavailable"""
        
        if observation_available:
            # Fresh observation: encode
            delta_o = (x - self.last_x) / self.dt if self.last_x is not None else 0.0
            o_t = torch.tensor([[x]], dtype=torch.float32, device=device)
            d_t = torch.tensor([[delta_o]], dtype=torch.float32, device=device)
            
            with torch.no_grad():
                self.z = self.model.encode(o_t, d_t)
            
            self.last_x = x
        else:
            # No observation: predict using last action
            if self.z is not None:
                a_t = torch.tensor([[a_last]], dtype=torch.float32, device=device)
                with torch.no_grad():
                    self.z = self.model.predict(self.z, a_t)
        
        # Decode velocity
        if self.z is not None:
            with torch.no_grad():
                v_est = self.model.decode_velocity(self.z).item()
            return v_est
        else:
            return 0.0


def evaluate(model, n_trials=100, dropout_steps=0):
    init_positions = [1.5, 2.0, 2.5, 3.0, 3.5]
    success = 0
    
    for x0 in init_positions:
        for _ in range(n_trials):
            env = Ball()
            env.reset(x0)
            controller = JEPABeliefController(model)
            controller.reset(x0)
            
            freeze_countdown = 0
            v_est = 0.0
            a_last = 0.0
            
            for t in range(30):
                # Check for impact
                if env.impact:
                    freeze_countdown = dropout_steps
                
                obs_available = freeze_countdown <= 0
                if freeze_countdown > 0:
                    freeze_countdown -= 1
                
                # Estimate velocity
                v_est = controller.estimate_velocity(env.x, a_last, obs_available)
                
                # Control
                a = np.clip(1.5 * (2 - env.x) + (-2) * (-v_est), -2, 2)
                a_last = a
                env.step(a)
                
                if abs(env.x - 2.0) < 0.3:
                    success += 1
                    break
    
    return success / (len(init_positions) * n_trials)


def evaluate_fd(n_trials=100, dropout_steps=0):
    """FD baseline for comparison"""
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
    print("JEPA Belief Model with Multi-Step Prediction")
    print("="*70)
    
    model = train(n_epochs=100, latent_dim=32, lr=1e-3, tau=0.996, lambda_v=1.0, lambda_pred=1.0)
    
    print("\n" + "="*70)
    print("Evaluation: Post-Impact Dropout")
    print("="*70)
    
    for dropout in [0, 1, 2, 3]:
        jepa_rate = evaluate(model, dropout_steps=dropout)
        fd_rate = evaluate_fd(dropout_steps=dropout)
        print(f"Dropout {dropout} steps: JEPA={jepa_rate:.1%}, FD={fd_rate:.1%}")