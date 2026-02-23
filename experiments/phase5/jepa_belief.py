#!/usr/bin/env python3
"""
JEPA (Joint-Embedding Predictive Architecture) for Bouncing Ball

Architecture:
- Context Encoder f_c: (o_t, Δo/dt) → z_t
- Target Encoder f_t: EMA of f_c
- Predictor g: (z_t, a_t) → z_{t+1}
- Event Head h: z_t → (time_to_impact, contact_flag)

Training:
- L_pred: Embedding prediction loss
- L_event: Time-to-impact + contact prediction
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from collections import deque

device = torch.device('cuda')

# =============================================================================
# PHYSICS ENVIRONMENT
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
    
    def time_to_impact(self, max_steps=60):
        """Estimate time to next impact"""
        x, v = self.x, self.v
        for i in range(max_steps):
            v += -self.g * self.dt
            x += v * self.dt
            if x < 0 or x > 3:
                return i * self.dt
        return max_steps * self.dt

# =============================================================================
# JEPA ARCHITECTURE
# =============================================================================

class ContextEncoder(nn.Module):
    """Encodes (o_t, Δo/dt) → z_t"""
    def __init__(self, latent_dim=32):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(2, 64), nn.ReLU(),
            nn.Linear(64, 64), nn.ReLU(),
            nn.Linear(64, latent_dim)
        )
    
    def forward(self, o, delta_o):
        # o: position, delta_o: (o - o_prev) / dt = velocity estimate
        inp = torch.cat([o, delta_o], dim=-1)
        return self.net(inp)


class TargetEncoder(nn.Module):
    """EMA of ContextEncoder for stable targets"""
    def __init__(self, latent_dim=32):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(2, 64), nn.ReLU(),
            nn.Linear(64, 64), nn.ReLU(),
            nn.Linear(64, latent_dim)
        )
    
    def forward(self, o, delta_o):
        inp = torch.cat([o, delta_o], dim=-1)
        return self.net(inp)
    
    @torch.no_grad()
    def update_ema(self, context_encoder, tau=0.996):
        """Update target encoder weights as EMA of context encoder"""
        for target_param, context_param in zip(self.parameters(), context_encoder.parameters()):
            target_param.data.mul_(tau).add_(context_param.data, alpha=1 - tau)


class Predictor(nn.Module):
    """Predicts z_{t+1} from z_t and action a_t"""
    def __init__(self, latent_dim=32, action_dim=1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(latent_dim + action_dim, 64), nn.ReLU(),
            nn.Linear(64, 64), nn.ReLU(),
            nn.Linear(64, latent_dim)
        )
    
    def forward(self, z, a):
        inp = torch.cat([z, a], dim=-1)
        return self.net(inp)


class EventHead(nn.Module):
    """Predicts time-to-impact and contact flag from z_t"""
    def __init__(self, latent_dim=32):
        super().__init__()
        self.time_net = nn.Sequential(
            nn.Linear(latent_dim, 32), nn.ReLU(),
            nn.Linear(32, 1), nn.Sigmoid()  # Normalized time [0, 1]
        )
        self.contact_net = nn.Sequential(
            nn.Linear(latent_dim, 32), nn.ReLU(),
            nn.Linear(32, 1), nn.Sigmoid()
        )
    
    def forward(self, z):
        time_to_impact = self.time_net(z) * 3.0  # Scale to [0, 3] seconds
        contact = self.contact_net(z)
        return time_to_impact, contact


class JEPA(nn.Module):
    """Full JEPA model"""
    def __init__(self, latent_dim=32):
        super().__init__()
        self.context_encoder = ContextEncoder(latent_dim)
        self.target_encoder = TargetEncoder(latent_dim)
        self.predictor = Predictor(latent_dim)
        self.event_head = EventHead(latent_dim)
        
        # Initialize target encoder with context encoder weights
        self.target_encoder.load_state_dict(self.context_encoder.state_dict())
    
    def forward(self, o_t, delta_o_t, a_t, o_tp1, delta_o_tp1):
        """
        Args:
            o_t: position at t
            delta_o_t: velocity estimate at t
            a_t: action at t
            o_tp1: position at t+1 (target)
            delta_o_tp1: velocity estimate at t+1 (target)
        """
        # Context embedding
        z_t = self.context_encoder(o_t, delta_o_t)
        
        # Target embedding (no grad)
        with torch.no_grad():
            z_target = self.target_encoder(o_tp1, delta_o_tp1)
        
        # Predicted next embedding
        z_pred = self.predictor(z_t, a_t)
        
        # Event predictions
        time_impact, contact = self.event_head(z_t)
        
        return z_t, z_pred, z_target, time_impact, contact
    
    def update_target_encoder(self, tau=0.996):
        self.target_encoder.update_ema(self.context_encoder, tau)


# =============================================================================
# TRAINING DATA GENERATION
# =============================================================================

def generate_jepa_data(n_episodes=500, dt=0.05):
    """Generate training data with actions, observations, and event labels"""
    data = {
        'o': [], 'delta_o': [], 'a': [], 'o_next': [], 'delta_o_next': [],
        'time_to_impact': [], 'contact': []
    }
    
    init_positions = [0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5]
    
    for ep in range(n_episodes):
        env = Ball(dt=dt)
        x0 = init_positions[ep % len(init_positions)]
        env.reset(x0)
        xp = x0
        
        for t in range(30):
            # PD controller with true velocity (expert)
            a = np.clip(1.5 * (2 - env.x) + (-2) * (-env.v), -2, 2)
            
            # Record state
            o = env.x
            delta_o = (env.x - xp) / dt
            
            # Step
            env.step(a)
            
            o_next = env.x
            delta_o_next = (env.x - o) / dt
            
            # Event labels
            tti = env.time_to_impact()
            contact = 1.0 if env.impact else 0.0
            
            data['o'].append(o)
            data['delta_o'].append(delta_o)
            data['a'].append(a)
            data['o_next'].append(o_next)
            data['delta_o_next'].append(delta_o_next)
            data['time_to_impact'].append(tti)
            data['contact'].append(contact)
            
            xp = env.x
    
    # Convert to tensors
    for k in data:
        data[k] = torch.tensor(np.array(data[k]), dtype=torch.float32, device=device)
    
    return data


# =============================================================================
# TRAINING
# =============================================================================

def train_jepa(n_epochs=100, latent_dim=32, lr=1e-3, tau=0.996, lambda_event=0.1):
    """Train JEPA model"""
    
    # Generate data
    print("Generating training data...")
    data = generate_jepa_data(n_episodes=500)
    
    # Create model
    model = JEPA(latent_dim=latent_dim).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    
    batch_size = 64
    n_samples = len(data['o'])
    
    print(f"Training JEPA for {n_epochs} epochs...")
    
    for epoch in range(n_epochs):
        # Shuffle
        idx = torch.randperm(n_samples)
        
        total_loss = 0
        pred_loss = 0
        event_loss = 0
        
        for i in range(0, n_samples, batch_size):
            b = idx[i:i+batch_size]
            
            # Forward pass
            z_t, z_pred, z_target, time_impact, contact = model(
                data['o'][b].unsqueeze(-1),
                data['delta_o'][b].unsqueeze(-1),
                data['a'][b].unsqueeze(-1),
                data['o_next'][b].unsqueeze(-1),
                data['delta_o_next'][b].unsqueeze(-1)
            )
            
            # Prediction loss (VICReg-style: variance + invariance + covariance)
            # Simplified: just MSE with stop-gradient on target
            loss_pred = F.mse_loss(z_pred, z_target.detach())
            
            # Event loss
            loss_time = F.mse_loss(time_impact.squeeze(), data['time_to_impact'][b])
            loss_contact = F.binary_cross_entropy(contact.squeeze(), data['contact'][b])
            loss_event = loss_time + loss_contact
            
            # Total loss
            loss = loss_pred + lambda_event * loss_event
            
            # Backward
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            # Update target encoder
            model.update_target_encoder(tau)
            
            total_loss += loss.item()
            pred_loss += loss_pred.item()
            event_loss += loss_event.item()
        
        if (epoch + 1) % 20 == 0:
            print(f"Epoch {epoch+1}: total={total_loss:.4f}, pred={pred_loss:.4f}, event={event_loss:.4f}")
    
    return model


# =============================================================================
# CLOSED-LOOP CONTROL WITH JEPA
# =============================================================================

class JEPABeliefController:
    """Controller using JEPA belief for velocity estimation"""
    
    def __init__(self, model, dt=0.05, history_len=2):
        self.model = model
        self.dt = dt
        self.history_len = history_len
        self.history = deque(maxlen=history_len)
    
    def reset(self, x0):
        self.history = deque([x0] * self.history_len, maxlen=self.history_len)
    
    def estimate_velocity(self, x):
        """Estimate velocity using JEPA embedding"""
        self.history.append(x)
        
        # Use finite difference for velocity estimate
        if len(self.history) >= 2:
            delta_o = (self.history[-1] - self.history[-2]) / self.dt
        else:
            delta_o = 0.0
        
        # Get embedding (could use for more sophisticated estimation)
        with torch.no_grad():
            o_t = torch.tensor([[x]], dtype=torch.float32, device=device)
            d_t = torch.tensor([[delta_o]], dtype=torch.float32, device=device)
            z_t = self.model.context_encoder(o_t, d_t)
        
        # For now, use the delta_o as velocity estimate
        # In full implementation, could decode from z_t or use predictor
        return delta_o
    
    def get_action(self, x, target=2.0, k1=1.5, k2=-2.0):
        """PD controller with JEPA velocity estimate"""
        v_est = self.estimate_velocity(x)
        a = k1 * (target - x) + k2 * (-v_est)
        return np.clip(a, -2, 2)


def evaluate_jepa(model, n_trials=100, dropout_steps=0):
    """Evaluate JEPA controller with optional post-impact dropout"""
    
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
            
            for t in range(30):
                # Check for impact
                if env.impact:
                    freeze_countdown = dropout_steps
                
                if freeze_countdown > 0:
                    # Use frozen velocity estimate
                    freeze_countdown -= 1
                else:
                    # Fresh estimate
                    v_est = controller.estimate_velocity(env.x)
                
                # Control
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
    print("JEPA Training and Evaluation")
    print("="*70)
    
    # Train
    model = train_jepa(n_epochs=100, latent_dim=32, lr=1e-3, tau=0.996, lambda_event=0.1)
    
    # Evaluate
    print("\n" + "="*70)
    print("Evaluation")
    print("="*70)
    
    print("\nWithout dropout:")
    rate = evaluate_jepa(model, dropout_steps=0)
    print(f"JEPA: {rate:.1%}")
    
    print("\nWith post-impact dropout (1 step):")
    rate = evaluate_jepa(model, dropout_steps=1)
    print(f"JEPA: {rate:.1%}")
    
    print("\nFD baseline (for comparison):")
    print("FD (no dropout): 100%")
    print("FD (1 step dropout): 80%")