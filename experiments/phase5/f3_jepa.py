#!/usr/bin/env python3
"""
F3-JEPA: Unified Architecture for Variance + Dropout

Key insight: 
- F3 (Δx/dt input) solves variance at boundary
- JEPA (multi-step prediction) solves dropout
- Must balance: velocity decoding >> JEPA prediction

Architecture:
- Encoder: f_c(o, Δo/dt) → z
- Target Encoder (EMA): f_t(o, Δo/dt) → z_target
- Velocity Decoder: d(z) → v
- Predictor: g(z, a) → z_next

Losses:
- L_vel: Velocity consistency (F3 anchor)
- L_pred: JEPA latent prediction
- L_event: Impact prediction (auxiliary)
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
# F3-JEPA ARCHITECTURE
# =============================================================================

class F3Encoder(nn.Module):
    """Encoder with F3-normalized input: (x, Δx/dt)"""
    def __init__(self, latent_dim=32):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(2, 64), nn.ReLU(),
            nn.Linear(64, 64), nn.ReLU(),
            nn.Linear(64, latent_dim)
        )
    
    def forward(self, x, x_prev, dt=0.05):
        # F3 normalization: Δx/dt as physics-informed feature
        delta_v = (x - x_prev) / dt
        inp = torch.cat([x, delta_v], dim=-1)
        return self.net(inp)


class TargetEncoder(nn.Module):
    """EMA of context encoder for stable targets"""
    def __init__(self, latent_dim=32):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(2, 64), nn.ReLU(),
            nn.Linear(64, 64), nn.ReLU(),
            nn.Linear(64, latent_dim)
        )
    
    def forward(self, x, x_prev, dt=0.05):
        delta_v = (x - x_prev) / dt
        inp = torch.cat([x, delta_v], dim=-1)
        return self.net(inp)
    
    @torch.no_grad()
    def update_ema(self, encoder, tau=0.996):
        for tp, ep in zip(self.parameters(), encoder.parameters()):
            tp.data.mul_(tau).add_(ep.data, alpha=1 - tau)


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


class LatentPredictor(nn.Module):
    """Predict z_next from z and action"""
    def __init__(self, latent_dim=32):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(latent_dim + 1, 64), nn.ReLU(),
            nn.Linear(64, 64), nn.ReLU(),
            nn.Linear(64, latent_dim)
        )
    
    def forward(self, z, a):
        return self.net(torch.cat([z, a], dim=-1))


class EventHead(nn.Module):
    """Predict impact probability from latent state"""
    def __init__(self, latent_dim=32):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(latent_dim, 32), nn.ReLU(),
            nn.Linear(32, 1), nn.Sigmoid()
        )
    
    def forward(self, z):
        return self.net(z)


class F3JEPA(nn.Module):
    """F3-JEPA: Unified architecture"""
    def __init__(self, latent_dim=32):
        super().__init__()
        self.encoder = F3Encoder(latent_dim)
        self.target_encoder = TargetEncoder(latent_dim)
        self.velocity_decoder = VelocityDecoder(latent_dim)
        self.predictor = LatentPredictor(latent_dim)
        self.event_head = EventHead(latent_dim)
        
        # Initialize target encoder
        self.target_encoder.load_state_dict(self.encoder.state_dict())
    
    def encode(self, x, x_prev, dt=0.05):
        return self.encoder(x, x_prev, dt)
    
    def encode_target(self, x, x_prev, dt=0.05):
        with torch.no_grad():
            return self.target_encoder(x, x_prev, dt)
    
    def decode_velocity(self, z):
        return self.velocity_decoder(z)
    
    def predict(self, z, a):
        return self.predictor(z, a)
    
    def predict_impact(self, z):
        return self.event_head(z)
    
    def update_target(self, tau=0.996):
        self.target_encoder.update_ema(self.encoder, tau)

# =============================================================================
# TRAINING DATA
# =============================================================================

def generate_data(n_episodes=500, dt=0.05):
    data = {
        'x': [], 'x_prev': [], 'v': [], 'a': [],
        'x_next': [], 'x_prev_next': [], 'v_next': [],
        'impact': []
    }
    
    init_positions = [0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5]
    
    for ep in range(n_episodes):
        env = Ball(dt=dt)
        x0 = init_positions[ep % len(init_positions)]
        env.reset(x0)
        xp = x0
        
        for t in range(30):
            a = np.clip(1.5 * (2 - env.x) + (-2) * (-env.v), -2, 2)
            
            x_t = env.x
            xp_t = xp
            v_t = env.v
            
            env.step(a)
            
            x_tp1 = env.x
            xp_tp1 = x_t  # For next step, x_prev = current x
            v_tp1 = env.v
            impact_t = 1.0 if env.impact else 0.0
            
            data['x'].append(x_t)
            data['x_prev'].append(xp_t)
            data['v'].append(v_t)
            data['a'].append(a)
            data['x_next'].append(x_tp1)
            data['x_prev_next'].append(xp_tp1)
            data['v_next'].append(v_tp1)
            data['impact'].append(impact_t)
            
            xp = x_t
    
    for k in data:
        data[k] = torch.tensor(np.array(data[k]), dtype=torch.float32, device=device)
    
    return data

# =============================================================================
# TRAINING WITH BALANCED LOSSES
# =============================================================================

def train_f3jepa(
    n_epochs=100,
    latent_dim=32,
    lr=1e-3,
    tau=0.996,
    lambda_vel=10.0,    # HIGH: velocity precision is critical
    lambda_pred=0.1,    # LOW: don't wash out physics
    lambda_event=0.5,
    dt=0.05
):
    print("Generating training data...")
    data = generate_data(n_episodes=500, dt=dt)
    
    model = F3JEPA(latent_dim=latent_dim).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    
    batch_size = 64
    n_samples = len(data['x'])
    
    print(f"Training F3-JEPA for {n_epochs} epochs...")
    print(f"Loss weights: vel={lambda_vel}, pred={lambda_pred}, event={lambda_event}")
    
    for epoch in range(n_epochs):
        idx = torch.randperm(n_samples)
        
        total_loss = 0
        vel_loss = 0
        pred_loss = 0
        event_loss = 0
        
        for i in range(0, n_samples, batch_size):
            b = idx[i:i+batch_size]
            
            x_t = data['x'][b].unsqueeze(-1)
            xp_t = data['x_prev'][b].unsqueeze(-1)
            v_t = data['v'][b].unsqueeze(-1)
            a_t = data['a'][b].unsqueeze(-1)
            x_tp1 = data['x_next'][b].unsqueeze(-1)
            xp_tp1 = data['x_prev_next'][b].unsqueeze(-1)
            v_tp1 = data['v_next'][b].unsqueeze(-1)
            impact_t = data['impact'][b].unsqueeze(-1)
            
            # Encode current state
            z_t = model.encode(x_t, xp_t, dt)
            
            # Encode target state (EMA, stop-gradient)
            z_target = model.encode_target(x_tp1, xp_tp1, dt)
            
            # Predict next latent
            z_pred = model.predict(z_t, a_t)
            
            # Decode velocities
            v_pred_t = model.decode_velocity(z_t)
            v_pred_tp1 = model.decode_velocity(z_pred)  # From predicted latent
            
            # Predict impact
            impact_pred = model.predict_impact(z_t)
            
            # Losses
            # L_vel: Velocity consistency (F3 anchor)
            loss_vel = F.mse_loss(v_pred_t, v_t) + 0.5 * F.mse_loss(v_pred_tp1, v_tp1)
            
            # L_pred: JEPA latent prediction (with stop-gradient on target)
            loss_pred = F.mse_loss(z_pred, z_target.detach())
            
            # L_event: Impact prediction
            loss_event = F.binary_cross_entropy(impact_pred, impact_t)
            
            # Total loss
            loss = lambda_vel * loss_vel + lambda_pred * loss_pred + lambda_event * loss_event
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            model.update_target(tau)
            
            total_loss += loss.item()
            vel_loss += loss_vel.item()
            pred_loss += loss_pred.item()
            event_loss += loss_event.item()
        
        if (epoch + 1) % 20 == 0:
            print(f"Epoch {epoch+1}: total={total_loss:.2f}, vel={vel_loss:.4f}, pred={pred_loss:.4f}, event={event_loss:.4f}")
    
    return model

# =============================================================================
# CLOSED-LOOP CONTROLLER
# =============================================================================

class F3JEPAController:
    """Controller using F3-JEPA for velocity estimation"""
    
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
        """Estimate velocity using F3-JEPA"""
        
        if observation_available:
            # Fresh observation: encode with F3 normalization
            x_t = torch.tensor([[x]], dtype=torch.float32, device=device)
            xp_t = torch.tensor([[self.last_x]], dtype=torch.float32, device=device)
            
            with torch.no_grad():
                self.z = self.model.encode(x_t, xp_t, self.dt)
                v = self.model.decode_velocity(self.z).item()
            
            self.last_x = x
        else:
            # No observation: use predicted latent
            if self.z is not None and self.last_a is not None:
                a_t = torch.tensor([[self.last_a]], dtype=torch.float32, device=device)
                with torch.no_grad():
                    self.z = self.model.predict(self.z, a_t)
                    v = self.model.decode_velocity(self.z).item()
            else:
                v = 0.0
        
        return v

# =============================================================================
# EVALUATION
# =============================================================================

def evaluate_f3jepa(model, n_trials=100, dropout_steps=0, dt=0.05):
    """Evaluate F3-JEPA controller with post-impact dropout"""
    
    init_positions = [1.5, 2.0, 2.5, 3.0, 3.5]
    success = 0
    
    for x0 in init_positions:
        for _ in range(n_trials):
            env = Ball(dt=dt)
            env.reset(x0)
            controller = F3JEPAController(model, dt=dt)
            controller.reset(x0)
            
            freeze_countdown = 0
            
            for t in range(30):
                # Trigger dropout after impact
                if env.impact:
                    freeze_countdown = dropout_steps
                
                obs_available = freeze_countdown <= 0
                if freeze_countdown > 0:
                    freeze_countdown -= 1
                
                # Get velocity estimate
                v_est = controller.get_velocity(env.x, obs_available)
                
                # PD control
                a = np.clip(1.5 * (2 - env.x) + (-2) * (-v_est), -2, 2)
                controller.last_a = a
                
                env.step(a)
                
                if abs(env.x - 2.0) < 0.3:
                    success += 1
                    break
    
    return success / (len(init_positions) * n_trials)


def evaluate_fd(n_trials=100, dropout_steps=0, dt=0.05):
    """FD baseline"""
    init_positions = [1.5, 2.0, 2.5, 3.0, 3.5]
    success = 0
    
    for x0 in init_positions:
        for _ in range(n_trials):
            env = Ball(dt=dt)
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
                    v_est = (env.x - xp) / dt
                    xp = env.x
                
                a = np.clip(1.5 * (2 - env.x) + (-2) * (-v_est), -2, 2)
                env.step(a)
                
                if abs(env.x - 2.0) < 0.3:
                    success += 1
                    break
    
    return success / (len(init_positions) * n_trials)


def evaluate_variance(model, n_seeds=10, n_trials=50):
    """Test variance across training seeds"""
    results = []
    
    for seed in range(n_seeds):
        # Reinitialize model with different seed
        torch.manual_seed(seed)
        np.random.seed(seed)
        
        # Quick training
        model = train_f3jepa(
            n_epochs=50,
            latent_dim=32,
            lr=1e-3,
            lambda_vel=10.0,
            lambda_pred=0.1,
            lambda_event=0.5
        )
        
        rate = evaluate_f3jepa(model, n_trials=n_trials, dropout_steps=0)
        results.append(rate)
    
    return results

# =============================================================================
# MAIN
# =============================================================================

if __name__ == '__main__':
    print("="*70)
    print("F3-JEPA: Unified Architecture")
    print("="*70)
    
    # Train with balanced losses
    model = train_f3jepa(
        n_epochs=100,
        latent_dim=32,
        lr=1e-3,
        tau=0.996,
        lambda_vel=10.0,    # HIGH: velocity precision
        lambda_pred=0.1,    # LOW: don't wash out physics
        lambda_event=0.5
    )
    
    print("\n" + "="*70)
    print("Evaluation: Post-Impact Dropout")
    print("="*70)
    print(f"{'Dropout':>8} | {'F3-JEPA':>8} | {'FD':>8}")
    print("-"*30)
    
    for dropout in [0, 1, 2, 3, 5]:
        f3jepa_rate = evaluate_f3jepa(model, dropout_steps=dropout)
        fd_rate = evaluate_fd(dropout_steps=dropout)
        print(f"{dropout:>8} | {f3jepa_rate:>8.1%} | {fd_rate:>8.1%}")