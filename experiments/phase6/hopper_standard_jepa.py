#!/usr/bin/env python3
"""
Standard Latent JEPA — Latent Rollout with Residual Dynamics

Architecture: z_next = z + predictor(z, a)  (residual latent dynamics)
Includes: EMA target encoder, stop-gradient, aggressive velocity loss.

This is the architecture whose latent rollout DIVERGES exponentially
in high-dimensional continuous control under sensor dropout.

The main result: despite residual dynamics, EMA targets, and multi-step
training, the latent representation accumulates prediction error that
makes it WORSE than just using frozen observations.

See bulletproof_negative.py for the rigorous multi-experiment validation.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import gymnasium as gym
from stable_baselines3 import SAC
import warnings
warnings.filterwarnings('ignore')

device = torch.device('cuda')
print(f"Using device: {device}")

# =============================================================================
# ORACLE
# =============================================================================

def get_oracle():
    import os
    if os.path.exists('hopper_sac.zip'):
        env = gym.make('Hopper-v4')
        return SAC.load('hopper_sac.zip', env=env)
    raise FileNotFoundError("Oracle not found. Run hopper_critical.py first.")

# =============================================================================
# CRITICAL DROPOUT ENV
# =============================================================================

class CriticalDropoutEnv:
    def __init__(self, dropout_duration=5, velocity_threshold=0.1):
        self.env = gym.make('Hopper-v4')
        self.dropout_duration = dropout_duration
        self.velocity_threshold = velocity_threshold
        self.obs_prev = None
        self.frozen_obs = None
        self.dropout_countdown = 0
        self.step_count = 0
        
    def reset(self):
        obs, _ = self.env.reset()
        self.obs_prev = obs.copy()
        self.frozen_obs = obs.copy()
        self.dropout_countdown = 0
        self.step_count = 0
        return obs, {}
    
    def step(self, action):
        obs, reward, term, trunc, info = self.env.step(action)
        self.step_count += 1
        
        if self.obs_prev is not None and self.dropout_countdown == 0:
            velocity = obs - self.obs_prev
            accel = np.abs(velocity).max()
            
            if accel > self.velocity_threshold and self.step_count > 10:
                self.dropout_countdown = self.dropout_duration
                self.frozen_obs = obs.copy()
        
        info['true_obs'] = obs.copy()
        info['dropout_active'] = self.dropout_countdown > 0
        info['dropout_step'] = self.dropout_duration - self.dropout_countdown
        info['frozen_obs'] = self.frozen_obs.copy()
        
        if self.dropout_countdown > 0:
            obs_return = self.frozen_obs.copy()
            self.dropout_countdown -= 1
        else:
            obs_return = obs.copy()
            self.obs_prev = obs.copy()
        
        return obs_return, reward, term, trunc, info
    
    @property
    def observation_space(self):
        return self.env.observation_space
    
    @property
    def action_space(self):
        return self.env.action_space
    
    def close(self):
        self.env.close()

# =============================================================================
# Standard Latent JEPA - RESIDUAL LATENT DYNAMICS
# =============================================================================

class StandardLatentJEPAHopper(nn.Module):
    """
    Standard Latent JEPA with residual dynamics.
    
    Architecture:
    1. Residual predictor: z_next = z + Δz
    2. EMA target encoder with stop-gradient
    3. Aggressive velocity loss (λ_vel >> λ_pred)
    
    Despite these stabilization techniques, latent rollout still diverges.
    """
    def __init__(self, obs_dim=11, action_dim=3, latent_dim=64):
        super().__init__()
        
        # Encoder: (obs, velocity) -> z
        self.encoder = nn.Sequential(
            nn.Linear(obs_dim * 2, 128), nn.ReLU(),
            nn.Linear(128, 128), nn.ReLU(),
            nn.Linear(128, latent_dim)
        )
        
        # Target encoder (EMA)
        self.target_encoder = nn.Sequential(
            nn.Linear(obs_dim * 2, 128), nn.ReLU(),
            nn.Linear(128, 128), nn.ReLU(),
            nn.Linear(128, latent_dim)
        )
        self.target_encoder.load_state_dict(self.encoder.state_dict())
        
        # Velocity decoder: z -> velocity
        self.velocity_decoder = nn.Sequential(
            nn.Linear(latent_dim, 64), nn.ReLU(),
            nn.Linear(64, obs_dim)
        )
        
        # Residual predictor: (z, a) -> Δz
        # z_next = z + predictor(z, a)
        self.predictor = nn.Sequential(
            nn.Linear(latent_dim + action_dim, 128), nn.ReLU(),
            nn.Linear(128, 128), nn.ReLU(),
            nn.Linear(128, latent_dim)
        )
        
        self.obs_dim = obs_dim
        self.action_dim = action_dim
        self.latent_dim = latent_dim
    
    def encode(self, obs, obs_prev, dt=0.002):
        """Encode observation with finite-difference velocity"""
        velocity = (obs - obs_prev) / dt
        inp = torch.cat([obs, velocity], dim=-1)
        return self.encoder(inp)
    
    def encode_target(self, obs, obs_prev, dt=0.002):
        """Encode with EMA target encoder (stop-gradient)"""
        velocity = (obs - obs_prev) / dt
        inp = torch.cat([obs, velocity], dim=-1)
        with torch.no_grad():
            return self.target_encoder(inp)
    
    def decode_velocity(self, z):
        """Decode velocity from latent"""
        return self.velocity_decoder(z)
    
    def predict_residual(self, z, action):
        """Predict residual: z_next = z + Δz"""
        return self.predictor(torch.cat([z, action], dim=-1))
    
    def forward_latent(self, z, action):
        """One-step latent rollout: z_next = z + Δz"""
        delta_z = self.predict_residual(z, action)
        return z + delta_z
    
    @torch.no_grad()
    def update_target(self, tau=0.996):
        """EMA update of target encoder"""
        for tp, ep in zip(self.target_encoder.parameters(), self.encoder.parameters()):
            tp.data.mul_(tau).add_(ep.data, alpha=1 - tau)

def generate_data(sac_model, n_episodes=200):
    """Generate training data"""
    env = gym.make('Hopper-v4')
    
    data = {
        'obs': [], 'obs_prev': [], 'action': [],
        'obs_next': [], 'obs_prev_next': []
    }
    
    for ep in range(n_episodes):
        obs, _ = env.reset()
        obs_prev = obs.copy()
        
        for step in range(1000):
            action, _ = sac_model.predict(obs, deterministic=True)
            
            data['obs'].append(obs)
            data['obs_prev'].append(obs_prev)
            data['action'].append(action)
            
            obs_next, _, term, trunc, _ = env.step(action)
            
            data['obs_next'].append(obs_next)
            data['obs_prev_next'].append(obs)
            
            obs_prev = obs.copy()
            obs = obs_next
            
            if term or trunc:
                break
    
    env.close()
    
    for k in data:
        data[k] = torch.tensor(np.array(data[k]), dtype=torch.float32, device=device)
    
    print(f"Generated {len(data['obs'])} transitions")
    return data

def train_standard_jepa(model, data, n_epochs=100, lambda_vel=10.0, lambda_pred=0.1, dt=0.002):
    """
    Train Standard Latent JEPA with aggressive velocity loss.
    
    λ_vel=10.0 >> λ_pred=0.1 (velocity dominates prediction)
    """
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    
    batch_size = 256
    n_samples = len(data['obs'])
    
    for epoch in range(n_epochs):
        idx = torch.randperm(n_samples)
        total_loss = 0
        total_vel_loss = 0
        total_pred_loss = 0
        
        for i in range(0, n_samples, batch_size):
            b = idx[i:i+batch_size]
            
            # Encode current and next states
            z_t = model.encode(data['obs'][b], data['obs_prev'][b], dt)
            z_target = model.encode_target(data['obs_next'][b], data['obs_prev_next'][b], dt)
            
            # Residual prediction: z_pred = z_t + Δz
            delta_z = model.predict_residual(z_t, data['action'][b])
            z_pred = z_t + delta_z
            
            # Decode velocity from current latent
            v_pred = model.decode_velocity(z_t)
            v_true = (data['obs'][b] - data['obs_prev'][b]) / dt
            
            # Losses with aggressive imbalance
            loss_vel = F.mse_loss(v_pred, v_true)
            loss_pred = F.mse_loss(z_pred, z_target.detach())  # Stop-gradient on target
            
            loss = lambda_vel * loss_vel + lambda_pred * loss_pred
            
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            
            # EMA update
            model.update_target()
            
            total_loss += loss.item()
            total_vel_loss += loss_vel.item()
            total_pred_loss += loss_pred.item()
        
        if (epoch + 1) % 20 == 0:
            print(f"Epoch {epoch+1}: loss={total_loss:.1f} (vel={total_vel_loss:.1f}, pred={total_pred_loss:.4f})")
    
    return model

# =============================================================================
# EVALUATION WITH TRUE LATENT ROLLOUT
# =============================================================================

def test_standard_jepa(sac_model, jepa_model, n_episodes=30, dropout_duration=5, velocity_threshold=0.1, dt=0.002):
    """
    Test Standard Latent JEPA with true iterative latent rollout during dropout.
    
    During dropout:
    1. Evolve latent: z = z + predictor(z, action)
    2. Decode velocity: v = velocity_decoder(z)
    3. Integrate observation: obs = obs + v * dt
    """
    env = CriticalDropoutEnv(dropout_duration=dropout_duration, velocity_threshold=velocity_threshold)
    
    results = {'velocity_error': [], 'reward': [], 'length': []}
    
    for ep in range(n_episodes):
        obs, _ = env.reset()
        obs_prev = obs.copy()
        total_reward = 0
        vel_errors = []
        
        # Latent state
        z = None
        last_action = None
        
        for step in range(1000):
            # Get action from policy
            action, _ = sac_model.predict(obs, deterministic=True)
            action_tensor = torch.tensor(action, dtype=torch.float32, device=device).unsqueeze(0)
            
            # Step environment
            obs_next, reward, term, trunc, info = env.step(action)
            total_reward += reward
            
            if info['dropout_active']:
                # TRUE JEPA: Evolve latent state
                if z is None:
                    # First dropout step - encode from frozen observation
                    obs_tensor = torch.tensor(obs, dtype=torch.float32, device=device).unsqueeze(0)
                    obs_prev_tensor = torch.tensor(obs_prev, dtype=torch.float32, device=device).unsqueeze(0)
                    with torch.no_grad():
                        z = jepa_model.encode(obs_tensor, obs_prev_tensor, dt)
                
                # Evolve latent: z = z + Δz
                with torch.no_grad():
                    delta_z = jepa_model.predict_residual(z, action_tensor)
                    z = z + delta_z
                
                # Decode velocity
                with torch.no_grad():
                    v_pred = jepa_model.decode_velocity(z).squeeze().cpu().numpy()
                
                # Integrate observation
                obs = obs + v_pred * dt
                
                # Measure velocity error
                true_v = (info['true_obs'] - info['frozen_obs']) / (dt * (info['dropout_step'] + 1))
                vel_errors.append(np.mean(np.abs(v_pred - true_v)))
            else:
                # Fresh encode when not in dropout
                obs_tensor = torch.tensor(obs, dtype=torch.float32, device=device).unsqueeze(0)
                obs_prev_tensor = torch.tensor(obs_prev, dtype=torch.float32, device=device).unsqueeze(0)
                with torch.no_grad():
                    z = jepa_model.encode(obs_tensor, obs_prev_tensor, dt)
                
                obs_prev = obs.copy()
                obs = obs_next
            
            if term or trunc:
                results['length'].append(step)
                break
        
        results['velocity_error'].extend(vel_errors)
        results['reward'].append(total_reward)
    
    env.close()
    return results

def test_fd_baseline(sac_model, n_episodes=30, dropout_duration=5, velocity_threshold=0.1):
    """FD baseline"""
    env = CriticalDropoutEnv(dropout_duration=dropout_duration, velocity_threshold=velocity_threshold)
    
    results = {'velocity_error': [], 'reward': [], 'length': []}
    
    for ep in range(n_episodes):
        obs, _ = env.reset()
        obs_prev = obs.copy()
        total_reward = 0
        vel_errors = []
        
        for step in range(1000):
            action, _ = sac_model.predict(obs, deterministic=True)
            obs_next, reward, term, trunc, info = env.step(action)
            total_reward += reward
            
            if info['dropout_active']:
                true_v = (info['true_obs'] - info['frozen_obs']) / (0.002 * (info['dropout_step'] + 1))
                fd_v = np.zeros(11)
                vel_errors.append(np.mean(np.abs(fd_v - true_v)))
            
            obs_prev = obs.copy()
            obs = obs_next
            
            if term or trunc:
                results['length'].append(step)
                break
        
        results['velocity_error'].extend(vel_errors)
        results['reward'].append(total_reward)
    
    env.close()
    return results

def test_oracle_velocity(sac_model, n_episodes=30, dropout_duration=5, velocity_threshold=0.1, dt=0.002):
    """Oracle velocity (upper bound)"""
    env = CriticalDropoutEnv(dropout_duration=dropout_duration, velocity_threshold=velocity_threshold)
    
    results = {'reward': [], 'length': []}
    
    for ep in range(n_episodes):
        obs, _ = env.reset()
        total_reward = 0
        
        for step in range(1000):
            action, _ = sac_model.predict(obs, deterministic=True)
            obs_next, reward, term, trunc, info = env.step(action)
            total_reward += reward
            
            if info['dropout_active']:
                # Use true velocity
                true_v = (info['true_obs'] - info['frozen_obs']) / (dt * (info['dropout_step'] + 1))
                obs = info['frozen_obs'] + true_v * dt * (info['dropout_step'] + 1)
            else:
                obs = obs_next
            
            if term or trunc:
                results['length'].append(step)
                break
        
        results['reward'].append(total_reward)
    
    env.close()
    return results

# =============================================================================
# MULTI-STEP PREDICTION TRAINING
# =============================================================================

def train_standard_jepa_multistep(model, data, n_epochs=100, n_rollout=3, lambda_vel=10.0, lambda_pred=0.1, dt=0.002):
    """
    Train with multi-step latent rollout to improve long-range prediction.
    """
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    
    batch_size = 256
    n_samples = len(data['obs'])
    
    for epoch in range(n_epochs):
        idx = torch.randperm(n_samples)
        total_loss = 0
        total_vel_loss = 0
        total_pred_loss = 0
        
        for i in range(0, n_samples - n_rollout, batch_size):
            b = idx[i:i+batch_size]
            
            # Single-step prediction loss
            z_t = model.encode(data['obs'][b], data['obs_prev'][b], dt)
            z_target = model.encode_target(data['obs_next'][b], data['obs_prev_next'][b], dt)
            
            delta_z = model.predict_residual(z_t, data['action'][b])
            z_pred = z_t + delta_z
            
            v_pred = model.decode_velocity(z_t)
            v_true = (data['obs'][b] - data['obs_prev'][b]) / dt
            
            loss_vel = F.mse_loss(v_pred, v_true)
            loss_pred = F.mse_loss(z_pred, z_target.detach())
            
            # Multi-step rollout loss
            loss_multistep = 0
            z_rollout = z_t.clone()
            for k in range(min(n_rollout, n_samples - i - batch_size)):
                idx_k = b + k + 1
                if idx_k.max() < n_samples:
                    delta_z_k = model.predict_residual(z_rollout, data['action'][idx_k - 1])
                    z_rollout = z_rollout + delta_z_k
                    
                    z_target_k = model.encode_target(
                        data['obs_next'][idx_k - 1], 
                        data['obs_prev_next'][idx_k - 1], 
                        dt
                    )
                    loss_multistep += F.mse_loss(z_rollout, z_target_k.detach())
            
            loss = lambda_vel * loss_vel + lambda_pred * (loss_pred + 0.1 * loss_multistep)
            
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            
            model.update_target()
            
            total_loss += loss.item()
            total_vel_loss += loss_vel.item()
            total_pred_loss += loss_pred.item()
        
        if (epoch + 1) % 20 == 0:
            print(f"Epoch {epoch+1}: loss={total_loss:.1f} (vel={total_vel_loss:.1f}, pred={total_pred_loss:.4f})")
    
    return model

# =============================================================================
# MAIN
# =============================================================================

if __name__ == '__main__':
    print("="*70)
    print("Standard Latent JEPA - TRUE LATENT ROLLOUT WITH RESIDUAL DYNAMICS")
    print("="*70)
    print("Key innovations:")
    print("  1. Residual predictor: z_next = z + Δz")
    print("  2. Aggressive loss: λ_vel=10.0 >> λ_pred=0.1")
    print("  3. EMA target encoder with stop-gradient")
    print("  4. Iterative latent rollout during dropout")
    print("="*70)
    
    # Load oracle
    sac_model = get_oracle()
    
    # Oracle baseline
    env = gym.make('Hopper-v4')
    obs, _ = env.reset()
    oracle_reward = 0
    for _ in range(1000):
        action, _ = sac_model.predict(obs, deterministic=True)
        obs, reward, term, trunc, _ = env.step(action)
        oracle_reward += reward
        if term or trunc:
            break
    env.close()
    print(f"\nOracle (no dropout): {oracle_reward:.1f}")
    
    # FD Baseline
    print("\n" + "="*70)
    print("FD BASELINE")
    print("="*70)
    fd_results = test_fd_baseline(sac_model, n_episodes=30)
    print(f"Reward: {np.mean(fd_results['reward']):.1f} ± {np.std(fd_results['reward']):.1f}")
    print(f"Velocity error: {np.mean(fd_results['velocity_error']):.2f}")
    
    # Oracle velocity (upper bound)
    print("\n" + "="*70)
    print("ORACLE VELOCITY (UPPER BOUND)")
    print("="*70)
    oracle_v_results = test_oracle_velocity(sac_model, n_episodes=30)
    print(f"Reward: {np.mean(oracle_v_results['reward']):.1f} ± {np.std(oracle_v_results['reward']):.1f}")
    
    # Train Standard Latent JEPA
    print("\n" + "="*70)
    print("TRAINING Standard Latent JEPA")
    print("="*70)
    data = generate_data(sac_model, n_episodes=300)
    jepa_model = StandardLatentJEPAHopper(latent_dim=64).to(device)
    
    # Phase 1: Single-step training
    print("\nPhase 1: Single-step prediction...")
    jepa_model = train_standard_jepa(jepa_model, data, n_epochs=50, lambda_vel=10.0, lambda_pred=0.1)
    
    # Phase 2: Multi-step training
    print("\nPhase 2: Multi-step rollout...")
    jepa_model = train_standard_jepa_multistep(jepa_model, data, n_epochs=50, n_rollout=3, lambda_vel=10.0, lambda_pred=0.1)
    
    # Test Standard Latent JEPA
    print("\n" + "="*70)
    print("Standard Latent JEPA EVALUATION")
    print("="*70)
    jepa_results = test_standard_jepa(sac_model, jepa_model, n_episodes=30)
    
    print(f"\nStandard Latent JEPA:")
    print(f"  Reward: {np.mean(jepa_results['reward']):.1f} ± {np.std(jepa_results['reward']):.1f}")
    print(f"  Velocity error: {np.mean(jepa_results['velocity_error']):.2f}")
    print(f"  Length: {np.mean(jepa_results['length']):.0f} steps")
    
    # Summary
    print("\n" + "="*70)
    print("FINAL RESULTS")
    print("="*70)
    print(f"Oracle (no dropout):     {oracle_reward:.1f}")
    print(f"Oracle velocity:         {np.mean(oracle_v_results['reward']):.1f}")
    print(f"FD Baseline:             {np.mean(fd_results['reward']):.1f}")
    print(f"Standard Latent JEPA:              {np.mean(jepa_results['reward']):.1f}")
    
    print(f"\nVelocity Error:")
    print(f"  FD:       {np.mean(fd_results['velocity_error']):.2f}")
    print(f"  Standard Latent JEPA: {np.mean(jepa_results['velocity_error']):.2f}")
    
    # Check for explosion
    if np.mean(jepa_results['velocity_error']) > 1000:
        print("\n⚠️  WARNING: Velocity error > 1000, latent may be exploding!")
    else:
        print("\n✓ Velocity error stable, no explosion detected.")
    
    # Improvement metrics
    fd_mean = np.mean(fd_results['reward'])
    jepa_mean = np.mean(jepa_results['reward'])
    oracle_mean = np.mean(oracle_v_results['reward'])
    
    improvement = (jepa_mean - fd_mean) / fd_mean * 100
    gap_to_oracle = (oracle_mean - jepa_mean) / oracle_mean * 100
    
    print(f"\nImprovement over FD: {improvement:+.1f}%")
    print(f"Gap to oracle velocity: {gap_to_oracle:.1f}% remaining")