#!/usr/bin/env python3
"""
Hopper F3-JEPA v2 - Proper predictor usage during dropout

Key fix: During dropout, roll forward the latent state using predictor,
then reconstruct observation from latent.
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
# F3-JEPA v2 - WITH OBSERVATION DECODER
# =============================================================================

class F3JEPAHopperV2(nn.Module):
    """
    F3-JEPA with observation decoder.
    
    Key: Can reconstruct observation from latent during dropout.
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
        
        # Velocity decoder: z -> velocity
        self.velocity_decoder = nn.Sequential(
            nn.Linear(latent_dim, 64), nn.ReLU(),
            nn.Linear(64, obs_dim)
        )
        
        # Observation decoder: z -> obs
        # (For reconstructing observation during dropout)
        self.obs_decoder = nn.Sequential(
            nn.Linear(latent_dim, 64), nn.ReLU(),
            nn.Linear(64, obs_dim)
        )
        
        # Latent predictor: (z, a) -> z_next
        self.predictor = nn.Sequential(
            nn.Linear(latent_dim + action_dim, 128), nn.ReLU(),
            nn.Linear(128, 128), nn.ReLU(),
            nn.Linear(128, latent_dim)
        )
        
        # Initialize target encoder
        self.target_encoder.load_state_dict(self.encoder.state_dict())
    
    def encode(self, obs, obs_prev, dt=0.002):
        velocity = (obs - obs_prev) / dt
        inp = torch.cat([obs, velocity], dim=-1)
        return self.encoder(inp)
    
    def encode_target(self, obs, obs_prev, dt=0.002):
        velocity = (obs - obs_prev) / dt
        inp = torch.cat([obs, velocity], dim=-1)
        with torch.no_grad():
            return self.target_encoder(inp)
    
    def decode_velocity(self, z):
        return self.velocity_decoder(z)
    
    def decode_obs(self, z):
        return self.obs_decoder(z)
    
    def predict_latent(self, z, action):
        return self.predictor(torch.cat([z, action], dim=-1))
    
    @torch.no_grad()
    def update_target(self, tau=0.996):
        for tp, ep in zip(self.target_encoder.parameters(), self.encoder.parameters()):
            tp.data.mul_(tau).add_(ep.data, alpha=1 - tau)

def generate_data(sac_model, n_episodes=200):
    """Generate more training data"""
    env = gym.make('Hopper-v4')
    
    data = {'obs': [], 'obs_prev': [], 'action': [], 'obs_next': [], 'obs_prev_next': []}
    
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

def train_f3jepa_v2(model, data, n_epochs=100, lambda_vel=10.0, lambda_obs=5.0, lambda_pred=0.1):
    """Train F3-JEPA v2 with observation reconstruction"""
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    
    batch_size = 256
    n_samples = len(data['obs'])
    
    for epoch in range(n_epochs):
        idx = torch.randperm(n_samples)
        total_loss = 0
        total_vel_loss = 0
        total_obs_loss = 0
        total_pred_loss = 0
        
        for i in range(0, n_samples, batch_size):
            b = idx[i:i+batch_size]
            
            # Encode
            z_t = model.encode(data['obs'][b], data['obs_prev'][b])
            z_target = model.encode_target(data['obs_next'][b], data['obs_prev_next'][b])
            
            # Predict
            z_pred = model.predict_latent(z_t, data['action'][b])
            
            # Decode velocity
            v_pred = model.decode_velocity(z_t)
            v_true = (data['obs'][b] - data['obs_prev'][b]) / 0.002
            
            # Decode observation
            obs_pred = model.decode_obs(z_t)
            
            # Losses
            loss_vel = F.mse_loss(v_pred, v_true)
            loss_obs = F.mse_loss(obs_pred, data['obs'][b])
            loss_pred = F.mse_loss(z_pred, z_target.detach())
            
            loss = lambda_vel * loss_vel + lambda_obs * loss_obs + lambda_pred * loss_pred
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            model.update_target()
            total_loss += loss.item()
            total_vel_loss += loss_vel.item()
            total_obs_loss += loss_obs.item()
            total_pred_loss += loss_pred.item()
        
        if (epoch + 1) % 20 == 0:
            print(f"Epoch {epoch+1}: loss={total_loss:.1f} (vel={total_vel_loss:.1f}, obs={total_obs_loss:.1f}, pred={total_pred_loss:.1f})")
    
    return model

# =============================================================================
# TEST WITH F3-JEPA v2 OBSERVER
# =============================================================================

def test_f3jepa_v2_observer(sac_model, jepa_model, n_episodes=20, dropout_duration=5, velocity_threshold=0.1):
    """
    Test F3-JEPA v2 as observer during critical dropout.
    
    Key: During dropout:
    1. Roll forward latent with predictor
    2. Reconstruct observation from latent
    3. Use reconstructed obs for policy
    """
    env = CriticalDropoutEnv(dropout_duration=dropout_duration, velocity_threshold=velocity_threshold)
    
    results = {'velocity_error': [], 'obs_error': [], 'reward': [], 'length': []}
    
    for ep in range(n_episodes):
        obs, _ = env.reset()
        obs_prev = obs.copy()
        total_reward = 0
        vel_errors = []
        obs_errors = []
        
        # Latent state
        z = None
        last_good_obs = obs.copy()
        last_good_obs_prev = obs_prev.copy()
        
        for step in range(1000):
            # Get action (use reconstructed obs if in dropout)
            action, _ = sac_model.predict(obs, deterministic=True)
            
            # Update latent if not in dropout
            if not env.dropout_countdown > 0:
                last_good_obs = obs.copy()
                last_good_obs_prev = obs_prev.copy()
                
                obs_tensor = torch.tensor(obs, dtype=torch.float32, device=device).unsqueeze(0)
                obs_prev_tensor = torch.tensor(obs_prev, dtype=torch.float32, device=device).unsqueeze(0)
                with torch.no_grad():
                    z = jepa_model.encode(obs_tensor, obs_prev_tensor)
            
            # Step environment
            obs_next, reward, term, trunc, info = env.step(action)
            total_reward += reward
            
            # During dropout, use predictor to update latent and reconstruct obs
            if info['dropout_active'] and z is not None:
                # Update latent with action
                action_tensor = torch.tensor(action, dtype=torch.float32, device=device).unsqueeze(0)
                with torch.no_grad():
                    z = jepa_model.predict_latent(z, action_tensor)
                
                    # Reconstruct observation for next step
                    obs_reconstructed = jepa_model.decode_obs(z).squeeze().cpu().numpy()
                    
                    # Use reconstructed obs for next action
                    obs = obs_reconstructed
                
                # Measure errors
                true_obs = info['true_obs']
                v_pred = jepa_model.decode_velocity(z).squeeze().detach().cpu().numpy()
                true_v = (true_obs - last_good_obs_prev) / 0.002
                vel_errors.append(np.mean(np.abs(v_pred - true_v)))
                obs_errors.append(np.mean(np.abs(obs_reconstructed - true_obs)))
            else:
                obs_prev = obs.copy()
                obs = obs_next
            
            if term or trunc:
                results['length'].append(step)
                break
        
        results['velocity_error'].extend(vel_errors)
        results['obs_error'].extend(obs_errors)
        results['reward'].append(total_reward)
    
    env.close()
    return results

# =============================================================================
# BASELINE TEST
# =============================================================================

def test_baseline(sac_model, n_episodes=20, dropout_duration=5, velocity_threshold=0.1):
    """Test FD baseline under dropout"""
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
                true_v = (info['true_obs'] - obs_prev) / 0.002
                fd_v = (obs - obs_prev) / 0.002  # Zero during dropout
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

# =============================================================================
# MAIN
# =============================================================================

if __name__ == '__main__':
    print("="*70)
    print("HOPPER F3-JEPA v2 - PROPER PREDICTOR USAGE")
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
    print(f"\nOracle reward (no dropout): {oracle_reward:.1f}")
    
    # FD Baseline
    print("\n" + "="*70)
    print("FD BASELINE TEST")
    print("="*70)
    
    fd_results = test_baseline(sac_model, n_episodes=30, dropout_duration=5, velocity_threshold=0.1)
    print(f"FD Baseline: reward={np.mean(fd_results['reward']):.1f} ± {np.std(fd_results['reward']):.1f}")
    print(f"  Velocity error: {np.mean(fd_results['velocity_error']):.2f}")
    print(f"  Length: {np.mean(fd_results['length']):.0f} steps")
    
    # Train F3-JEPA v2
    print("\n" + "="*70)
    print("TRAINING F3-JEPA v2")
    print("="*70)
    
    data = generate_data(sac_model, n_episodes=300)
    jepa_model = F3JEPAHopperV2(latent_dim=128).to(device)
    jepa_model = train_f3jepa_v2(jepa_model, data, n_epochs=100, lambda_vel=10.0, lambda_obs=5.0, lambda_pred=0.1)
    
    # Test F3-JEPA v2
    print("\n" + "="*70)
    print("F3-JEPA v2 OBSERVER TEST")
    print("="*70)
    
    jepa_results = test_f3jepa_v2_observer(sac_model, jepa_model, n_episodes=30, dropout_duration=5, velocity_threshold=0.1)
    
    print(f"\nF3-JEPA v2:")
    print(f"  Reward: {np.mean(jepa_results['reward']):.1f} ± {np.std(jepa_results['reward']):.1f}")
    print(f"  Velocity error: {np.mean(jepa_results['velocity_error']):.2f}")
    print(f"  Obs error: {np.mean(jepa_results['obs_error']):.4f}")
    print(f"  Length: {np.mean(jepa_results['length']):.0f} steps")
    
    # Summary
    print("\n" + "="*70)
    print("FINAL RESULTS")
    print("="*70)
    print(f"Oracle (no dropout): {oracle_reward:.1f}")
    print(f"FD Baseline (dropout): {np.mean(fd_results['reward']):.1f}")
    print(f"F3-JEPA v2 (dropout): {np.mean(jepa_results['reward']):.1f}")
    
    improvement = (np.mean(jepa_results['reward']) - np.mean(fd_results['reward'])) / np.mean(fd_results['reward']) * 100
    print(f"\nImprovement: {improvement:+.1f}%")
