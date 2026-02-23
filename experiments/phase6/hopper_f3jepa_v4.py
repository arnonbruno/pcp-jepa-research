#!/usr/bin/env python3
"""
Hopper F3-JEPA v4 - Integrate velocity to update observation

Key: During dropout, use velocity estimate to predict true observation:
    obs_est = frozen_obs + velocity * dt * steps_since_dropout

This is what actually helps the policy.
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
    raise FileNotFoundError("Oracle not found.")

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
        self.dropout_start_step = 0
        
    def reset(self):
        obs, _ = self.env.reset()
        self.obs_prev = obs.copy()
        self.frozen_obs = obs.copy()
        self.dropout_countdown = 0
        self.step_count = 0
        self.dropout_start_step = 0
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
                self.dropout_start_step = self.step_count
        
        info['true_obs'] = obs.copy()
        info['dropout_active'] = self.dropout_countdown > 0
        info['dropout_step'] = self.dropout_duration - self.dropout_countdown  # 0-indexed
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
# VELOCITY PREDICTOR
# =============================================================================

class VelocityPredictor(nn.Module):
    """Predict velocity from action history"""
    def __init__(self, obs_dim=11, action_dim=3, history_len=5, latent_dim=64):
        super().__init__()
        self.history_len = history_len
        
        input_dim = obs_dim + history_len * action_dim
        
        self.net = nn.Sequential(
            nn.Linear(input_dim, 128), nn.ReLU(),
            nn.Linear(128, 128), nn.ReLU(),
            nn.Linear(128, latent_dim),
            nn.ReLU(),
            nn.Linear(latent_dim, obs_dim)
        )
    
    def forward(self, obs, action_history):
        if len(action_history.shape) == 2:
            action_history = action_history.unsqueeze(0)
        
        action_flat = action_history.reshape(action_history.shape[0], -1)
        inp = torch.cat([obs, action_flat], dim=-1)
        return self.net(inp)

def generate_data_with_history(sac_model, n_episodes=200, history_len=5):
    """Generate training data with action history"""
    env = gym.make('Hopper-v4')
    
    data = {
        'obs': [],
        'action_history': [],
        'velocity': []
    }
    
    for ep in range(n_episodes):
        obs, _ = env.reset()
        obs_prev = obs.copy()
        
        action_history = [np.zeros(3) for _ in range(history_len)]
        
        for step in range(1000):
            action, _ = sac_model.predict(obs, deterministic=True)
            
            action_history.pop(0)
            action_history.append(action.copy())
            
            obs_next, _, term, trunc, _ = env.step(action)
            
            velocity = (obs - obs_prev) / 0.002
            data['obs'].append(obs.copy())
            data['action_history'].append(np.array(action_history))
            data['velocity'].append(velocity)
            
            obs_prev = obs.copy()
            obs = obs_next
            
            if term or trunc:
                break
    
    env.close()
    
    for k in data:
        data[k] = torch.tensor(np.array(data[k]), dtype=torch.float32, device=device)
    
    print(f"Generated {len(data['obs'])} transitions")
    return data

def train_velocity_predictor(model, data, n_epochs=100):
    """Train velocity predictor"""
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    
    batch_size = 256
    n_samples = len(data['obs'])
    
    for epoch in range(n_epochs):
        idx = torch.randperm(n_samples)
        total_loss = 0
        
        for i in range(0, n_samples, batch_size):
            b = idx[i:i+batch_size]
            
            v_pred = model(data['obs'][b], data['action_history'][b])
            loss = F.mse_loss(v_pred, data['velocity'][b])
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
        
        if (epoch + 1) % 20 == 0:
            print(f"Epoch {epoch+1}: loss={total_loss:.4f}")
    
    return model

# =============================================================================
# TEST METHODS
# =============================================================================

def test_fd_baseline(sac_model, n_episodes=30, dropout_duration=5, velocity_threshold=0.1):
    """FD baseline: use frozen observation as-is"""
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
                true_v = (info['true_obs'] - info['frozen_obs']) / 0.002
                fd_v = np.zeros(11)  # FD gives zero during dropout
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

def test_oracle_velocity(sac_model, n_episodes=30, dropout_duration=5, velocity_threshold=0.1):
    """
    Oracle velocity: use TRUE velocity to estimate observation.
    
    obs_est = frozen_obs + true_velocity * dt * (step + 1)
    
    This is the upper bound on velocity-based estimation.
    """
    env = CriticalDropoutEnv(dropout_duration=dropout_duration, velocity_threshold=velocity_threshold)
    
    results = {'velocity_error': [], 'reward': [], 'length': []}
    
    for ep in range(n_episodes):
        obs, _ = env.reset()
        total_reward = 0
        vel_errors = []
        
        for step in range(1000):
            action, _ = sac_model.predict(obs, deterministic=True)
            obs_next, reward, term, trunc, info = env.step(action)
            total_reward += reward
            
            if info['dropout_active']:
                # Use TRUE velocity to estimate observation
                dropout_step = info['dropout_step']
                true_obs = info['true_obs']
                frozen_obs = info['frozen_obs']
                
                # Estimate: obs_est = frozen_obs + velocity * dt * (step + 1)
                true_velocity = (true_obs - frozen_obs) / (0.002 * (dropout_step + 1))
                obs_est = frozen_obs + true_velocity * 0.002 * (dropout_step + 1)
                
                # Use estimated observation for next step
                obs = obs_est
                
                # Measure error
                vel_errors.append(np.mean(np.abs(obs_est - true_obs)))
            else:
                obs = obs_next
            
            if term or trunc:
                results['length'].append(step)
                break
        
        results['velocity_error'].extend(vel_errors)
        results['reward'].append(total_reward)
    
    env.close()
    return results

def test_f3jepa_v4(sac_model, velocity_model, n_episodes=30, dropout_duration=5, velocity_threshold=0.1, history_len=5):
    """
    F3-JEPA v4: Use predicted velocity to estimate observation.
    
    obs_est = frozen_obs + predicted_velocity * dt * (step + 1)
    """
    env = CriticalDropoutEnv(dropout_duration=dropout_duration, velocity_threshold=velocity_threshold)
    
    results = {'velocity_error': [], 'reward': [], 'length': []}
    
    for ep in range(n_episodes):
        obs, _ = env.reset()
        total_reward = 0
        vel_errors = []
        
        action_history = [np.zeros(3) for _ in range(history_len)]
        
        for step in range(1000):
            action, _ = sac_model.predict(obs, deterministic=True)
            
            action_history.pop(0)
            action_history.append(action.copy())
            
            obs_next, reward, term, trunc, info = env.step(action)
            total_reward += reward
            
            if info['dropout_active']:
                # Predict velocity using F3-JEPA
                dropout_step = info['dropout_step']
                true_obs = info['true_obs']
                frozen_obs = info['frozen_obs']
                
                obs_tensor = torch.tensor(obs, dtype=torch.float32, device=device).unsqueeze(0)
                action_hist_tensor = torch.tensor(np.array(action_history), dtype=torch.float32, device=device).unsqueeze(0)
                
                with torch.no_grad():
                    v_pred = velocity_model(obs_tensor, action_hist_tensor).squeeze().cpu().numpy()
                
                # Estimate observation
                obs_est = frozen_obs + v_pred * 0.002 * (dropout_step + 1)
                
                # Use estimated observation for next step
                obs = obs_est
                
                # Measure error
                true_velocity = (true_obs - frozen_obs) / (0.002 * (dropout_step + 1))
                vel_errors.append(np.mean(np.abs(v_pred - true_velocity)))
            else:
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
    print("HOPPER F3-JEPA v4 - INTEGRATE VELOCITY TO UPDATE OBSERVATION")
    print("="*70)
    
    # Load oracle
    sac_model = get_oracle()
    
    # Oracle baseline (no dropout)
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
    
    # Test FD baseline
    print("\n" + "="*70)
    print("FD BASELINE")
    print("="*70)
    fd_results = test_fd_baseline(sac_model, n_episodes=30)
    print(f"Reward: {np.mean(fd_results['reward']):.1f} ± {np.std(fd_results['reward']):.1f}")
    print(f"Velocity error: {np.mean(fd_results['velocity_error']):.2f}")
    
    # Test oracle velocity (upper bound)
    print("\n" + "="*70)
    print("ORACLE VELOCITY (UPPER BOUND)")
    print("="*70)
    oracle_v_results = test_oracle_velocity(sac_model, n_episodes=30)
    print(f"Reward: {np.mean(oracle_v_results['reward']):.1f} ± {np.std(oracle_v_results['reward']):.1f}")
    print(f"Obs error: {np.mean(oracle_v_results['velocity_error']):.4f}")
    
    # Train velocity predictor
    print("\n" + "="*70)
    print("TRAINING F3-JEPA v4")
    print("="*70)
    data = generate_data_with_history(sac_model, n_episodes=300, history_len=5)
    velocity_model = VelocityPredictor(history_len=5).to(device)
    velocity_model = train_velocity_predictor(velocity_model, data, n_epochs=100)
    
    # Test F3-JEPA v4
    print("\n" + "="*70)
    print("F3-JEPA v4")
    print("="*70)
    jepa_results = test_f3jepa_v4(sac_model, velocity_model, n_episodes=30)
    print(f"Reward: {np.mean(jepa_results['reward']):.1f} ± {np.std(jepa_results['reward']):.1f}")
    print(f"Velocity error: {np.mean(jepa_results['velocity_error']):.2f}")
    
    # Summary
    print("\n" + "="*70)
    print("FINAL RESULTS")
    print("="*70)
    print(f"Oracle (no dropout): {oracle_reward:.1f}")
    print(f"Oracle velocity (upper bound): {np.mean(oracle_v_results['reward']):.1f}")
    print(f"FD Baseline: {np.mean(fd_results['reward']):.1f}")
    print(f"F3-JEPA v4: {np.mean(jepa_results['reward']):.1f}")
    
    print(f"\nVelocity error:")
    print(f"  FD: {np.mean(fd_results['velocity_error']):.2f}")
    print(f"  F3-JEPA v4: {np.mean(jepa_results['velocity_error']):.2f}")
    
    # Improvement
    fd_mean = np.mean(fd_results['reward'])
    jepa_mean = np.mean(jepa_results['reward'])
    oracle_v_mean = np.mean(oracle_v_results['reward'])
    
    print(f"\nImprovement over FD: {(jepa_mean - fd_mean) / fd_mean * 100:+.1f}%")
    print(f"Gap to oracle velocity: {(oracle_v_mean - jepa_mean) / oracle_v_mean * 100:.1f}% remaining")
