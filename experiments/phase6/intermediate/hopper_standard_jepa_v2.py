#!/usr/bin/env python3
"""
F3-JEPA v5.1 - Fixed observation integration

Fix: During dropout, accumulate velocity from frozen base, not from updated obs.
This prevents compounding drift.

obs_est = frozen_obs + cumulative_velocity * dt * step
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
# F3-JEPA v5.1
# =============================================================================

class F3JEPAHopperV51(nn.Module):
    """F3-JEPA v5.1 with residual dynamics"""
    def __init__(self, obs_dim=11, action_dim=3, latent_dim=64):
        super().__init__()
        
        self.encoder = nn.Sequential(
            nn.Linear(obs_dim * 2, 128), nn.ReLU(),
            nn.Linear(128, 128), nn.ReLU(),
            nn.Linear(128, latent_dim)
        )
        
        self.target_encoder = nn.Sequential(
            nn.Linear(obs_dim * 2, 128), nn.ReLU(),
            nn.Linear(128, 128), nn.ReLU(),
            nn.Linear(128, latent_dim)
        )
        self.target_encoder.load_state_dict(self.encoder.state_dict())
        
        self.velocity_decoder = nn.Sequential(
            nn.Linear(latent_dim, 64), nn.ReLU(),
            nn.Linear(64, obs_dim)
        )
        
        self.predictor = nn.Sequential(
            nn.Linear(latent_dim + action_dim, 128), nn.ReLU(),
            nn.Linear(128, 128), nn.ReLU(),
            nn.Linear(128, latent_dim)
        )
    
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
    
    def predict_residual(self, z, action):
        return self.predictor(torch.cat([z, action], dim=-1))
    
    @torch.no_grad()
    def update_target(self, tau=0.996):
        for tp, ep in zip(self.target_encoder.parameters(), self.encoder.parameters()):
            tp.data.mul_(tau).add_(ep.data, alpha=1 - tau)

def generate_data(sac_model, n_episodes=200):
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

def train_f3jepa(model, data, n_epochs=100, lambda_vel=10.0, lambda_pred=0.1, dt=0.002):
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
            
            z_t = model.encode(data['obs'][b], data['obs_prev'][b], dt)
            z_target = model.encode_target(data['obs_next'][b], data['obs_prev_next'][b], dt)
            
            delta_z = model.predict_residual(z_t, data['action'][b])
            z_pred = z_t + delta_z
            
            v_pred = model.decode_velocity(z_t)
            v_true = (data['obs'][b] - data['obs_prev'][b]) / dt
            
            loss_vel = F.mse_loss(v_pred, v_true)
            loss_pred = F.mse_loss(z_pred, z_target.detach())
            
            loss = lambda_vel * loss_vel + lambda_pred * loss_pred
            
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            
            model.update_target()
            
            total_loss += loss.item()
            total_vel_loss += loss_vel.item()
            total_pred_loss += loss_pred.item()
        
        if (epoch + 1) % 20 == 0:
            print(f"Epoch {epoch+1}: vel_loss={total_vel_loss:.1f}, pred_loss={total_pred_loss:.4f}")
    
    return model

# =============================================================================
# EVALUATION - TWO STRATEGIES
# =============================================================================

def test_f3jepa_accumulated(sac_model, jepa_model, n_episodes=30, dropout_duration=5, velocity_threshold=0.1, dt=0.002):
    """
    Strategy 1: Accumulate velocity from frozen base.
    
    obs_est = frozen_obs + cumulative_velocity * dt
    
    This prevents drift by always starting from frozen_obs.
    """
    env = CriticalDropoutEnv(dropout_duration=dropout_duration, velocity_threshold=velocity_threshold)
    
    results = {'velocity_error': [], 'reward': [], 'length': []}
    
    for ep in range(n_episodes):
        obs, _ = env.reset()
        obs_prev = obs.copy()
        total_reward = 0
        vel_errors = []
        
        for step in range(1000):
            action, _ = sac_model.predict(obs, deterministic=True)
            action_tensor = torch.tensor(action, dtype=torch.float32, device=device).unsqueeze(0)
            
            obs_next, reward, term, trunc, info = env.step(action)
            total_reward += reward
            
            if info['dropout_active']:
                dropout_step = info['dropout_step']
                frozen_obs = info['frozen_obs']
                true_obs = info['true_obs']
                
                # Encode from frozen observation
                frozen_tensor = torch.tensor(frozen_obs, dtype=torch.float32, device=device).unsqueeze(0)
                obs_prev_tensor = torch.tensor(obs_prev, dtype=torch.float32, device=device).unsqueeze(0)
                
                with torch.no_grad():
                    z = jepa_model.encode(frozen_tensor, obs_prev_tensor, dt)
                
                # Rollout latent for dropout_step steps
                # Note: We need to track actions during dropout
                # For simplicity, use current action
                with torch.no_grad():
                    for _ in range(dropout_step + 1):
                        delta_z = jepa_model.predict_residual(z, action_tensor)
                        z = z + delta_z
                
                # Decode velocity
                with torch.no_grad():
                    v_pred = jepa_model.decode_velocity(z).squeeze().cpu().numpy()
                
                # Estimate observation from frozen base
                obs_est = frozen_obs + v_pred * dt * (dropout_step + 1)
                
                # Use estimated observation
                obs = obs_est
                
                # Velocity error
                true_v = (true_obs - frozen_obs) / (dt * (dropout_step + 1))
                vel_errors.append(np.mean(np.abs(v_pred - true_v)))
            else:
                obs_prev = obs.copy()
                obs = obs_next
            
            if term or trunc:
                results['length'].append(step)
                break
        
        results['velocity_error'].extend(vel_errors)
        results['reward'].append(total_reward)
    
    env.close()
    return results

def test_f3jepa_instantaneous(sac_model, jepa_model, n_episodes=30, dropout_duration=5, velocity_threshold=0.1, dt=0.002):
    """
    Strategy 2: Use instantaneous velocity estimate.
    
    obs_est = frozen_obs + v_pred * dt * (dropout_step + 1)
    
    Direct velocity prediction without latent rollout.
    """
    env = CriticalDropoutEnv(dropout_duration=dropout_duration, velocity_threshold=velocity_threshold)
    
    results = {'velocity_error': [], 'reward': [], 'length': []}
    
    for ep in range(n_episodes):
        obs, _ = env.reset()
        obs_prev = obs.copy()
        total_reward = 0
        vel_errors = []
        
        # Track actions during dropout
        dropout_actions = []
        
        for step in range(1000):
            action, _ = sac_model.predict(obs, deterministic=True)
            action_tensor = torch.tensor(action, dtype=torch.float32, device=device).unsqueeze(0)
            
            obs_next, reward, term, trunc, info = env.step(action)
            total_reward += reward
            
            if info['dropout_active']:
                dropout_step = info['dropout_step']
                frozen_obs = info['frozen_obs']
                true_obs = info['true_obs']
                
                # Store action
                dropout_actions.append(action.copy())
                
                # Encode from frozen observation with FD velocity
                frozen_tensor = torch.tensor(frozen_obs, dtype=torch.float32, device=device).unsqueeze(0)
                obs_prev_tensor = torch.tensor(obs_prev, dtype=torch.float32, device=device).unsqueeze(0)
                
                with torch.no_grad():
                    # Single encode, no rollout
                    z = jepa_model.encode(frozen_tensor, obs_prev_tensor, dt)
                    v_pred = jepa_model.decode_velocity(z).squeeze().cpu().numpy()
                
                # Estimate observation
                obs_est = frozen_obs + v_pred * dt * (dropout_step + 1)
                obs = obs_est
                
                # Velocity error
                true_v = (true_obs - frozen_obs) / (dt * (dropout_step + 1))
                vel_errors.append(np.mean(np.abs(v_pred - true_v)))
            else:
                dropout_actions = []
                obs_prev = obs.copy()
                obs = obs_next
            
            if term or trunc:
                results['length'].append(step)
                break
        
        results['velocity_error'].extend(vel_errors)
        results['reward'].append(total_reward)
    
    env.close()
    return results

def test_f3jepa_with_action_history(sac_model, jepa_model, n_episodes=30, dropout_duration=5, velocity_threshold=0.1, dt=0.002):
    """
    Strategy 3: Use action-conditioned prediction.
    
    Accumulate predicted deltas using actual actions taken during dropout.
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
        frozen_obs = None
        
        for step in range(1000):
            action, _ = sac_model.predict(obs, deterministic=True)
            action_tensor = torch.tensor(action, dtype=torch.float32, device=device).unsqueeze(0)
            
            obs_next, reward, term, trunc, info = env.step(action)
            total_reward += reward
            
            if info['dropout_active']:
                dropout_step = info['dropout_step']
                
                if dropout_step == 0:
                    # First dropout step - initialize latent
                    frozen_obs = info['frozen_obs']
                    obs_prev_tensor = torch.tensor(obs_prev, dtype=torch.float32, device=device).unsqueeze(0)
                    frozen_tensor = torch.tensor(frozen_obs, dtype=torch.float32, device=device).unsqueeze(0)
                    
                    with torch.no_grad():
                        z = jepa_model.encode(frozen_tensor, obs_prev_tensor, dt)
                
                # Roll forward latent
                with torch.no_grad():
                    delta_z = jepa_model.predict_residual(z, action_tensor)
                    z = z + delta_z
                    v_pred = jepa_model.decode_velocity(z).squeeze().cpu().numpy()
                
                # Update observation estimate
                obs = frozen_obs + v_pred * dt * (dropout_step + 1)
                
                # Error
                true_v = (info['true_obs'] - frozen_obs) / (dt * (dropout_step + 1))
                vel_errors.append(np.mean(np.abs(v_pred - true_v)))
            else:
                obs_prev = obs.copy()
                obs = obs_next
                z = None
                frozen_obs = None
            
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
# MAIN
# =============================================================================

if __name__ == '__main__':
    print("="*70)
    print("F3-JEPA v5.1 - FIXED OBSERVATION INTEGRATION")
    print("="*70)
    
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
    
    # Oracle velocity
    print("\n" + "="*70)
    print("ORACLE VELOCITY")
    print("="*70)
    oracle_v_results = test_oracle_velocity(sac_model, n_episodes=30)
    print(f"Reward: {np.mean(oracle_v_results['reward']):.1f} ± {np.std(oracle_v_results['reward']):.1f}")
    
    # Train F3-JEPA
    print("\n" + "="*70)
    print("TRAINING F3-JEPA v5.1")
    print("="*70)
    data = generate_data(sac_model, n_episodes=300)
    jepa_model = F3JEPAHopperV51(latent_dim=64).to(device)
    jepa_model = train_f3jepa(jepa_model, data, n_epochs=100, lambda_vel=10.0, lambda_pred=0.1)
    
    # Test all strategies
    print("\n" + "="*70)
    print("TESTING STRATEGIES")
    print("="*70)
    
    print("\nStrategy 1: Accumulated from frozen base...")
    s1_results = test_f3jepa_accumulated(sac_model, jepa_model, n_episodes=30)
    print(f"  Reward: {np.mean(s1_results['reward']):.1f} ± {np.std(s1_results['reward']):.1f}")
    print(f"  Velocity error: {np.mean(s1_results['velocity_error']):.2f}")
    
    print("\nStrategy 2: Instantaneous velocity...")
    s2_results = test_f3jepa_instantaneous(sac_model, jepa_model, n_episodes=30)
    print(f"  Reward: {np.mean(s2_results['reward']):.1f} ± {np.std(s2_results['reward']):.1f}")
    print(f"  Velocity error: {np.mean(s2_results['velocity_error']):.2f}")
    
    print("\nStrategy 3: Action-conditioned rollout...")
    s3_results = test_f3jepa_with_action_history(sac_model, jepa_model, n_episodes=30)
    print(f"  Reward: {np.mean(s3_results['reward']):.1f} ± {np.std(s3_results['reward']):.1f}")
    print(f"  Velocity error: {np.mean(s3_results['velocity_error']):.2f}")
    
    # Summary
    print("\n" + "="*70)
    print("FINAL RESULTS")
    print("="*70)
    print(f"Oracle (no dropout):     {oracle_reward:.1f}")
    print(f"Oracle velocity:         {np.mean(oracle_v_results['reward']):.1f}")
    print(f"FD Baseline:             {np.mean(fd_results['reward']):.1f}")
    print(f"Strategy 1 (accumulated): {np.mean(s1_results['reward']):.1f}")
    print(f"Strategy 2 (instantaneous): {np.mean(s2_results['reward']):.1f}")
    print(f"Strategy 3 (action-cond): {np.mean(s3_results['reward']):.1f}")
    
    # Best strategy
    best_reward = max(
        np.mean(s1_results['reward']),
        np.mean(s2_results['reward']),
        np.mean(s3_results['reward'])
    )
    best_strategy = np.argmax([
        np.mean(s1_results['reward']),
        np.mean(s2_results['reward']),
        np.mean(s3_results['reward'])
    ]) + 1
    
    print(f"\nBest: Strategy {best_strategy} with reward {best_reward:.1f}")
    improvement = (best_reward - np.mean(fd_results['reward'])) / np.mean(fd_results['reward']) * 100
    print(f"Improvement over FD: {improvement:+.1f}%")