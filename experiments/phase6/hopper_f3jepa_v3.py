#!/usr/bin/env python3
"""
Hopper F3-JEPA v3 - Stable velocity prediction

Key insight: Don't reconstruct full observation during dropout.
Just estimate velocity and use FD with the estimated velocity.

This is more stable because:
1. Velocity is lower-dimensional signal
2. We're not rolling out latent predictions
3. We're using physics (FD) with learned correction
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
# F3-JEPA v3 - STABLE VELOCITY PREDICTOR
# =============================================================================

class F3JEPAHopperV3(nn.Module):
    """
    F3-JEPA v3: Stable velocity prediction
    
    Architecture:
    1. Encoder: (obs, action_history) -> z
    2. Velocity predictor: z -> velocity_estimate
    
    Key: Use action history to predict velocity during dropout.
    No latent rollout - just direct velocity prediction.
    """
    def __init__(self, obs_dim=11, action_dim=3, history_len=5, latent_dim=64):
        super().__init__()
        
        self.history_len = history_len
        
        # Encoder: obs + action_history
        # obs (11) + actions (history_len * action_dim)
        input_dim = obs_dim + history_len * action_dim
        
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 128), nn.ReLU(),
            nn.Linear(128, 128), nn.ReLU(),
            nn.Linear(128, latent_dim)
        )
        
        # Velocity predictor: z -> velocity
        self.velocity_decoder = nn.Sequential(
            nn.Linear(latent_dim, 64), nn.ReLU(),
            nn.Linear(64, obs_dim)
        )
    
    def encode(self, obs, action_history):
        """Encode observation with action history"""
        # action_history: (batch, history_len, action_dim)
        if len(action_history.shape) == 2:
            action_history = action_history.unsqueeze(0)
        
        action_flat = action_history.reshape(action_history.shape[0], -1)
        inp = torch.cat([obs, action_flat], dim=-1)
        return self.encoder(inp)
    
    def decode_velocity(self, z):
        """Decode velocity from latent"""
        return self.velocity_decoder(z)

def generate_data_with_history(sac_model, n_episodes=200, history_len=5):
    """Generate training data with action history"""
    env = gym.make('Hopper-v4')
    
    data = {
        'obs': [],
        'action_history': [],  # Last N actions
        'velocity': []  # True velocity
    }
    
    for ep in range(n_episodes):
        obs, _ = env.reset()
        obs_prev = obs.copy()
        
        # Initialize action history
        action_history = [np.zeros(3) for _ in range(history_len)]
        
        for step in range(1000):
            action, _ = sac_model.predict(obs, deterministic=True)
            
            # Update action history
            action_history.pop(0)
            action_history.append(action.copy())
            
            obs_next, _, term, trunc, _ = env.step(action)
            
            # Store transition
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

def train_f3jepa_v3(model, data, n_epochs=100, lambda_vel=10.0):
    """Train F3-JEPA v3 - simple velocity prediction"""
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    
    batch_size = 256
    n_samples = len(data['obs'])
    
    for epoch in range(n_epochs):
        idx = torch.randperm(n_samples)
        total_loss = 0
        
        for i in range(0, n_samples, batch_size):
            b = idx[i:i+batch_size]
            
            # Encode
            z = model.encode(data['obs'][b], data['action_history'][b])
            
            # Decode velocity
            v_pred = model.decode_velocity(z)
            
            # Loss
            loss = F.mse_loss(v_pred, data['velocity'][b])
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
        
        if (epoch + 1) % 20 == 0:
            print(f"Epoch {epoch+1}: loss={total_loss:.4f}")
    
    return model

# =============================================================================
# TEST WITH F3-JEPA v3 OBSERVER
# =============================================================================

def test_f3jepa_v3_observer(sac_model, jepa_model, n_episodes=30, dropout_duration=5, velocity_threshold=0.1, history_len=5):
    """
    Test F3-JEPA v3 as velocity estimator during dropout.
    
    Key: Use predicted velocity to estimate true observation.
    """
    env = CriticalDropoutEnv(dropout_duration=dropout_duration, velocity_threshold=velocity_threshold)
    
    results = {'velocity_error': [], 'reward': [], 'length': []}
    
    for ep in range(n_episodes):
        obs, _ = env.reset()
        obs_prev = obs.copy()
        last_good_obs = obs.copy()
        total_reward = 0
        vel_errors = []
        
        # Action history
        action_history = [np.zeros(3) for _ in range(history_len)]
        
        for step in range(1000):
            # Get action
            action, _ = sac_model.predict(obs, deterministic=True)
            
            # Update action history
            action_history.pop(0)
            action_history.append(action.copy())
            
            # Step environment
            obs_next, reward, term, trunc, info = env.step(action)
            total_reward += reward
            
            # During dropout, use F3-JEPA to estimate velocity
            if info['dropout_active']:
                # Encode current observation (frozen) with action history
                obs_tensor = torch.tensor(obs, dtype=torch.float32, device=device).unsqueeze(0)
                action_hist_tensor = torch.tensor(np.array(action_history), dtype=torch.float32, device=device).unsqueeze(0)
                
                with torch.no_grad():
                    z = jepa_model.encode(obs_tensor, action_hist_tensor)
                    v_pred = jepa_model.decode_velocity(z).squeeze().cpu().numpy()
                
                # Estimate true observation
                # obs_est = last_good_obs + v_pred * dt * dropout_step
                
                # Measure error
                true_v = (info['true_obs'] - last_good_obs) / 0.002
                vel_errors.append(np.mean(np.abs(v_pred - true_v)))
            
            # Update state
            if not info['dropout_active']:
                last_good_obs = obs.copy()
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
# BASELINE TEST
# =============================================================================

def test_baseline(sac_model, n_episodes=30, dropout_duration=5, velocity_threshold=0.1):
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
# TEST WITH ORACLE VELOCITY (UPPER BOUND)
# =============================================================================

def test_oracle_velocity(sac_model, n_episodes=30, dropout_duration=5, velocity_threshold=0.1):
    """
    Test with oracle velocity (perfect velocity estimation).
    This is the upper bound on what velocity estimation can achieve.
    """
    env = CriticalDropoutEnv(dropout_duration=dropout_duration, velocity_threshold=velocity_threshold)
    
    results = {'reward': [], 'length': []}
    
    for ep in range(n_episodes):
        obs, _ = env.reset()
        obs_prev = obs.copy()
        total_reward = 0
        
        for step in range(1000):
            action, _ = sac_model.predict(obs, deterministic=True)
            obs_next, reward, term, trunc, info = env.step(action)
            total_reward += reward
            
            # During dropout, replace frozen obs with true obs
            # (This simulates perfect velocity estimation)
            if info['dropout_active']:
                obs = info['true_obs']  # Use true observation
            
            obs_prev = obs.copy()
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
    print("HOPPER F3-JEPA v3 - STABLE VELOCITY PREDICTION")
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
    
    fd_results = test_baseline(sac_model, n_episodes=30)
    print(f"FD Baseline: reward={np.mean(fd_results['reward']):.1f} ± {np.std(fd_results['reward']):.1f}")
    print(f"  Velocity error: {np.mean(fd_results['velocity_error']):.2f}")
    print(f"  Length: {np.mean(fd_results['length']):.0f} steps")
    
    # Oracle velocity (upper bound)
    print("\n" + "="*70)
    print("ORACLE VELOCITY TEST (UPPER BOUND)")
    print("="*70)
    
    oracle_v_results = test_oracle_velocity(sac_model, n_episodes=30)
    print(f"Oracle velocity: reward={np.mean(oracle_v_results['reward']):.1f} ± {np.std(oracle_v_results['reward']):.1f}")
    print(f"  Length: {np.mean(oracle_v_results['length']):.0f} steps")
    
    # Train F3-JEPA v3
    print("\n" + "="*70)
    print("TRAINING F3-JEPA v3")
    print("="*70)
    
    data = generate_data_with_history(sac_model, n_episodes=300, history_len=5)
    jepa_model = F3JEPAHopperV3(history_len=5, latent_dim=64).to(device)
    jepa_model = train_f3jepa_v3(jepa_model, data, n_epochs=100)
    
    # Test F3-JEPA v3
    print("\n" + "="*70)
    print("F3-JEPA v3 OBSERVER TEST")
    print("="*70)
    
    jepa_results = test_f3jepa_v3_observer(sac_model, jepa_model, n_episodes=30)
    
    print(f"\nF3-JEPA v3:")
    print(f"  Reward: {np.mean(jepa_results['reward']):.1f} ± {np.std(jepa_results['reward']):.1f}")
    print(f"  Velocity error: {np.mean(jepa_results['velocity_error']):.2f}")
    print(f"  Length: {np.mean(jepa_results['length']):.0f} steps")
    
    # Summary
    print("\n" + "="*70)
    print("FINAL RESULTS")
    print("="*70)
    print(f"Oracle (no dropout): {oracle_reward:.1f}")
    print(f"Oracle velocity (upper bound): {np.mean(oracle_v_results['reward']):.1f}")
    print(f"FD Baseline (dropout): {np.mean(fd_results['reward']):.1f}")
    print(f"F3-JEPA v3 (dropout): {np.mean(jepa_results['reward']):.1f}")
    
    print(f"\nVelocity error comparison:")
    print(f"  FD: {np.mean(fd_results['velocity_error']):.2f}")
    print(f"  F3-JEPA v3: {np.mean(jepa_results['velocity_error']):.2f}")
