#!/usr/bin/env python3
"""
Hopper Critical Dropout Experiment

Focus on the specific failure mode from bouncing ball:
- Post-impact dropout (velocity becomes unreliable)
- Long dropout duration (5-10 steps)
- Measure velocity estimation error during dropout

Key insight: FD fails because it can't estimate velocity during blackout.
F3-JEPA should maintain velocity estimates via latent prediction.
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
# ORACLE TRAINING
# =============================================================================

def get_oracle(train_timesteps=100000):
    """Get or train SAC oracle"""
    import os
    if os.path.exists('hopper_sac.zip'):
        print("Loading existing oracle...")
        env = gym.make('Hopper-v4')
        return SAC.load('hopper_sac.zip', env=env)
    
    print("Training oracle...")
    env = gym.make('Hopper-v4')
    model = SAC('MlpPolicy', env, learning_rate=3e-4, buffer_size=100000, 
                learning_starts=1000, batch_size=256, verbose=0)
    model.learn(total_timesteps=train_timesteps, progress_bar=True)
    model.save('hopper_sac.zip')
    return model

# =============================================================================
# CRITICAL DROPOUT ENVIRONMENT
# =============================================================================

class CriticalDropoutEnv:
    """
    Dropout triggered at critical moments: when velocity changes rapidly.
    
    Simulates sensor blackout during high-acceleration events.
    """
    
    def __init__(self, dropout_duration=5, velocity_threshold=0.1):
        self.env = gym.make('Hopper-v4')
        self.dropout_duration = dropout_duration
        self.velocity_threshold = velocity_threshold
        
        self.obs_prev = None
        self.frozen_obs = None
        self.dropout_countdown = 0
        self.step_count = 0
        
        # Track dropout events
        self.dropout_events = []
        
    def reset(self):
        obs, _ = self.env.reset()
        self.obs_prev = obs.copy()
        self.frozen_obs = obs.copy()
        self.dropout_countdown = 0
        self.step_count = 0
        self.dropout_events = []
        return obs, {}
    
    def step(self, action):
        obs, reward, term, trunc, info = self.env.step(action)
        self.step_count += 1
        
        # Compute velocity change
        if self.obs_prev is not None and self.dropout_countdown == 0:
            velocity = obs - self.obs_prev
            accel = np.abs(velocity).max()
            
            # Trigger dropout on high acceleration
            if accel > self.velocity_threshold and self.step_count > 10:
                self.dropout_countdown = self.dropout_duration
                self.frozen_obs = obs.copy()
                self.dropout_events.append(self.step_count)
        
        # Store true observation for evaluation
        info['true_obs'] = obs.copy()
        info['dropout_active'] = self.dropout_countdown > 0
        
        # Apply dropout
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
# VELOCITY ESTIMATION TEST
# =============================================================================

def test_velocity_estimation(model, n_episodes=20, dropout_duration=5, velocity_threshold=0.1):
    """
    Test how well different methods estimate velocity during dropout.
    
    This is the key metric: can we estimate velocity when observations are frozen?
    """
    env = CriticalDropoutEnv(dropout_duration=dropout_duration, velocity_threshold=velocity_threshold)
    
    results = {
        'fd': {'velocity_error': [], 'reward': [], 'length': []},
        'oracle': {'velocity_error': [], 'reward': [], 'length': []}
    }
    
    for ep in range(n_episodes):
        obs, _ = env.reset()
        obs_prev = obs.copy()
        total_reward = 0
        fd_errors = []
        
        for step in range(1000):
            # Get action
            action, _ = model.predict(obs, deterministic=True)
            
            # Step environment
            obs_next, reward, term, trunc, info = env.step(action)
            total_reward += reward
            
            # During dropout, measure velocity estimation error
            if info['dropout_active']:
                true_obs = info['true_obs']
                
                # FD estimate (using frozen obs)
                fd_velocity = (obs - obs_prev) / 0.002
                true_velocity = (true_obs - obs_prev) / 0.002
                fd_error = np.mean(np.abs(fd_velocity - true_velocity))
                fd_errors.append(fd_error)
            
            obs_prev = obs.copy()
            obs = obs_next
            
            if term or trunc:
                results['fd']['length'].append(step)
                break
        
        results['fd']['velocity_error'].extend(fd_errors)
        results['fd']['reward'].append(total_reward)
    
    # Oracle baseline (no dropout)
    env = gym.make('Hopper-v4')
    for ep in range(n_episodes):
        obs, _ = env.reset()
        total_reward = 0
        for step in range(1000):
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, term, trunc, _ = env.step(action)
            total_reward += reward
            if term or trunc:
                results['oracle']['length'].append(step)
                break
        results['oracle']['reward'].append(total_reward)
    env.close()
    
    return results

# =============================================================================
# F3-JEPA FOR HOPPER
# =============================================================================

class F3JEPAHopper(nn.Module):
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
        self.velocity_decoder = nn.Sequential(
            nn.Linear(latent_dim, 64), nn.ReLU(),
            nn.Linear(64, obs_dim)
        )
        self.predictor = nn.Sequential(
            nn.Linear(latent_dim + action_dim, 128), nn.ReLU(),
            nn.Linear(128, 128), nn.ReLU(),
            nn.Linear(128, latent_dim)
        )
        
        # Initialize target encoder
        self.target_encoder.load_state_dict(self.encoder.state_dict())
    
    def encode(self, obs, obs_prev):
        velocity = (obs - obs_prev) / 0.002
        inp = torch.cat([obs, velocity], dim=-1)
        return self.encoder(inp)
    
    def encode_target(self, obs, obs_prev):
        velocity = (obs - obs_prev) / 0.002
        inp = torch.cat([obs, velocity], dim=-1)
        with torch.no_grad():
            return self.target_encoder(inp)
    
    def decode_velocity(self, z):
        return self.velocity_decoder(z)
    
    def predict_latent(self, z, action):
        return self.predictor(torch.cat([z, action], dim=-1))
    
    @torch.no_grad()
    def update_target(self, tau=0.996):
        for tp, ep in zip(self.target_encoder.parameters(), self.encoder.parameters()):
            tp.data.mul_(tau).add_(ep.data, alpha=1 - tau)

def train_f3jepa(model, data, n_epochs=50, lambda_vel=10.0, lambda_pred=0.1):
    """Train F3-JEPA with aggressive velocity loss"""
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    
    batch_size = 256
    n_samples = len(data['obs'])
    
    for epoch in range(n_epochs):
        idx = torch.randperm(n_samples)
        total_loss = 0
        
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
            
            # Losses
            loss_vel = F.mse_loss(v_pred, v_true)
            loss_pred = F.mse_loss(z_pred, z_target.detach())
            
            loss = lambda_vel * loss_vel + lambda_pred * loss_pred
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            model.update_target()
            total_loss += loss.item()
        
        if (epoch + 1) % 10 == 0:
            print(f"Epoch {epoch+1}: loss={total_loss:.2f}")
    
    return model

def generate_data(sac_model, n_episodes=100):
    """Generate training data from SAC oracle"""
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

# =============================================================================
# TEST WITH F3-JEPA OBSERVER
# =============================================================================

def test_f3jepa_observer(sac_model, jepa_model, n_episodes=20, dropout_duration=5, velocity_threshold=0.1):
    """
    Test F3-JEPA as observer during critical dropout.
    
    Key: During dropout, use predictor to maintain latent state,
    then decode velocity from latent.
    """
    env = CriticalDropoutEnv(dropout_duration=dropout_duration, velocity_threshold=velocity_threshold)
    
    results = {'velocity_error': [], 'reward': [], 'length': []}
    
    for ep in range(n_episodes):
        obs, _ = env.reset()
        obs_prev = obs.copy()
        total_reward = 0
        errors = []
        
        # Latent state
        z = None
        
        for step in range(1000):
            # Get action
            action, _ = sac_model.predict(obs, deterministic=True)
            
            # Update latent if not in dropout
            if not env.dropout_countdown > 0:
                obs_tensor = torch.tensor(obs, dtype=torch.float32, device=device).unsqueeze(0)
                obs_prev_tensor = torch.tensor(obs_prev, dtype=torch.float32, device=device).unsqueeze(0)
                with torch.no_grad():
                    z = jepa_model.encode(obs_tensor, obs_prev_tensor)
            
            # Step environment
            obs_next, reward, term, trunc, info = env.step(action)
            total_reward += reward
            
            # During dropout, use predictor
            if info['dropout_active'] and z is not None:
                action_tensor = torch.tensor(action, dtype=torch.float32, device=device).unsqueeze(0)
                with torch.no_grad():
                    z = jepa_model.predict_latent(z, action_tensor)
                
                # Decode velocity estimate
                v_pred = jepa_model.decode_velocity(z).squeeze().detach().cpu().numpy()
                true_v = (info['true_obs'] - obs_prev) / 0.002
                error = np.mean(np.abs(v_pred - true_v))
                errors.append(error)
            
            obs_prev = obs.copy()
            obs = obs_next
            
            if term or trunc:
                results['length'].append(step)
                break
        
        results['velocity_error'].extend(errors)
        results['reward'].append(total_reward)
    
    env.close()
    return results

# =============================================================================
# MAIN
# =============================================================================

if __name__ == '__main__':
    print("="*70)
    print("HOPPER CRITICAL DROPOUT EXPERIMENT")
    print("="*70)
    
    # Get oracle
    sac_model = get_oracle(train_timesteps=100000)
    
    # Evaluate oracle
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
    
    # Test velocity estimation under dropout
    print("\n" + "="*70)
    print("VELOCITY ESTIMATION TEST")
    print("="*70)
    
    results = test_velocity_estimation(
        sac_model, 
        n_episodes=20, 
        dropout_duration=5,
        velocity_threshold=0.1
    )
    
    print(f"\nFD Baseline:")
    print(f"  Mean velocity error: {np.mean(results['fd']['velocity_error']):.4f}")
    print(f"  Mean reward: {np.mean(results['fd']['reward']):.1f} ± {np.std(results['fd']['reward']):.1f}")
    print(f"  Mean length: {np.mean(results['fd']['length']):.0f} steps")
    
    print(f"\nOracle (no dropout):")
    print(f"  Mean reward: {np.mean(results['oracle']['reward']):.1f} ± {np.std(results['oracle']['reward']):.1f}")
    print(f"  Mean length: {np.mean(results['oracle']['length']):.0f} steps")
    
    # Train F3-JEPA
    print("\n" + "="*70)
    print("TRAINING F3-JEPA")
    print("="*70)
    
    data = generate_data(sac_model, n_episodes=100)
    jepa_model = F3JEPAHopper().to(device)
    jepa_model = train_f3jepa(jepa_model, data, n_epochs=50, lambda_vel=10.0, lambda_pred=0.1)
    
    # Test F3-JEPA
    print("\n" + "="*70)
    print("F3-JEPA OBSERVER TEST")
    print("="*70)
    
    jepa_results = test_f3jepa_observer(
        sac_model, jepa_model,
        n_episodes=20,
        dropout_duration=5,
        velocity_threshold=0.1
    )
    
    print(f"\nF3-JEPA:")
    print(f"  Mean velocity error: {np.mean(jepa_results['velocity_error']):.4f}")
    print(f"  Mean reward: {np.mean(jepa_results['reward']):.1f} ± {np.std(jepa_results['reward']):.1f}")
    print(f"  Mean length: {np.mean(jepa_results['length']):.0f} steps")
    
    # Summary
    print("\n" + "="*70)
    print("FINAL RESULTS")
    print("="*70)
    print(f"Oracle (no dropout): {np.mean(results['oracle']['reward']):.1f}")
    print(f"FD Baseline (dropout): {np.mean(results['fd']['reward']):.1f}")
    print(f"F3-JEPA (dropout): {np.mean(jepa_results['reward']):.1f}")
    print()
    print(f"Velocity error - FD: {np.mean(results['fd']['velocity_error']):.4f}")
    print(f"Velocity error - F3-JEPA: {np.mean(jepa_results['velocity_error']):.4f}")
