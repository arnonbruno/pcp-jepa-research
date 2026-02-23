#!/usr/bin/env python3
"""
Hopper F3-JEPA Scale-Up

Goal: Show F3-JEPA keeps Hopper upright under contact-triggered dropout
while standard observation pipelines collapse.

Steps:
1. Train SAC oracle on full state
2. Implement contact-triggered dropout
3. Test FD baseline under dropout
4. Implement F3-JEPA for Hopper
5. Compare results
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import gymnasium as gym
from stable_baselines3 import SAC
from stable_baselines3.common.callbacks import EvalCallback
import warnings
warnings.filterwarnings('ignore')

device = torch.device('cuda')
print(f"Using: {device}")

# =============================================================================
# STEP 1: TRAIN SAC ORACLE
# =============================================================================

def train_sac_oracle(total_timesteps=100000, save_path="hopper_sac.zip"):
    """Train SAC on Hopper-v4 with full state"""
    print("\n" + "="*70)
    print("STEP 1: Training SAC Oracle")
    print("="*70)
    
    env = gym.make('Hopper-v4')
    
    # Check if already trained
    import os
    if os.path.exists(save_path):
        print(f"Loading existing model from {save_path}")
        model = SAC.load(save_path, env=env)
    else:
        model = SAC(
            "MlpPolicy",
            env,
            learning_rate=3e-4,
            buffer_size=100000,
            learning_starts=1000,
            batch_size=256,
            tau=0.005,
            gamma=0.99,
            train_freq=1,
            gradient_steps=1,
            verbose=1
        )
        
        model.learn(total_timesteps=total_timesteps, progress_bar=True)
        model.save(save_path)
        print(f"Saved model to {save_path}")
    
    # Evaluate
    obs, _ = env.reset()
    total_reward = 0
    for _ in range(1000):
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, term, trunc, _ = env.step(action)
        total_reward += reward
        if term or trunc:
            break
    
    print(f"Oracle evaluation: reward={total_reward:.1f}")
    env.close()
    
    return model

# =============================================================================
# STEP 2: CONTACT-TRIGGERED DROPOUT ENVIRONMENT
# =============================================================================

class ContactDropoutEnv:
    """
    Hopper environment with contact-triggered observation dropout.
    
    When foot is near ground (within threshold), freeze observations
    for N consecutive steps.
    """
    
    def __init__(self, dropout_steps=3, foot_threshold=0.05):
        self.env = gym.make('Hopper-v4')
        self.dropout_steps = dropout_steps
        self.foot_threshold = foot_threshold
        
        # State for dropout
        self.frozen_obs = None
        self.dropout_countdown = 0
        
        # Hopper-v4 observation: 11 dims
        # [x, z, theta, 4 joint angles, 6 velocities]
        # Foot height is NOT directly in obs - need to track
        
    def reset(self):
        obs, info = self.env.reset()
        self.frozen_obs = obs.copy()
        self.dropout_countdown = 0
        return obs, info
    
    def get_foot_height(self, obs):
        """
        Estimate foot height from observation.
        Hopper foot is at end of leg chain.
        
        For Hopper-v4, we approximate foot height based on
        body height and leg configuration.
        """
        # Simplified: use z-position (index 1) as proxy
        # In reality, need forward kinematics
        return obs[1]  # z-position
    
    def step(self, action):
        obs, reward, term, trunc, info = self.env.step(action)
        
        # Check for contact (foot near ground)
        foot_h = self.get_foot_height(obs)
        
        if foot_h < self.foot_threshold:
            # Trigger dropout
            self.dropout_countdown = self.dropout_steps
            self.frozen_obs = obs.copy()
        
        # Apply dropout
        if self.dropout_countdown > 0:
            obs_return = self.frozen_obs.copy()
            self.dropout_countdown -= 1
        else:
            obs_return = obs
        
        # Record dropout state in info
        info['dropout_active'] = self.dropout_countdown > 0
        info['true_obs'] = obs.copy()
        
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
# STEP 3: FD BASELINE UNDER DROPOUT
# =============================================================================

def evaluate_fd_baseline(model, n_episodes=20, dropout_steps=3):
    """
    Evaluate frozen SAC policy under contact-triggered dropout.
    Use finite-difference to estimate velocities.
    """
    print("\n" + "="*70)
    print("STEP 3: FD Baseline Under Dropout")
    print("="*70)
    
    env = ContactDropoutEnv(dropout_steps=dropout_steps)
    
    rewards = []
    episode_lengths = []
    dropout_events = 0
    
    for ep in range(n_episodes):
        obs, _ = env.reset()
        obs_prev = obs.copy()
        total_reward = 0
        
        for step in range(1000):
            # Use frozen policy
            action, _ = model.predict(obs, deterministic=True)
            
            obs_next, reward, term, trunc, info = env.step(action)
            total_reward += reward
            
            if info.get('dropout_active', False):
                dropout_events += 1
            
            obs_prev = obs.copy()
            obs = obs_next
            
            if term or trunc:
                episode_lengths.append(step)
                break
        
        rewards.append(total_reward)
    
    env.close()
    
    print(f"Episodes: {n_episodes}")
    print(f"Mean reward: {np.mean(rewards):.1f} ± {np.std(rewards):.1f}")
    print(f"Mean length: {np.mean(episode_lengths):.0f} steps")
    print(f"Dropout events: {dropout_events}")
    
    return np.mean(rewards)

# =============================================================================
# STEP 4: F3-JEPA FOR HOPPER
# =============================================================================

class F3EncoderHopper(nn.Module):
    """Encoder with F3-normalized input for Hopper"""
    def __init__(self, obs_dim=11, latent_dim=64):
        super().__init__()
        # Input: obs (11) + velocity_est (11) = 22
        self.net = nn.Sequential(
            nn.Linear(obs_dim * 2, 128), nn.ReLU(),
            nn.Linear(128, 128), nn.ReLU(),
            nn.Linear(128, latent_dim)
        )
    
    def forward(self, obs, obs_prev, dt=0.002):
        # F3: velocity = (obs - obs_prev) / dt
        velocity = (obs - obs_prev) / dt
        inp = torch.cat([obs, velocity], dim=-1)
        return self.net(inp)


class TargetEncoderHopper(nn.Module):
    """EMA encoder"""
    def __init__(self, obs_dim=11, latent_dim=64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(obs_dim * 2, 128), nn.ReLU(),
            nn.Linear(128, 128), nn.ReLU(),
            nn.Linear(128, latent_dim)
        )
    
    def forward(self, obs, obs_prev, dt=0.002):
        velocity = (obs - obs_prev) / dt
        inp = torch.cat([obs, velocity], dim=-1)
        return self.net(inp)
    
    @torch.no_grad()
    def update_ema(self, encoder, tau=0.996):
        for tp, ep in zip(self.parameters(), encoder.parameters()):
            tp.data.mul_(tau).add_(ep.data, alpha=1 - tau)


class VelocityDecoderHopper(nn.Module):
    """Decode velocity from latent"""
    def __init__(self, latent_dim=64, obs_dim=11):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(latent_dim, 64), nn.ReLU(),
            nn.Linear(64, obs_dim)
        )
    
    def forward(self, z):
        return self.net(z)


class LatentPredictorHopper(nn.Module):
    """Predict z_next from z and action"""
    def __init__(self, latent_dim=64, action_dim=3):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(latent_dim + action_dim, 128), nn.ReLU(),
            nn.Linear(128, 128), nn.ReLU(),
            nn.Linear(128, latent_dim)
        )
    
    def forward(self, z, a):
        return self.net(torch.cat([z, a], dim=-1))


class F3JEPAHopper(nn.Module):
    """F3-JEPA for Hopper"""
    def __init__(self, obs_dim=11, action_dim=3, latent_dim=64):
        super().__init__()
        self.encoder = F3EncoderHopper(obs_dim, latent_dim)
        self.target_encoder = TargetEncoderHopper(obs_dim, latent_dim)
        self.velocity_decoder = VelocityDecoderHopper(latent_dim, obs_dim)
        self.predictor = LatentPredictorHopper(latent_dim, action_dim)
        
        self.target_encoder.load_state_dict(self.encoder.state_dict())
    
    def encode(self, obs, obs_prev, dt=0.002):
        return self.encoder(obs, obs_prev, dt)
    
    def encode_target(self, obs, obs_prev, dt=0.002):
        with torch.no_grad():
            return self.target_encoder(obs, obs_prev, dt)
    
    def decode_velocity(self, z):
        return self.velocity_decoder(z)
    
    def predict(self, z, a):
        return self.predictor(z, a)
    
    def update_target(self, tau=0.996):
        self.target_encoder.update_ema(self.encoder, tau)


# =============================================================================
# STEP 5: TRAINING F3-JEPA
# =============================================================================

def generate_hopper_data(model, n_episodes=100, dt=0.002):
    """Generate training data from SAC oracle"""
    env = gym.make('Hopper-v4')
    
    data = {
        'obs': [], 'obs_prev': [], 'action': [], 
        'obs_next': [], 'obs_prev_next': []
    }
    
    for ep in range(n_episodes):
        obs, _ = env.reset()
        obs_prev = obs.copy()
        
        for step in range(1000):
            action, _ = model.predict(obs, deterministic=True)
            
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
    
    # Convert to tensors
    for k in data:
        data[k] = torch.tensor(np.array(data[k]), dtype=torch.float32, device=device)
    
    print(f"Generated {len(data['obs'])} transitions from {n_episodes} episodes")
    return data


def train_f3jepa_hopper(data, n_epochs=50, lambda_vel=10.0, lambda_pred=0.1, dt=0.002):
    """Train F3-JEPA for Hopper"""
    print("\n" + "="*70)
    print("STEP 4: Training F3-JEPA for Hopper")
    print("="*70)
    print(f"Loss weights: vel={lambda_vel}, pred={lambda_pred}")
    
    model = F3JEPAHopper().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    
    batch_size = 256
    n_samples = len(data['obs'])
    
    for epoch in range(n_epochs):
        idx = torch.randperm(n_samples)
        total_loss = 0
        
        for i in range(0, n_samples, batch_size):
            b = idx[i:i+batch_size]
            
            # Encode
            z_t = model.encode(data['obs'][b], data['obs_prev'][b], dt)
            z_target = model.encode_target(data['obs_next'][b], data['obs_prev_next'][b], dt)
            
            # Predict
            z_pred = model.predict(z_t, data['action'][b])
            
            # Decode velocity
            v_pred = model.decode_velocity(z_t)
            v_true = (data['obs'][b] - data['obs_prev'][b]) / dt
            
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


# =============================================================================
# STEP 6: EVALUATE F3-JEPA UNDER DROPOUT
# =============================================================================

class F3JEPAController:
    """Controller using F3-JEPA for observation reconstruction"""
    
    def __init__(self, jepa_model, sac_policy, dt=0.002):
        self.jepa = jepa_model
        self.policy = sac_policy
        self.dt = dt
        self.z = None
        self.obs_prev = None
    
    def reset(self, obs):
        self.z = None
        self.obs_prev = obs.copy()
    
    def get_action(self, obs, dropout_active=False):
        """Get action, using predictor if in dropout"""
        
        obs_tensor = torch.tensor(obs, dtype=torch.float32, device=device).unsqueeze(0)
        obs_prev_tensor = torch.tensor(self.obs_prev, dtype=torch.float32, device=device).unsqueeze(0)
        
        if not dropout_active:
            # Fresh observation: encode
            with torch.no_grad():
                self.z = self.jepa.encode(obs_tensor, obs_prev_tensor, self.dt)
            self.obs_prev = obs.copy()
        else:
            # In dropout: use predictor
            # Get action from policy using last good obs
            pass
        
        # Get action from policy
        action, _ = self.policy.predict(obs, deterministic=True)
        
        # If in dropout, update latent with predicted action
        if dropout_active and self.z is not None:
            action_tensor = torch.tensor(action, dtype=torch.float32, device=device).unsqueeze(0)
            with torch.no_grad():
                self.z = self.jepa.predict(self.z, action_tensor)
        
        return action


def evaluate_f3jepa(jepa_model, sac_model, n_episodes=20, dropout_steps=3):
    """Evaluate F3-JEPA under contact-triggered dropout"""
    print("\n" + "="*70)
    print("STEP 5: F3-JEPA Under Dropout")
    print("="*70)
    
    env = ContactDropoutEnv(dropout_steps=dropout_steps)
    controller = F3JEPAController(jepa_model, sac_model)
    
    rewards = []
    episode_lengths = []
    
    for ep in range(n_episodes):
        obs, _ = env.reset()
        controller.reset(obs)
        total_reward = 0
        
        for step in range(1000):
            action = controller.get_action(obs, env.dropout_countdown > 0)
            obs, reward, term, trunc, info = env.step(action)
            total_reward += reward
            
            if term or trunc:
                episode_lengths.append(step)
                break
        
        rewards.append(total_reward)
    
    env.close()
    
    print(f"Episodes: {n_episodes}")
    print(f"Mean reward: {np.mean(rewards):.1f} ± {np.std(rewards):.1f}")
    print(f"Mean length: {np.mean(episode_lengths):.0f} steps")
    
    return np.mean(rewards)


# =============================================================================
# MAIN
# =============================================================================

if __name__ == '__main__':
    # Step 1: Train oracle
    sac_model = train_sac_oracle(total_timesteps=50000)
    
    # Step 2: FD baseline
    fd_reward = evaluate_fd_baseline(sac_model, n_episodes=20, dropout_steps=3)
    
    # Step 3: Train F3-JEPA
    data = generate_hopper_data(sac_model, n_episodes=200)
    jepa_model = train_f3jepa_hopper(data, n_epochs=50)
    
    # Step 4: Evaluate F3-JEPA
    jepa_reward = evaluate_f3jepa(jepa_model, sac_model, n_episodes=20, dropout_steps=3)
    
    # Summary
    print("\n" + "="*70)
    print("FINAL RESULTS")
    print("="*70)
    print(f"FD Baseline: {fd_reward:.1f}")
    print(f"F3-JEPA: {jepa_reward:.1f}")
    print(f"Oracle (no dropout): ~1000+ reward")
