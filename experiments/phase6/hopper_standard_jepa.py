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
from src.models.jepa import StandardLatentJEPA
from src.envs.contact_dropout import ContactDropoutEnv, CriticalDropoutEnv
from src.utils.data import generate_jepa_data_episodes as generate_data
from src.utils.training import train_standard_jepa, train_standard_jepa_multistep

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


# =============================================================================
# Standard Latent JEPA - RESIDUAL LATENT DYNAMICS
# =============================================================================




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
    jepa_model = StandardLatentJEPA(latent_dim=64).to(device)
    
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