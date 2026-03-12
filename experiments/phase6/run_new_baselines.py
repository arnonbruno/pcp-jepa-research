#!/usr/bin/env python3
"""
Evaluate new baselines: Simple Moving Average (SMA) and LSTM Observer.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import gymnasium as gym
import os
import sys
import argparse
import warnings
warnings.filterwarnings('ignore')

from stable_baselines3 import SAC
from huggingface_sb3 import load_from_hub

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))
from src.envs.contact_dropout import ContactDropoutEnv
from src.evaluation.stats import summarize_results, save_results
from experiments.phase6.hopper_pano import get_pretrained_oracle, get_env_dt

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class LSTMObserver(nn.Module):
    def __init__(self, obs_dim, act_dim, hidden_dim=128):
        super().__init__()
        self.lstm = nn.LSTM(input_size=obs_dim + act_dim, hidden_size=hidden_dim, batch_first=True)
        self.fc = nn.Linear(hidden_dim, obs_dim)
        
    def forward(self, x, hidden=None):
        out, hidden = self.lstm(x, hidden)
        return self.fc(out), hidden

def train_lstm(model, sac_model, env_id, n_episodes=300, epochs=50):
    print("Training LSTM Observer...")
    if env_id == 'Ant-v4':
        env = gym.make(env_id, use_contact_forces=True)
    else:
        env = gym.make(env_id)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    
    # Generate data sequences
    trajectories = []
    for ep in range(n_episodes):
        obs, _ = env.reset(seed=42+ep)
        seq = []
        for _ in range(1000):
            action, _ = sac_model.predict(obs, deterministic=True)
            obs_next, _, term, trunc, _ = env.step(action)
            seq.append((obs, action, obs_next))
            obs = obs_next
            if term or trunc:
                break
        if len(seq) > 10:
            trajectories.append(seq)
    env.close()

    model.train()
    for epoch in range(epochs):
        total_loss = 0
        for seq in trajectories:
            obs_seq = torch.FloatTensor(np.array([s[0] for s in seq])).to(device)
            act_seq = torch.FloatTensor(np.array([s[1] for s in seq])).to(device)
            obs_next_seq = torch.FloatTensor(np.array([s[2] for s in seq])).to(device)
            
            x = torch.cat([obs_seq, act_seq], dim=-1).unsqueeze(0)
            pred, _ = model(x)
            
            loss = F.mse_loss(pred.squeeze(0), obs_next_seq)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        
        if (epoch+1) % 10 == 0:
            print(f"Epoch {epoch+1}/{epochs}, Loss: {total_loss/len(trajectories):.4f}")
    return model

def eval_sma(sac_model, window_size=5, n_episodes=100, env_id='Hopper-v4', dropout_duration=5, velocity_threshold=0.1, seed=42):
    env = ContactDropoutEnv(env_id, dropout_duration, velocity_threshold)
    rewards = []
    state_errors = []
    dt = get_env_dt(env)
    
    for ep in range(n_episodes):
        obs, _ = env.reset(seed=seed+ep)
        history = [obs.copy() for _ in range(window_size)]
        total_reward = 0
        
        for _ in range(1000):
            action, _ = sac_model.predict(obs, deterministic=True)
            obs_raw, reward, term, trunc, info = env.step(action)
            total_reward += reward
            
            if info['dropout_active']:
                # Calculate average velocity from history
                vels = [(history[i] - history[i-1])/dt for i in range(1, len(history))]
                avg_vel = np.mean(vels, axis=0) if vels else np.zeros_like(obs)
                obs = info['frozen_obs'] + avg_vel * dt * (info['dropout_step'] + 1)
                state_errors.append(np.mean(np.abs(obs - info['true_obs'])))
            else:
                obs = obs_raw
                history.append(obs.copy())
                history.pop(0)
                
            if term or trunc:
                break
        rewards.append(total_reward)
    env.close()
    res = summarize_results('SMA Baseline', np.array(rewards), np.array(state_errors) if state_errors else None)
    return res

def eval_lstm_baseline(sac_model, lstm_model, n_episodes=100, env_id='Hopper-v4', dropout_duration=5, velocity_threshold=0.1, seed=42):
    env = ContactDropoutEnv(env_id, dropout_duration, velocity_threshold)
    rewards = []
    state_errors = []
    lstm_model.eval()
    
    for ep in range(n_episodes):
        obs, _ = env.reset(seed=seed+ep)
        total_reward = 0
        hidden = None
        
        for _ in range(1000):
            action, _ = sac_model.predict(obs, deterministic=True)
            obs_raw, reward, term, trunc, info = env.step(action)
            total_reward += reward
            
            # Update LSTM hidden state or predict
            x = torch.cat([torch.FloatTensor(obs), torch.FloatTensor(action)]).unsqueeze(0).unsqueeze(0).to(device)
            with torch.no_grad():
                pred_obs, next_hidden = lstm_model(x, hidden)
            
            if info['dropout_active']:
                obs = pred_obs.squeeze().cpu().numpy()
                hidden = next_hidden
                state_errors.append(np.mean(np.abs(obs - info['true_obs'])))
            else:
                obs = obs_raw
                hidden = next_hidden
                
            if term or trunc:
                break
        rewards.append(total_reward)
    env.close()
    res = summarize_results('LSTM Baseline', np.array(rewards), np.array(state_errors) if state_errors else None)
    return res

def run_baselines(env_id='Hopper-v4', seed=42, results_dir='../../results/neurips'):
    os.makedirs(results_dir, exist_ok=True)
    sac_model = get_pretrained_oracle(env_id)
    
    if env_id == 'Ant-v4':
        env = gym.make(env_id, use_contact_forces=True)
    else:
        env = gym.make(env_id)
    obs_dim = env.observation_space.shape[0]
    act_dim = env.action_space.shape[0]
    env.close()
    
    lstm_model = LSTMObserver(obs_dim, act_dim).to(device)
    lstm_model = train_lstm(lstm_model, sac_model, env_id, n_episodes=50, epochs=50)
    
    print(f"\nEvaluating on {env_id} with seed {seed}")
    res_sma = eval_sma(sac_model, env_id=env_id, seed=seed)
    res_lstm = eval_lstm_baseline(sac_model, lstm_model, env_id=env_id, seed=seed)
    
    print(f"SMA Reward: {res_sma['reward_mean']:.1f} ± {res_sma['reward_std']:.1f}")
    print(f"LSTM Reward: {res_lstm['reward_mean']:.1f} ± {res_lstm['reward_std']:.1f}")
    
    out = {
        'env_id': env_id,
        'seed': seed,
        'sma': res_sma,
        'lstm': res_lstm
    }
    save_results(out, os.path.join(results_dir, f'new_baselines_{env_id}_seed{seed}.json'))

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--env-id', type=str, default='Hopper-v4')
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--results-dir', type=str, default='../../results/neurips')
    args = parser.parse_args()
    
    run_baselines(args.env_id, args.seed, args.results_dir)
