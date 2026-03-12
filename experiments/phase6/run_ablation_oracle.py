#!/usr/bin/env python3
"""
Ablation Study: Why does PANO beat Oracle?
Tests whether smoothing Oracle's perfect observations improves performance,
explaining the surprising result that PANO (which effectively smooths via integration)
can outperform the true Oracle.
"""

import numpy as np
import gymnasium as gym
import os
import sys
import argparse
from stable_baselines3 import SAC
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))
from experiments.phase6.hopper_pano import get_pretrained_oracle
from src.evaluation.stats import summarize_results, save_results

def eval_smoothed_oracle(sac_model, filter_type='sma', window_size=3, alpha=0.5, n_episodes=100, env_id='Hopper-v4', seed=42):
    if env_id == 'Ant-v4':
        env = gym.make(env_id, use_contact_forces=True)
    else:
        env = gym.make(env_id)
    rewards = []
    
    for ep in range(n_episodes):
        obs, _ = env.reset(seed=seed+ep)
        total_reward = 0
        
        if filter_type == 'sma':
            history = [obs.copy() for _ in range(window_size)]
        else: # ema
            ema_obs = obs.copy()
            
        for _ in range(1000):
            # Apply filter to get smoothed observation for the policy
            if filter_type == 'sma':
                smoothed_obs = np.mean(history, axis=0)
            else: # ema
                ema_obs = alpha * obs + (1 - alpha) * ema_obs
                smoothed_obs = ema_obs
                
            action, _ = sac_model.predict(smoothed_obs, deterministic=True)
            obs_next, reward, term, trunc, _ = env.step(action)
            total_reward += reward
            
            obs = obs_next
            if filter_type == 'sma':
                history.append(obs.copy())
                history.pop(0)
                
            if term or trunc:
                break
        rewards.append(total_reward)
    env.close()
    
    res = summarize_results(f'Smoothed Oracle ({filter_type})', np.array(rewards))
    return res

def run_ablation(env_id='Hopper-v4', seed=42, results_dir='../../results/neurips'):
    os.makedirs(results_dir, exist_ok=True)
    print(f"Running Oracle Smoothing Ablation on {env_id}...")
    sac_model = get_pretrained_oracle(env_id)
    
    # Standard Oracle
    if env_id == 'Ant-v4':
        env = gym.make(env_id, use_contact_forces=True)
    else:
        env = gym.make(env_id)
    rewards = []
    for ep in range(50): # 50 episodes for speed
        obs, _ = env.reset(seed=seed+ep)
        total_reward = 0
        for _ in range(1000):
            action, _ = sac_model.predict(obs, deterministic=True)
            obs, reward, term, trunc, _ = env.step(action)
            total_reward += reward
            if term or trunc: break
        rewards.append(total_reward)
    env.close()
    res_oracle = summarize_results('Oracle', np.array(rewards))
    
    res_sma_3 = eval_smoothed_oracle(sac_model, 'sma', window_size=3, n_episodes=50, env_id=env_id, seed=seed)
    res_sma_5 = eval_smoothed_oracle(sac_model, 'sma', window_size=5, n_episodes=50, env_id=env_id, seed=seed)
    res_ema_05 = eval_smoothed_oracle(sac_model, 'ema', alpha=0.5, n_episodes=50, env_id=env_id, seed=seed)
    res_ema_02 = eval_smoothed_oracle(sac_model, 'ema', alpha=0.2, n_episodes=50, env_id=env_id, seed=seed)
    
    print(f"Oracle: {res_oracle['reward_mean']:.1f}")
    print(f"SMA(3): {res_sma_3['reward_mean']:.1f}")
    print(f"SMA(5): {res_sma_5['reward_mean']:.1f}")
    print(f"EMA(0.5): {res_ema_05['reward_mean']:.1f}")
    print(f"EMA(0.2): {res_ema_02['reward_mean']:.1f}")
    
    out = {
        'env_id': env_id,
        'seed': seed,
        'oracle': res_oracle,
        'sma_3': res_sma_3,
        'sma_5': res_sma_5,
        'ema_0.5': res_ema_05,
        'ema_0.2': res_ema_02
    }
    save_results(out, os.path.join(results_dir, f'ablation_oracle_{env_id}_seed{seed}.json'))

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--env-id', type=str, default='Hopper-v4')
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--results-dir', type=str, default='../../results/neurips')
    args = parser.parse_args()
    
    run_ablation(args.env_id, args.seed, args.results_dir)
