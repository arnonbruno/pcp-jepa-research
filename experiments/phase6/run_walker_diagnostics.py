#!/usr/bin/env python3
"""
Diagnostic experiments for Walker2d: Why does PANO fail?
Tests:
1. Standard PANO (baseline for failure)
2. Larger Network (depth=5, hidden=256)
3. More Training Data (1000 episodes instead of 300)
"""

import torch
import numpy as np
import gymnasium as gym
import os
import sys
import argparse

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))
from src.models.pano import PANOVelocityPredictor
from src.utils.data import generate_pano_data
from src.utils.training import train_pano
from experiments.phase6.hopper_pano import get_pretrained_oracle, eval_pano
from src.evaluation.stats import save_results

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def run_diagnostics(seed=42, results_dir='../../results/neurips'):
    os.makedirs(results_dir, exist_ok=True)
    env_id = 'Walker2d-v4'
    print(f"Running Walker2d Diagnostics (seed {seed})...")
    
    sac_model = get_pretrained_oracle(env_id)
    if env_id == 'Ant-v4':
        env = gym.make(env_id, use_contact_forces=True)
    else:
        env = gym.make(env_id)
    obs_dim = env.observation_space.shape[0]
    act_dim = env.action_space.shape[0]
    env.close()

    # Base data (300 eps)
    print("\nGenerating standard data (300 eps)...")
    data_standard = generate_pano_data(sac_model, n_episodes=300, env_id=env_id, seed=seed)
    
    # Large data (1000 eps)
    print("\nGenerating large data (1000 eps)...")
    data_large = generate_pano_data(sac_model, n_episodes=1000, env_id=env_id, seed=seed)

    print("\n1. Training Standard PANO...")
    pano_std = PANOVelocityPredictor(obs_dim, act_dim).to(device)
    pano_std = train_pano(pano_std, data_standard, n_epochs=50)
    res_std = eval_pano(sac_model, pano_std, n_episodes=50, env_id=env_id, seed=seed)
    print(f"Standard PANO Reward: {res_std['reward_mean']:.1f}")

    print("\n2. Training Large Network PANO...")
    pano_large_net = PANOVelocityPredictor(obs_dim, act_dim, hidden_dim=256, depth=5).to(device)
    pano_large_net = train_pano(pano_large_net, data_standard, n_epochs=50)
    res_large_net = eval_pano(sac_model, pano_large_net, n_episodes=50, env_id=env_id, seed=seed)
    print(f"Large Net PANO Reward: {res_large_net['reward_mean']:.1f}")

    print("\n3. Training PANO on Large Data...")
    pano_large_data = PANOVelocityPredictor(obs_dim, act_dim).to(device)
    pano_large_data = train_pano(pano_large_data, data_large, n_epochs=50)
    res_large_data = eval_pano(sac_model, pano_large_data, n_episodes=50, env_id=env_id, seed=seed)
    print(f"Large Data PANO Reward: {res_large_data['reward_mean']:.1f}")

    out = {
        'env_id': env_id,
        'seed': seed,
        'standard': res_std,
        'large_net': res_large_net,
        'large_data': res_large_data
    }
    save_results(out, os.path.join(results_dir, f'walker_diagnostics_seed{seed}.json'))

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--results-dir', type=str, default='../../results/neurips')
    args = parser.parse_args()
    
    run_diagnostics(args.seed, args.results_dir)
