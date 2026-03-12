#!/usr/bin/env python3
"""
SOTA Baselines Evaluation: DreamerV3-style RSSM and TD-MPC2-style TOLD
Tests whether modern World Models fail at contact boundaries under sensor dropout.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import gymnasium as gym
import json
import os
import sys
import warnings
import argparse

warnings.filterwarnings('ignore')

from stable_baselines3 import SAC
from huggingface_sb3 import load_from_hub

# Add project root to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))
from src.envs.contact_dropout import ContactDropoutEnv
from src.evaluation.stats import summarize_results, compare_methods, save_results

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
TRAIN_EVAL_SEED_OFFSET = 10_000

# =============================================================================
# DATA GENERATION FOR SEQUENTIAL MODELS
# =============================================================================

def generate_trajectory_data(sac_model, n_episodes=300, max_len=1000, env_id='Hopper-v4', seed=42):
    """Generate full trajectories for training sequence models (World Models)."""
    env = gym.make(env_id)
    trajectories = []
    
    for ep in range(n_episodes):
        obs, _ = env.reset(seed=seed + ep)
        traj = {'obs': [obs.copy()], 'action': []}
        
        for step in range(max_len):
            action, _ = sac_model.predict(obs, deterministic=True)
            obs, _, term, trunc, _ = env.step(action)
            
            traj['action'].append(action.copy())
            traj['obs'].append(obs.copy())
            
            if term or trunc:
                break
                
        # Pad or use variable length
        traj['obs'] = np.array(traj['obs'])
        traj['action'] = np.array(traj['action'])
        trajectories.append(traj)
        
    env.close()
    return trajectories


def relabel_state_error_metric(result):
    """Rename legacy velocity-error keys to the actual state-estimation metric."""
    if 'velocity_error_mean' in result:
        result['state_estimation_error_mean'] = result.pop('velocity_error_mean')
    if 'velocity_error_std' in result:
        result['state_estimation_error_std'] = result.pop('velocity_error_std')
    result['error_metric'] = 'mean_absolute_state_estimation_error'
    return result

def create_batches(trajectories, seq_len=16, batch_size=64):
    """Create fixed-length sequence batches from trajectories."""
    all_obs_seqs = []
    all_act_seqs = []
    
    for traj in trajectories:
        obs = traj['obs']
        acts = traj['action']
        T = len(acts)
        if T < seq_len:
            continue
            
        for t in range(0, T - seq_len + 1, seq_len // 2):
            all_obs_seqs.append(obs[t:t+seq_len+1])  # +1 for target
            all_act_seqs.append(acts[t:t+seq_len])
            
    all_obs_seqs = np.array(all_obs_seqs)
    all_act_seqs = np.array(all_act_seqs)
    
    dataset_size = len(all_obs_seqs)
    indices = np.random.permutation(dataset_size)
    
    batches = []
    for i in range(0, dataset_size, batch_size):
        idx = indices[i:i+batch_size]
        if len(idx) == batch_size:
            b_obs = torch.FloatTensor(all_obs_seqs[idx]).to(device)
            b_act = torch.FloatTensor(all_act_seqs[idx]).to(device)
            batches.append((b_obs, b_act))
            
    return batches

# =============================================================================
# MODEL ARCHITECTURES
# =============================================================================

class DreamerRSSM(nn.Module):
    """
    Simplified Recurrent State Space Model (DreamerV3 style).
    Deterministic RNN + Stochastic latent state.
    """
    def __init__(self, obs_dim, act_dim, hidden_dim=256, latent_dim=32):
        super().__init__()
        self.latent_dim = latent_dim
        self.hidden_dim = hidden_dim
        
        # Encoder: obs -> hidden
        self.encoder = nn.Sequential(
            nn.Linear(obs_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ELU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ELU()
        )
        
        # Prior: h_t -> z_t (hallucination)
        self.prior_net = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ELU(),
            nn.Linear(hidden_dim, latent_dim * 2) # mean, std
        )
        
        # Posterior: h_t, encoder(x_t) -> z_t (training)
        self.posterior_net = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ELU(),
            nn.Linear(hidden_dim, latent_dim * 2)
        )
        
        # RNN (Deterministic state): h_t = RNN(h_{t-1}, z_{t-1}, a_{t-1})
        self.rnn = nn.GRUCell(latent_dim + act_dim, hidden_dim)
        
        # Decoder: h_t, z_t -> x_t
        self.decoder = nn.Sequential(
            nn.Linear(hidden_dim + latent_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ELU(),
            nn.Linear(hidden_dim, obs_dim)
        )

    def get_z(self, stats):
        mean, log_std = torch.chunk(stats, 2, dim=-1)
        std = torch.exp(log_std.clamp(-5, 2))
        return mean + std * torch.randn_like(std)

    def forward(self, obs_seq, act_seq):
        """
        obs_seq: (B, T+1, obs_dim)
        act_seq: (B, T, act_dim)
        Returns KL divergence and reconstruction loss
        """
        B, T, _ = act_seq.shape
        
        h = torch.zeros(B, self.hidden_dim, device=obs_seq.device)
        z = torch.zeros(B, self.latent_dim, device=obs_seq.device)
        
        kl_loss = 0
        recon_loss = 0
        
        for t in range(T):
            # Deterministic step
            rnn_in = torch.cat([z, act_seq[:, t]], dim=-1)
            h = self.rnn(rnn_in, h)
            
            # Encode next obs
            obs_enc = self.encoder(obs_seq[:, t+1])
            
            # Prior and Posterior
            prior_stats = self.prior_net(h)
            post_stats = self.posterior_net(torch.cat([h, obs_enc], dim=-1))
            
            prior_mean, prior_log_std = torch.chunk(prior_stats, 2, dim=-1)
            post_mean, post_log_std = torch.chunk(post_stats, 2, dim=-1)
            prior_std = torch.exp(prior_log_std.clamp(-5, 2))
            post_std = torch.exp(post_log_std.clamp(-5, 2))
            
            # Sample z from posterior for training
            z = self.get_z(post_stats)
            
            # Decode
            obs_pred = self.decoder(torch.cat([h, z], dim=-1))
            
            # Losses
            recon_loss += F.mse_loss(obs_pred, obs_seq[:, t+1])
            
            # KL Divergence: KL(q(z|x) || p(z))
            # Simplified Gaussian KL
            var_ratio = (post_std / prior_std).pow(2)
            t1 = ((prior_mean - post_mean) / prior_std).pow(2)
            kl = 0.5 * (var_ratio + t1 - 1 - 2 * (post_log_std - prior_log_std))
            kl_loss += kl.sum(dim=-1).mean()
            
        return recon_loss / T, kl_loss / T
        
    def hallucinate(self, h, z, action):
        """Step forward without observation (dropout)"""
        rnn_in = torch.cat([z, action], dim=-1)
        h_next = self.rnn(rnn_in, h)
        prior_stats = self.prior_net(h_next)
        z_next = self.get_z(prior_stats)
        obs_pred = self.decoder(torch.cat([h_next, z_next], dim=-1))
        return h_next, z_next, obs_pred

    def encode_step(self, h, z, action, obs):
        """Step forward with observation"""
        rnn_in = torch.cat([z, action], dim=-1)
        h_next = self.rnn(rnn_in, h)
        obs_enc = self.encoder(obs)
        post_stats = self.posterior_net(torch.cat([h_next, obs_enc], dim=-1))
        z_next = self.get_z(post_stats)
        return h_next, z_next


class TDMPCTOLD(nn.Module):
    """
    Simplified Task-Oriented Latent Dynamics (TD-MPC2 style).
    Joint-embedding predictive architecture.
    """
    def __init__(self, obs_dim, act_dim, latent_dim=128):
        super().__init__()
        self.latent_dim = latent_dim
        
        # Representation: x -> z
        self.encoder = nn.Sequential(
            nn.Linear(obs_dim, 256),
            nn.LayerNorm(256),
            nn.Mish(),
            nn.Linear(256, latent_dim)
        )
        
        # Dynamics: z, a -> z_next
        self.dynamics = nn.Sequential(
            nn.Linear(latent_dim + act_dim, 256),
            nn.LayerNorm(256),
            nn.Mish(),
            nn.Linear(256, latent_dim)
        )
        
        # Observation predictor: z -> x_hat
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, 256),
            nn.LayerNorm(256),
            nn.Mish(),
            nn.Linear(256, obs_dim)
        )
        
    def forward(self, obs_seq, act_seq):
        """
        obs_seq: (B, T+1, obs_dim)
        act_seq: (B, T, act_dim)
        Returns predictive loss (latent and observation)
        """
        B, T, _ = act_seq.shape
        
        # Encode initial state
        z = self.encoder(obs_seq[:, 0])
        
        latent_loss = 0
        recon_loss = 0
        
        for t in range(T):
            z = self.dynamics(torch.cat([z, act_seq[:, t]], dim=-1))
            
            with torch.no_grad():
                z_target = self.encoder(obs_seq[:, t+1])
                
            obs_pred = self.decoder(z)
            
            latent_loss += F.mse_loss(z, z_target)
            recon_loss += F.mse_loss(obs_pred, obs_seq[:, t+1])
            
        return recon_loss / T, latent_loss / T

    def step(self, z, action):
        """Predict next latent and observation"""
        z_next = self.dynamics(torch.cat([z, action], dim=-1))
        obs_pred = self.decoder(z_next)
        return z_next, obs_pred

# =============================================================================
# TRAINING FUNCTIONS
# =============================================================================

def train_world_models(obs_dim, act_dim, trajectories, epochs=100):
    print("\nTraining DreamerV3-style RSSM and TD-MPC2-style TOLD...")
    
    rssm = DreamerRSSM(obs_dim, act_dim).to(device)
    told = TDMPCTOLD(obs_dim, act_dim).to(device)
    
    opt_rssm = torch.optim.Adam(rssm.parameters(), lr=1e-3)
    opt_told = torch.optim.Adam(told.parameters(), lr=1e-3)
    
    batches = create_batches(trajectories)
    
    for epoch in range(epochs):
        rssm_loss_epoch = 0
        told_loss_epoch = 0
        
        for obs_seq, act_seq in batches:
            # Train RSSM
            opt_rssm.zero_grad()
            recon_r, kl_r = rssm(obs_seq, act_seq)
            loss_r = recon_r + 0.1 * kl_r # KL weight
            loss_r.backward()
            opt_rssm.step()
            rssm_loss_epoch += loss_r.item()
            
            # Train TOLD
            opt_told.zero_grad()
            recon_t, lat_t = told(obs_seq, act_seq)
            loss_t = recon_t + 0.5 * lat_t
            loss_t.backward()
            opt_told.step()
            told_loss_epoch += loss_t.item()
            
        if (epoch+1) % 20 == 0:
            print(f"  Epoch {epoch+1}/{epochs} | RSSM Loss: {rssm_loss_epoch/len(batches):.4f} | TOLD Loss: {told_loss_epoch/len(batches):.4f}")
            
    return rssm, told

# =============================================================================
# EVALUATION
# =============================================================================

def get_pretrained_oracle(env_id='Hopper-v4'):
    """Load pretrained expert from RL Baselines3 Zoo on Hugging Face."""
    env_to_hf = {
        'Hopper-v4': ('sb3/sac-Hopper-v3', 'sac-Hopper-v3.zip'),
    }
    repo_id, filename = env_to_hf[env_id]
    checkpoint = load_from_hub(repo_id=repo_id, filename=filename)
    env = gym.make(env_id)
    model = SAC.load(checkpoint, env=env)
    env.close()
    return model

def eval_oracle(sac_model, n_episodes=100, env_id='Hopper-v4', seed=42):
    env = gym.make(env_id)
    rewards = []
    for ep in range(n_episodes):
        obs, _ = env.reset(seed=seed + ep)
        total_reward = 0.0
        for _ in range(1000):
            action, _ = sac_model.predict(obs, deterministic=True)
            obs, reward, term, trunc, _ = env.step(action)
            total_reward += reward
            if term or trunc:
                break
        rewards.append(total_reward)
    env.close()
    return summarize_results('Oracle (no dropout)', np.array(rewards))

def eval_frozen_baseline(sac_model, n_episodes=100, dropout_duration=5, velocity_threshold=0.1, env_id='Hopper-v4', seed=42):
    env = ContactDropoutEnv(env_id, dropout_duration, velocity_threshold)
    rewards, state_errors = [], []
    for ep in range(n_episodes):
        obs, _ = env.reset(seed=seed + ep)
        total_reward = 0.0
        for _ in range(1000):
            action, _ = sac_model.predict(obs, deterministic=True)
            obs, reward, term, trunc, info = env.step(action)
            total_reward += reward
            if info['dropout_active']:
                state_errors.append(np.mean(np.abs(info['true_obs'] - info['frozen_obs'])))
            if term or trunc:
                break
        rewards.append(total_reward)
    res = summarize_results('Frozen Baseline', np.array(rewards), np.array(state_errors) if state_errors else None)
    res = relabel_state_error_metric(res)
    env.close()
    return res

def eval_rssm(sac_model, rssm, n_episodes=100, dropout_duration=5, velocity_threshold=0.1, env_id='Hopper-v4', seed=42):
    env = ContactDropoutEnv(env_id, dropout_duration, velocity_threshold)
    rewards, state_errors = [], []
    
    rssm.eval()
    
    for ep in range(n_episodes):
        obs, _ = env.reset(seed=seed + ep)
        total_reward = 0.0
        
        # Init latent state
        h = torch.zeros(1, rssm.hidden_dim, device=device)
        z = torch.zeros(1, rssm.latent_dim, device=device)
        
        for step in range(1000):
            action, _ = sac_model.predict(obs, deterministic=True)
            act_t = torch.FloatTensor(action).unsqueeze(0).to(device)
            obs_t = torch.FloatTensor(obs).unsqueeze(0).to(device)
            
            # Environment step
            obs_raw, reward, term, trunc, info = env.step(action)
            total_reward += reward
            
            if info['dropout_active']:
                # Hallucinate next state
                with torch.no_grad():
                    h, z, obs_pred = rssm.hallucinate(h, z, act_t)
                obs = obs_pred.squeeze(0).cpu().numpy()
                state_errors.append(np.mean(np.abs(obs - info['true_obs'])))
            else:
                # Update latent with true observation
                with torch.no_grad():
                    h, z = rssm.encode_step(h, z, act_t, torch.FloatTensor(obs_raw).unsqueeze(0).to(device))
                obs = obs_raw
                
            if term or trunc:
                break
        rewards.append(total_reward)
        
    res = summarize_results('Simplified RSSM', np.array(rewards), np.array(state_errors) if state_errors else None)
    res = relabel_state_error_metric(res)
    env.close()
    return res

def eval_told(sac_model, told, n_episodes=100, dropout_duration=5, velocity_threshold=0.1, env_id='Hopper-v4', seed=42):
    env = ContactDropoutEnv(env_id, dropout_duration, velocity_threshold)
    rewards, state_errors = [], []
    
    told.eval()
    
    for ep in range(n_episodes):
        obs, _ = env.reset(seed=seed + ep)
        total_reward = 0.0
        
        # Init latent state
        with torch.no_grad():
            z = told.encoder(torch.FloatTensor(obs).unsqueeze(0).to(device))
            
        for step in range(1000):
            action, _ = sac_model.predict(obs, deterministic=True)
            act_t = torch.FloatTensor(action).unsqueeze(0).to(device)
            
            obs_raw, reward, term, trunc, info = env.step(action)
            total_reward += reward
            
            if info['dropout_active']:
                # Predict next observation and update z
                with torch.no_grad():
                    z, obs_pred = told.step(z, act_t)
                obs = obs_pred.squeeze(0).cpu().numpy()
                state_errors.append(np.mean(np.abs(obs - info['true_obs'])))
            else:
                # Re-encode true observation
                with torch.no_grad():
                    z = told.encoder(torch.FloatTensor(obs_raw).unsqueeze(0).to(device))
                obs = obs_raw
                
            if term or trunc:
                break
        rewards.append(total_reward)
        
    res = summarize_results('Simplified TOLD', np.array(rewards), np.array(state_errors) if state_errors else None)
    res = relabel_state_error_metric(res)
    env.close()
    return res

# =============================================================================
# MAIN
# =============================================================================

def run_experiment(n_episodes=100, dropout_duration=5, velocity_threshold=0.1, 
                   results_dir=None, seed=42, train_episodes=300, train_epochs=100):
    
    if results_dir is None:
        results_dir = os.path.join(os.path.dirname(__file__), '..', '..', 'results', 'phase6')
    
    np.random.seed(seed)
    torch.manual_seed(seed)
    train_seed = seed
    eval_seed = seed + TRAIN_EVAL_SEED_OFFSET
    
    print("=" * 70)
    print("SIMPLIFIED WORLD MODELS EVALUATION: Contact-Triggered Dropout")
    print("=" * 70)
    print(f"Train seed: {train_seed}")
    print(f"Eval seed:  {eval_seed}")
    
    print("\n[1/4] Loading pretrained Oracle...")
    sac_model = get_pretrained_oracle('Hopper-v4')
    
    print(f"\n[2/4] Generating training data for World Models ({train_episodes} episodes)...")
    trajectories = generate_trajectory_data(sac_model, n_episodes=train_episodes, seed=train_seed)
    env = gym.make('Hopper-v4')
    obs_dim = env.observation_space.shape[0]
    act_dim = env.action_space.shape[0]
    env.close()
    
    rssm, told = train_world_models(obs_dim, act_dim, trajectories, epochs=train_epochs)
    
    print(f"\n[3/4] Evaluating Models ({n_episodes} episodes)...")
    
    oracle_res = eval_oracle(sac_model, n_episodes, seed=eval_seed)
    print(f"  Oracle: {oracle_res['reward_mean']:.1f}")
    
    frozen_res = eval_frozen_baseline(sac_model, n_episodes, dropout_duration, velocity_threshold, seed=eval_seed)
    print(f"  Frozen: {frozen_res['reward_mean']:.1f}")
    
    rssm_res = eval_rssm(sac_model, rssm, n_episodes, dropout_duration, velocity_threshold, seed=eval_seed)
    print(f"  Simplified RSSM: {rssm_res['reward_mean']:.1f}")
    
    told_res = eval_told(sac_model, told, n_episodes, dropout_duration, velocity_threshold, seed=eval_seed)
    print(f"  Simplified TOLD: {told_res['reward_mean']:.1f}")
    
    print("\n" + "=" * 70)
    print("STATISTICAL TESTS (Welch's t-test)")
    print("=" * 70)
    
    comparisons = {}
    
    comp_rssm_frozen = compare_methods(rssm_res, frozen_res, "Simplified RSSM vs Frozen")
    comparisons['rssm_vs_frozen'] = comp_rssm_frozen
    print(f"  RSSM vs Frozen: Δ={comp_rssm_frozen['improvement_absolute']:+.1f} "
          f"({comp_rssm_frozen['improvement_pct']:+.1f}%), "
          f"p={comp_rssm_frozen['p_value']:.4f}")
          
    comp_told_frozen = compare_methods(told_res, frozen_res, "Simplified TOLD vs Frozen")
    comparisons['told_vs_frozen'] = comp_told_frozen
    print(f"  TOLD vs Frozen:   Δ={comp_told_frozen['improvement_absolute']:+.1f} "
          f"({comp_told_frozen['improvement_pct']:+.1f}%), "
          f"p={comp_told_frozen['p_value']:.4f}")

    print("\n" + "=" * 70)
    print("RESULTS SUMMARY")
    print("=" * 70)
    print(f"{'Method':<30} {'Reward':>12} {'95% CI':>20} {'State Err':>10}")
    print("-" * 72)
    for r in [oracle_res, frozen_res, rssm_res, told_res]:
        vel = (
            f"{r.get('state_estimation_error_mean', 0):.1f}"
            if 'state_estimation_error_mean' in r else '-'
        )
        print(f"{r['method']:<30} {r['reward_mean']:>8.1f} ± {r['reward_std']:>4.1f}"
              f"  [{r['reward_ci_95_lower']:>6.1f}, {r['reward_ci_95_upper']:>6.1f}]"
              f"  {vel:>10}")
              
    os.makedirs(results_dir, exist_ok=True)
    out_file = os.path.join(results_dir, 'sota_baselines_results.json')
    
    all_results = {
        'experiment': 'sota_baselines',
        'env_id': 'Hopper-v4',
        'n_episodes': n_episodes,
        'dropout_duration': dropout_duration,
        'seed': seed,
        'train_seed': train_seed,
        'eval_seed': eval_seed,
        'methods': {
            'oracle': oracle_res,
            'frozen': frozen_res,
            'rssm': rssm_res,
            'told': told_res
        },
        'comparisons': comparisons
    }
    
    save_results(all_results, out_file)
    print(f"\nResults saved to {out_file}")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='SOTA World Models on Contact Dropout')
    parser.add_argument('--n-episodes', type=int, default=100)
    parser.add_argument('--dropout-duration', type=int, default=5)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--train-episodes', type=int, default=300)
    parser.add_argument('--train-epochs', type=int, default=100)
    parser.add_argument('--results-dir', type=str, default=None)
    args = parser.parse_args()
    
    run_experiment(
        n_episodes=args.n_episodes,
        dropout_duration=args.dropout_duration,
        results_dir=args.results_dir,
        seed=args.seed,
        train_episodes=args.train_episodes,
        train_epochs=args.train_epochs
    )
