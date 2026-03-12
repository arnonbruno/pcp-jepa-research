#!/usr/bin/env python3
"""
PANO (Physics-Anchored Neural Observer) vs Baselines on MuJoCo Hopper

Evaluates state estimation under contact-triggered sensor dropout:
  1. Oracle (no dropout) — upper bound
  2. Frozen Baseline (dropout, no estimation) — lower bound
  3. EKF Baseline (Extended Kalman Filter velocity estimation)
  4. PANO (learned velocity prediction + Euler integration)

All results are saved as structured JSON with statistical tests.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import gymnasium as gym
from stable_baselines3 import SAC
import json
import os
import sys
import warnings
import argparse
warnings.filterwarnings('ignore')

# Hugging Face pretrained models
from huggingface_sb3 import load_from_hub

# Add project root to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))
from src.models.pano import PANOVelocityPredictor
from src.models.ekf import EKFEstimator
from src.envs.contact_dropout import ContactDropoutEnv
from src.utils.data import generate_pano_data as generate_training_data
from src.utils.training import train_pano
from src.models.pano import PANOVelocityPredictor
from src.models.ekf import EKFEstimator
from src.envs.contact_dropout import ContactDropoutEnv
from src.utils.data import generate_pano_data as generate_training_data
from src.utils.training import train_pano
from src.evaluation.stats import summarize_results, compare_methods, save_results, welch_ttest

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# =============================================================================
# ORACLE - PRETRAINED EXPERT FROM HUGGING FACE
# =============================================================================

def get_pretrained_oracle(env_id='Hopper-v4'):
    """Load pretrained expert from RL Baselines3 Zoo on Hugging Face.
    
    Expert scores: Hopper >3000, Walker2d >4000, HalfCheetah >8000
    """
    # Map v4 envs to v3 model names (same state/action spaces)
    env_to_hf = {
        'Hopper-v4': ('sb3/sac-Hopper-v3', 'sac-Hopper-v3.zip'),
        'Walker2d-v4': ('sb3/sac-Walker2d-v3', 'sac-Walker2d-v3.zip'),
        'HalfCheetah-v4': ('sb3/sac-HalfCheetah-v3', 'sac-HalfCheetah-v3.zip'),
    }
    
    if env_id not in env_to_hf:
        raise ValueError(f"No pretrained model for {env_id}")
    
    repo_id, filename = env_to_hf[env_id]
    
    print(f"Downloading pretrained expert from HuggingFace: {repo_id}")
    checkpoint = load_from_hub(repo_id=repo_id, filename=filename)
    
    env = gym.make(env_id)
    model = SAC.load(checkpoint, env=env)
    print(f"✓ Expert loaded successfully")
    
    # Quick test
    obs, _ = env.reset()
    total_reward = 0
    for _ in range(1000):
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, term, trunc, _ = env.step(action)
        total_reward += reward
        if term or trunc:
            break
    env.close()
    print(f"  Expert performance: {total_reward:.0f} (expected >2000 for Hopper)")
    
    return model


def get_oracle(model_path='hopper_sac.zip'):
    """Legacy: load local oracle if exists, otherwise use pretrained."""
    if os.path.exists(model_path):
        env = gym.make('Hopper-v4')
        return SAC.load(model_path, env=env)
    return get_pretrained_oracle('Hopper-v4')


def train_oracle(model_path='hopper_sac.zip', total_timesteps=1_000_000, seed=42):
    """Train a strong SAC oracle (1M steps for proper convergence)."""
    print(f"Training SAC oracle for {total_timesteps} steps...")
    env = gym.make('Hopper-v4')
    model = SAC('MlpPolicy', env, learning_rate=3e-4, buffer_size=300_000,
                learning_starts=10_000, batch_size=256, tau=0.005,
                gamma=0.99, verbose=0, seed=seed)
    model.learn(total_timesteps=total_timesteps, progress_bar=True)
    model.save(model_path)
    print(f"Oracle saved to {model_path}")
    env.close()
    return model

# =============================================================================
# CONTACT-TRIGGERED DROPOUT ENVIRONMENT
# =============================================================================


# =============================================================================
# PANO: PHYSICS-ANCHORED NEURAL OBSERVER
# =============================================================================


# =============================================================================
# EKF BASELINE (Extended Kalman Filter)
# =============================================================================


# =============================================================================
# DATA GENERATION AND TRAINING
# =============================================================================




# =============================================================================
# EVALUATION METHODS
# =============================================================================

def eval_oracle(sac_model, n_episodes=100, env_id='Hopper-v4', seed=42):
    """Oracle: full observation, no dropout."""
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


def eval_frozen_baseline(sac_model, n_episodes=100, dropout_duration=5,
                          velocity_threshold=0.1, env_id='Hopper-v4', seed=42):
    """Frozen baseline: during dropout, observation is frozen (no estimation)."""
    env = ContactDropoutEnv(env_id, dropout_duration, velocity_threshold)
    dt = env.env.unwrapped.dt if hasattr(env.env.unwrapped, 'dt') else 0.002
    rewards, vel_errors = [], []

    for ep in range(n_episodes):
        obs, _ = env.reset(seed=seed + ep)
        total_reward = 0.0
        for _ in range(1000):
            action, _ = sac_model.predict(obs, deterministic=True)
            obs, reward, term, trunc, info = env.step(action)
            total_reward += reward
            if info['dropout_active']:
                true_v = info['true_obs'] - info['frozen_obs']
                vel_errors.append(np.mean(np.abs(true_v)))
            if term or trunc:
                break
        rewards.append(total_reward)

    env.close()
    return summarize_results('Frozen Baseline (dropout)', np.array(rewards),
                              np.array(vel_errors) if vel_errors else None)


def eval_ekf(sac_model, n_episodes=100, dropout_duration=5,
              velocity_threshold=0.1, env_id='Hopper-v4', seed=42):
    """EKF baseline: Extended Kalman Filter for state estimation under dropout."""
    env = ContactDropoutEnv(env_id, dropout_duration, velocity_threshold)
    obs_dim = env.observation_space.shape[0]
    dt = env.env.unwrapped.dt if hasattr(env.env.unwrapped, 'dt') else 0.002
    rewards, vel_errors = [], []

    for ep in range(n_episodes):
        obs, _ = env.reset(seed=seed + ep)
        ekf = EKFEstimator(obs_dim=obs_dim, dt=dt)
        ekf.reset(obs)
        total_reward = 0.0

        for step in range(1000):
            action, _ = sac_model.predict(obs, deterministic=True)
            obs_raw, reward, term, trunc, info = env.step(action)
            total_reward += reward

            ekf.predict()

            if info['dropout_active']:
                # No measurement update; use EKF prediction
                obs = ekf.get_obs_estimate()
                vel_errors.append(np.mean(np.abs(obs - info['true_obs'])))
            else:
                ekf.update(obs_raw)
                obs = obs_raw

            if term or trunc:
                break
        rewards.append(total_reward)

    env.close()
    return summarize_results('EKF Baseline (dropout)', np.array(rewards),
                              np.array(vel_errors) if vel_errors else None)


def eval_pano(sac_model, velocity_model, n_episodes=100, dropout_duration=5,
               velocity_threshold=0.1, history_len=5, env_id='Hopper-v4', seed=42):
    """PANO: Physics-Anchored Neural Observer with Euler integration."""
    env = ContactDropoutEnv(env_id, dropout_duration, velocity_threshold)
    action_dim = env.action_space.shape[0]
    dt = env.env.unwrapped.dt if hasattr(env.env.unwrapped, 'dt') else 0.002
    rewards, vel_errors = [], []

    for ep in range(n_episodes):
        obs, _ = env.reset(seed=seed + ep)
        action_history = [np.zeros(action_dim) for _ in range(history_len)]
        total_reward = 0.0

        for step in range(1000):
            action, _ = sac_model.predict(obs, deterministic=True)
            action_history.pop(0)
            action_history.append(action.copy())

            obs_raw, reward, term, trunc, info = env.step(action)
            total_reward += reward

            if info['dropout_active']:
                # PANO: predict velocity, integrate
                obs_t = torch.tensor(obs, dtype=torch.float32, device=device).unsqueeze(0)
                ah_t = torch.tensor(np.array(action_history), dtype=torch.float32, device=device).unsqueeze(0)
                with torch.no_grad():
                    v_pred = velocity_model(obs_t, ah_t).squeeze().cpu().numpy()
                    # NaN protection
                    if np.any(np.isnan(v_pred)) or np.any(np.isinf(v_pred)):
                        v_pred = np.zeros_like(v_pred)
                    v_pred = np.clip(v_pred, -100, 100)

                dropout_step = info['dropout_step']
                frozen_obs = info['frozen_obs']
                obs = frozen_obs + v_pred * dt * (dropout_step + 1)

                vel_errors.append(np.mean(np.abs(obs - info['true_obs'])))
            else:
                obs = obs_raw

            if term or trunc:
                break
        rewards.append(total_reward)

    env.close()
    return summarize_results('PANO (dropout)', np.array(rewards),
                              np.array(vel_errors) if vel_errors else None)

# =============================================================================
# MAIN
# =============================================================================

def run_experiment(n_episodes=100, dropout_duration=5, velocity_threshold=0.1,
                   oracle_path='hopper_sac.zip', oracle_steps=1_000_000,
                   results_dir='../../results', seed=42, retrain_oracle=False,
                   use_pretrained=True, train_episodes=300, train_epochs=100):
    """Run full PANO experiment with all baselines and statistical tests."""

    np.random.seed(seed)
    torch.manual_seed(seed)

    print("=" * 70)
    print("PANO EVALUATION: Hopper-v4 with Contact-Triggered Dropout")
    print("=" * 70)
    print(f"  Episodes per method: {n_episodes}")
    print(f"  Dropout duration:    {dropout_duration} steps")
    print(f"  Seed:                {seed}")
    print("=" * 70)

    # --- Load oracle ---
    if use_pretrained:
        print("\n[0/5] Loading pretrained expert from HuggingFace...")
        sac_model = get_pretrained_oracle('Hopper-v4')
    elif os.path.exists(oracle_path) and not retrain_oracle:
        print(f"  Loading existing oracle from {oracle_path}")
        print(f"  WARNING: If this oracle was trained with fewer steps, delete it and re-run.")
        sac_model = get_oracle(oracle_path)
    else:
        if os.path.exists(oracle_path):
            os.remove(oracle_path)
            print(f"  Removed stale oracle {oracle_path}, retraining...")
        sac_model = train_oracle(oracle_path, total_timesteps=oracle_steps, seed=seed)

    # --- Train PANO ---
    print("\n[1/5] Training PANO velocity predictor...")
    obs_dim = 11
    action_dim = 3
    data = generate_training_data(sac_model, n_episodes=train_episodes, history_len=5)
    pano_model = PANOVelocityPredictor(obs_dim, action_dim, history_len=5).to(device)
    pano_model = train_pano(pano_model, data, n_epochs=train_epochs)

    # --- Evaluate all methods ---
    print(f"\n[2/5] Evaluating Oracle (no dropout, {n_episodes} episodes)...")
    oracle_results = eval_oracle(sac_model, n_episodes=n_episodes, seed=seed)
    print(f"  Reward: {oracle_results['reward_mean']:.1f} "
          f"[{oracle_results['reward_ci_95_lower']:.1f}, {oracle_results['reward_ci_95_upper']:.1f}]")

    print(f"\n[3/5] Evaluating Frozen Baseline ({n_episodes} episodes)...")
    frozen_results = eval_frozen_baseline(sac_model, n_episodes=n_episodes,
                                           dropout_duration=dropout_duration,
                                           velocity_threshold=velocity_threshold, seed=seed)
    print(f"  Reward: {frozen_results['reward_mean']:.1f} "
          f"[{frozen_results['reward_ci_95_lower']:.1f}, {frozen_results['reward_ci_95_upper']:.1f}]")

    print(f"\n[4/5] Evaluating EKF Baseline ({n_episodes} episodes)...")
    ekf_results = eval_ekf(sac_model, n_episodes=n_episodes,
                            dropout_duration=dropout_duration,
                            velocity_threshold=velocity_threshold, seed=seed)
    print(f"  Reward: {ekf_results['reward_mean']:.1f} "
          f"[{ekf_results['reward_ci_95_lower']:.1f}, {ekf_results['reward_ci_95_upper']:.1f}]")

    print(f"\n[5/5] Evaluating PANO ({n_episodes} episodes)...")
    pano_results = eval_pano(sac_model, pano_model, n_episodes=n_episodes,
                              dropout_duration=dropout_duration,
                              velocity_threshold=velocity_threshold, seed=seed)
    print(f"  Reward: {pano_results['reward_mean']:.1f} "
          f"[{pano_results['reward_ci_95_lower']:.1f}, {pano_results['reward_ci_95_upper']:.1f}]")

    # --- Statistical tests ---
    print("\n" + "=" * 70)
    print("STATISTICAL TESTS (Welch's t-test)")
    print("=" * 70)

    comparisons = {}

    comp_pano_frozen = compare_methods(pano_results, frozen_results, "PANO vs Frozen Baseline")
    comparisons['pano_vs_frozen'] = comp_pano_frozen
    print(f"\n  PANO vs Frozen:  Δ={comp_pano_frozen['improvement_absolute']:+.1f} "
          f"({comp_pano_frozen['improvement_pct']:+.1f}%), "
          f"p={comp_pano_frozen['p_value']:.4f}, d={comp_pano_frozen['cohens_d']:.2f}")

    comp_pano_ekf = compare_methods(pano_results, ekf_results, "PANO vs EKF")
    comparisons['pano_vs_ekf'] = comp_pano_ekf
    print(f"  PANO vs EKF:     Δ={comp_pano_ekf['improvement_absolute']:+.1f} "
          f"({comp_pano_ekf['improvement_pct']:+.1f}%), "
          f"p={comp_pano_ekf['p_value']:.4f}, d={comp_pano_ekf['cohens_d']:.2f}")

    comp_ekf_frozen = compare_methods(ekf_results, frozen_results, "EKF vs Frozen Baseline")
    comparisons['ekf_vs_frozen'] = comp_ekf_frozen
    print(f"  EKF vs Frozen:   Δ={comp_ekf_frozen['improvement_absolute']:+.1f} "
          f"({comp_ekf_frozen['improvement_pct']:+.1f}%), "
          f"p={comp_ekf_frozen['p_value']:.4f}, d={comp_ekf_frozen['cohens_d']:.2f}")

    # --- Summary table ---
    print("\n" + "=" * 70)
    print("RESULTS SUMMARY")
    print("=" * 70)
    print(f"{'Method':<30} {'Reward':>12} {'95% CI':>20} {'Vel Err':>10}")
    print("-" * 72)
    for r in [oracle_results, frozen_results, ekf_results, pano_results]:
        vel = f"{r.get('velocity_error_mean', 0):.1f}" if 'velocity_error_mean' in r else '-'
        print(f"{r['method']:<30} {r['reward_mean']:>8.1f} ± {r['reward_std']:>4.1f}"
              f"  [{r['reward_ci_95_lower']:>6.1f}, {r['reward_ci_95_upper']:>6.1f}]"
              f"  {vel:>10}")

    # --- Save results ---
    all_results = {
        'experiment': 'hopper_pano',
        'env_id': 'Hopper-v4',
        'n_episodes': n_episodes,
        'dropout_duration': dropout_duration,
        'velocity_threshold': velocity_threshold,
        'seed': seed,
        'methods': {
            'oracle': oracle_results,
            'frozen_baseline': frozen_results,
            'ekf': ekf_results,
            'pano': pano_results,
        },
        'comparisons': comparisons,
    }

    os.makedirs(results_dir, exist_ok=True)
    save_results(all_results, os.path.join(results_dir, 'hopper_pano_results.json'))

    return all_results


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='PANO Hopper Experiment')
    parser.add_argument('--n-episodes', type=int, default=100)
    parser.add_argument('--dropout-duration', type=int, default=5)
    parser.add_argument('--oracle-steps', type=int, default=1_000_000)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--results-dir', type=str, default='../../results/phase6')
    parser.add_argument('--retrain-oracle', action='store_true',
                        help='Force retrain oracle even if hopper_sac.zip exists')
    parser.add_argument('--train-episodes', type=int, default=300)
    parser.add_argument('--train-epochs', type=int, default=100)
    args = parser.parse_args()

    run_experiment(
        n_episodes=args.n_episodes,
        dropout_duration=args.dropout_duration,
        oracle_steps=args.oracle_steps,
        results_dir=args.results_dir,
        seed=args.seed,
        retrain_oracle=args.retrain_oracle,
        train_episodes=args.train_episodes,
        train_epochs=args.train_epochs,
    )
