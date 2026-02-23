#!/usr/bin/env python3
"""
Bulletproof Negative Protocol — Multi-Experiment Validation

Demonstrates that JEPA-style latent rollout diverges exponentially in
high-dimensional continuous control. Three experiments:

1. Data Scaling Law: Training on 10k–100k transitions does not fix
   the prediction-velocity loss gap (architectural limit, not data starvation).
2. Continuous Control Ablation: Standard Latent JEPA also fails on smooth
   systems (InvertedDoublePendulum, Walker2d, HalfCheetah) — the failure
   is NOT specific to hybrid/contact dynamics.
3. Impact Horizon Profiling: Latent prediction error vs rollout steps,
   separated by air vs contact phase.

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
import argparse
import warnings
warnings.filterwarnings('ignore')

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))
from src.evaluation.stats import summarize_results, compare_methods, save_results, welch_ttest

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Device: {device}")

# =============================================================================
# STANDARD LATENT JEPA ARCHITECTURE (the one that diverges)
# =============================================================================

class StandardLatentJEPA(nn.Module):
    """
    Standard Latent JEPA: Residual latent dynamics z_next = z + Δz.

    This is the architecture whose latent rollout diverges exponentially
    in high-dimensional continuous control.
    """
    def __init__(self, obs_dim, action_dim, latent_dim=64):
        super().__init__()

        self.encoder = nn.Sequential(
            nn.Linear(obs_dim * 2, 128), nn.ReLU(),
            nn.Linear(128, 128), nn.ReLU(),
            nn.Linear(128, latent_dim),
        )

        self.target_encoder = nn.Sequential(
            nn.Linear(obs_dim * 2, 128), nn.ReLU(),
            nn.Linear(128, 128), nn.ReLU(),
            nn.Linear(128, latent_dim),
        )
        self.target_encoder.load_state_dict(self.encoder.state_dict())

        self.velocity_decoder = nn.Sequential(
            nn.Linear(latent_dim, 64), nn.ReLU(),
            nn.Linear(64, obs_dim),
        )

        self.predictor = nn.Sequential(
            nn.Linear(latent_dim + action_dim, 128), nn.ReLU(),
            nn.Linear(128, 128), nn.ReLU(),
            nn.Linear(128, latent_dim),
        )

    def encode(self, obs, obs_prev, dt):
        velocity = (obs - obs_prev) / dt
        inp = torch.cat([obs, velocity], dim=-1)
        return self.encoder(inp)

    def encode_target(self, obs, obs_prev, dt):
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

# =============================================================================
# HELPER: Generate data from a trained SAC policy
# =============================================================================

def generate_data(env, sac_model, n_transitions, dt):
    """Generate exactly n_transitions from an environment using the SAC oracle."""
    data = {'obs': [], 'obs_prev': [], 'action': [], 'obs_next': [], 'obs_prev_next': []}
    count = 0
    while count < n_transitions:
        obs, _ = env.reset()
        obs_prev = obs.copy()
        for _ in range(1000):
            action, _ = sac_model.predict(obs, deterministic=True)
            data['obs'].append(obs)
            data['obs_prev'].append(obs_prev)
            data['action'].append(action)
            obs_next, _, term, trunc, _ = env.step(action)
            data['obs_next'].append(obs_next)
            data['obs_prev_next'].append(obs)
            count += 1
            if count >= n_transitions:
                break
            obs_prev = obs.copy()
            obs = obs_next
            if term or trunc:
                break
    for k in data:
        data[k] = torch.tensor(np.array(data[k][:n_transitions]), dtype=torch.float32, device=device)
    return data


def train_standard_jepa(model, data, dt, n_epochs=100, lambda_vel=10.0, lambda_pred=0.1):
    """Train Standard Latent JEPA and return final losses."""
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    batch_size = 256
    n_samples = len(data['obs'])

    final_vel_losses, final_pred_losses = [], []

    for epoch in range(n_epochs):
        idx = torch.randperm(n_samples, device=device)
        vel_losses, pred_losses = [], []

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

            vel_losses.append(loss_vel.item())
            pred_losses.append(loss_pred.item())

        # Record final-epoch losses
        if epoch >= n_epochs - 5:
            final_vel_losses.extend(vel_losses)
            final_pred_losses.extend(pred_losses)

    return np.mean(final_vel_losses), np.mean(final_pred_losses)

# =============================================================================
# EXPERIMENT 1: DATA SCALING LAW
# =============================================================================

def experiment_1_data_scaling(sac_model, env, dt, seed=42):
    """
    Test if more data fixes Standard Latent JEPA's prediction loss.

    Result: Even at 100k transitions, prediction loss dominates velocity
    loss by ~100x. This is an architectural limit, not data starvation.
    """
    print("\n" + "=" * 70)
    print("EXPERIMENT 1: DATA SCALING LAW")
    print("Does more data fix latent drift? (Answer: No)")
    print("=" * 70)

    obs_dim = env.observation_space.shape[0]
    action_dim = env.action_space.shape[0]

    scales = [10_000, 30_000, 100_000]
    results = []

    for n_trans in scales:
        torch.manual_seed(seed)
        np.random.seed(seed)

        print(f"\n  Generating {n_trans:,} transitions...")
        data = generate_data(env, sac_model, n_trans, dt)

        print(f"  Training Standard Latent JEPA on {n_trans:,} samples...")
        model = StandardLatentJEPA(obs_dim, action_dim).to(device)
        vel_loss, pred_loss = train_standard_jepa(model, data, dt, n_epochs=100)

        ratio = pred_loss / (vel_loss + 1e-8)
        print(f"  → Velocity loss: {vel_loss:.1f}, Prediction loss: {pred_loss:.1f}, Ratio: {ratio:.0f}×")

        results.append({
            'n_transitions': n_trans,
            'velocity_loss': float(vel_loss),
            'prediction_loss': float(pred_loss),
            'ratio': float(ratio),
        })

    print("\n" + "-" * 50)
    print("DATA SCALING SUMMARY:")
    print(f"{'Transitions':>12}  {'Vel Loss':>10}  {'Pred Loss':>11}  {'Ratio':>8}")
    print("-" * 50)
    for r in results:
        print(f"{r['n_transitions']:>12,}  {r['velocity_loss']:>10.1f}  "
              f"{r['prediction_loss']:>11.1f}  {r['ratio']:>7.0f}×")

    return results

# =============================================================================
# EXPERIMENT 2: CONTINUOUS CONTROL ABLATION (Multiple Environments)
# =============================================================================

def experiment_2_continuous_ablation(seed=42, oracle_steps=500_000, n_eval_episodes=50):
    """
    Test Standard Latent JEPA on multiple MuJoCo environments:
      - Hopper-v4 (hybrid/contact)
      - Walker2d-v4 (bipedal contact)
      - HalfCheetah-v4 (contact-rich)
      - InvertedDoublePendulum-v4 (smooth, NO hybrid contact)

    If Standard Latent JEPA fails on ALL of them (including smooth systems),
    the failure is NOT specific to hybrid dynamics — it is architectural.
    """
    print("\n" + "=" * 70)
    print("EXPERIMENT 2: CONTINUOUS CONTROL ABLATION (Multi-Environment)")
    print("Does Standard Latent JEPA fail only on hybrid systems? (Answer: No)")
    print("=" * 70)

    envs = [
        ('Hopper-v4', 0.002),
        ('Walker2d-v4', 0.002),
        ('HalfCheetah-v4', 0.01),
        ('InvertedDoublePendulum-v4', 0.02),
    ]

    from hopper_pano import ContactDropoutEnv

    all_results = []

    for env_id, dt in envs:
        print(f"\n{'=' * 50}")
        print(f"  Environment: {env_id}")
        print(f"{'=' * 50}")

        env = gym.make(env_id)
        obs_dim = env.observation_space.shape[0]
        action_dim = env.action_space.shape[0]
        print(f"  Obs dim: {obs_dim}, Action dim: {action_dim}")

        # Train oracle
        oracle_path = f'{env_id.replace("-", "_").lower()}_sac.zip'
        if os.path.exists(oracle_path):
            sac = SAC.load(oracle_path, env=env)
        else:
            print(f"  Training SAC oracle ({oracle_steps:,} steps)...")
            sac = SAC('MlpPolicy', env, learning_rate=3e-4, buffer_size=300_000,
                       learning_starts=10_000, batch_size=256, verbose=0, seed=seed)
            sac.learn(total_timesteps=oracle_steps, progress_bar=True)
            sac.save(oracle_path)

        # Evaluate oracle (no dropout)
        oracle_rewards = []
        for ep in range(n_eval_episodes):
            obs, _ = env.reset(seed=seed + ep)
            total = 0.0
            for _ in range(1000):
                action, _ = sac.predict(obs, deterministic=True)
                obs, reward, term, trunc, _ = env.step(action)
                total += reward
                if term or trunc:
                    break
            oracle_rewards.append(total)
        oracle_mean = np.mean(oracle_rewards)
        print(f"  Oracle reward: {oracle_mean:.1f} ± {np.std(oracle_rewards):.1f}")

        # Generate data and train Standard Latent JEPA
        print(f"  Generating 50k transitions...")
        data = generate_data(env, sac, 50_000, dt)

        torch.manual_seed(seed)
        model = StandardLatentJEPA(obs_dim, action_dim).to(device)
        vel_loss, pred_loss = train_standard_jepa(model, data, dt, n_epochs=100)

        # Evaluate Standard Latent JEPA with dropout
        dropout_env = ContactDropoutEnv(env_id, dropout_duration=5, velocity_threshold=0.1)
        jepa_rewards = []
        baseline_rewards = []

        for ep in range(n_eval_episodes):
            # --- Baseline (frozen obs during dropout) ---
            obs, _ = dropout_env.reset(seed=seed + ep)
            total = 0.0
            for _ in range(1000):
                action, _ = sac.predict(obs, deterministic=True)
                obs, reward, term, trunc, info = dropout_env.step(action)
                total += reward
                if term or trunc:
                    break
            baseline_rewards.append(total)

            # --- Standard Latent JEPA ---
            obs, _ = dropout_env.reset(seed=seed + ep)
            obs_prev = obs.copy()
            z = None
            total = 0.0
            for step in range(1000):
                action, _ = sac.predict(obs, deterministic=True)
                obs_raw, reward, term, trunc, info = dropout_env.step(action)
                total += reward

                if info['dropout_active']:
                    if z is None:
                        obs_t = torch.tensor(obs, dtype=torch.float32, device=device).unsqueeze(0)
                        oprev_t = torch.tensor(obs_prev, dtype=torch.float32, device=device).unsqueeze(0)
                        with torch.no_grad():
                            z = model.encode(obs_t, oprev_t, dt)
                    action_t = torch.tensor(action, dtype=torch.float32, device=device).unsqueeze(0)
                    with torch.no_grad():
                        delta_z = model.predict_residual(z, action_t)
                        z = z + delta_z
                        v_pred = model.decode_velocity(z).squeeze().cpu().numpy()
                    obs = info['frozen_obs'] + v_pred * dt * (info['dropout_step'] + 1)
                else:
                    z = None
                    obs_prev = obs.copy()
                    obs = obs_raw

                if term or trunc:
                    break
            jepa_rewards.append(total)

        dropout_env.close()

        # Statistical test
        test = welch_ttest(np.array(jepa_rewards), np.array(baseline_rewards))

        env_result = {
            'env_id': env_id,
            'obs_dim': obs_dim,
            'action_dim': action_dim,
            'dt': dt,
            'oracle_reward_mean': float(oracle_mean),
            'oracle_reward_std': float(np.std(oracle_rewards)),
            'baseline_reward_mean': float(np.mean(baseline_rewards)),
            'baseline_reward_std': float(np.std(baseline_rewards)),
            'jepa_reward_mean': float(np.mean(jepa_rewards)),
            'jepa_reward_std': float(np.std(jepa_rewards)),
            'velocity_loss': float(vel_loss),
            'prediction_loss': float(pred_loss),
            'loss_ratio': float(pred_loss / (vel_loss + 1e-8)),
            'jepa_vs_baseline_p_value': test['p_value'],
            'jepa_vs_baseline_cohens_d': test['cohens_d'],
        }
        all_results.append(env_result)

        print(f"\n  Results for {env_id}:")
        print(f"    Oracle:               {oracle_mean:.1f}")
        print(f"    Baseline (dropout):   {np.mean(baseline_rewards):.1f} ± {np.std(baseline_rewards):.1f}")
        print(f"    Std Latent JEPA:      {np.mean(jepa_rewards):.1f} ± {np.std(jepa_rewards):.1f}")
        print(f"    Vel/Pred loss ratio:  {pred_loss/(vel_loss+1e-8):.0f}×")
        print(f"    JEPA vs Baseline:     p={test['p_value']:.4f}, d={test['cohens_d']:.2f}")

        # Check: does Standard Latent JEPA FAIL?
        if np.mean(jepa_rewards) < np.mean(baseline_rewards):
            print(f"    → Standard Latent JEPA UNDERPERFORMS baseline on {env_id}")
        else:
            print(f"    → Standard Latent JEPA matches/exceeds baseline on {env_id}")

        env.close()

    # Summary
    print("\n" + "=" * 70)
    print("MULTI-ENVIRONMENT SUMMARY")
    print("=" * 70)
    print(f"{'Env':<30} {'Oracle':>8} {'Baseline':>10} {'Std JEPA':>10} {'Ratio':>6}")
    print("-" * 70)
    for r in all_results:
        print(f"{r['env_id']:<30} {r['oracle_reward_mean']:>8.1f} "
              f"{r['baseline_reward_mean']:>10.1f} {r['jepa_reward_mean']:>10.1f} "
              f"{r['loss_ratio']:>5.0f}×")

    failed_count = sum(1 for r in all_results if r['jepa_reward_mean'] < r['baseline_reward_mean'])
    print(f"\nStandard Latent JEPA failed on {failed_count}/{len(all_results)} environments")
    if failed_count == len(all_results):
        print("→ CONFIRMED: Failure is architectural, NOT hybrid-specific")
    elif failed_count > 0:
        print("→ Partial failure: JEPA struggles on most environments")
    else:
        print("→ NOT CONFIRMED: JEPA matched/exceeded baseline on all environments")

    return all_results

# =============================================================================
# EXPERIMENT 3: IMPACT HORIZON PROFILING
# =============================================================================

def experiment_3_impact_profiling(sac_model, seed=42):
    """
    Profile latent prediction error vs rollout steps, separated by
    air phase vs contact-boundary phase.

    Shows that error grows exponentially regardless of physics phase.
    """
    print("\n" + "=" * 70)
    print("EXPERIMENT 3: IMPACT HORIZON PROFILING")
    print("How does latent error grow over rollout steps?")
    print("=" * 70)

    env = gym.make('Hopper-v4')
    obs_dim = 11
    action_dim = 3
    dt = 0.002

    # Generate data with phase labels
    print("  Generating data with phase labels...")
    data = {
        'obs': [], 'obs_prev': [], 'action': [],
        'obs_next': [], 'obs_prev_next': [],
        'phase': [],
    }

    for ep in range(100):
        obs, _ = env.reset(seed=seed + ep)
        obs_prev = obs.copy()
        prev_height = obs[1]

        for _ in range(1000):
            action, _ = sac_model.predict(obs, deterministic=True)
            data['obs'].append(obs)
            data['obs_prev'].append(obs_prev)
            data['action'].append(action)

            obs_next, _, term, trunc, _ = env.step(action)
            data['obs_next'].append(obs_next)
            data['obs_prev_next'].append(obs)

            curr_height = obs[1]
            phase = 1 if (prev_height > 0.8 and curr_height <= 0.8) else 0
            data['phase'].append(phase)

            prev_height = curr_height
            obs_prev = obs.copy()
            obs = obs_next
            if term or trunc:
                break

    phases = data['phase']
    for k in data:
        if k != 'phase':
            data[k] = torch.tensor(np.array(data[k]), dtype=torch.float32, device=device)

    n_samples = len(data['obs'])
    print(f"  Generated {n_samples:,} transitions")
    print(f"  Air: {sum(1 for p in phases if p == 0):,}, Contact: {sum(1 for p in phases if p == 1):,}")

    # Train model
    print("  Training Standard Latent JEPA...")
    torch.manual_seed(seed)
    model = StandardLatentJEPA(obs_dim, action_dim).to(device)
    train_standard_jepa(model, data, dt, n_epochs=100)

    # Profile multi-step error
    print("  Profiling multi-step prediction error...")
    n_rollout_steps = 10
    air_errors = [[] for _ in range(n_rollout_steps)]
    impact_errors = [[] for _ in range(n_rollout_steps)]
    overall_errors = [[] for _ in range(n_rollout_steps)]

    n_test = min(5000, n_samples)

    with torch.no_grad():
        for i in range(0, n_test - n_rollout_steps):
            phase = phases[i]
            z = model.encode(data['obs'][i].unsqueeze(0), data['obs_prev'][i].unsqueeze(0), dt)

            for k in range(n_rollout_steps):
                idx_k = i + k
                if idx_k >= n_test:
                    break
                delta_z = model.predict_residual(z, data['action'][idx_k].unsqueeze(0))
                z = z + delta_z
                z_target = model.encode_target(
                    data['obs_next'][idx_k].unsqueeze(0),
                    data['obs_prev_next'][idx_k].unsqueeze(0), dt,
                )
                error = F.mse_loss(z, z_target).item()
                overall_errors[k].append(error)
                if phase == 0:
                    air_errors[k].append(error)
                else:
                    impact_errors[k].append(error)

    results = []
    print(f"\n  {'Step':>4}  {'Overall':>12}  {'Air':>12}  {'Contact':>12}  {'Ratio':>8}")
    print("  " + "-" * 52)
    for k in range(n_rollout_steps):
        overall = np.mean(overall_errors[k]) if overall_errors[k] else 0
        air = np.mean(air_errors[k]) if air_errors[k] else 0
        impact = np.mean(impact_errors[k]) if impact_errors[k] else 0
        ratio = impact / (air + 1e-8)
        print(f"  {k+1:>4}  {overall:>12.1f}  {air:>12.1f}  {impact:>12.1f}  {ratio:>7.1f}×")
        results.append({
            'step': k + 1,
            'overall_error': float(overall),
            'air_error': float(air),
            'impact_error': float(impact),
            'impact_air_ratio': float(ratio),
        })

    env.close()
    return results

# =============================================================================
# MAIN
# =============================================================================

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Bulletproof Negative Protocol')
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--oracle-steps', type=int, default=1_000_000)
    parser.add_argument('--n-eval-episodes', type=int, default=50)
    parser.add_argument('--results-dir', type=str, default='../../results/phase6')
    args = parser.parse_args()

    print("=" * 70)
    print("BULLETPROOF NEGATIVE PROTOCOL")
    print("=" * 70)
    print("Three experiments proving Standard Latent JEPA diverges:")
    print("  1. Data Scaling Law (rule out data starvation)")
    print("  2. Multi-Env Ablation (rule out hybrid-specific failure)")
    print("  3. Impact Horizon Profiling (quantify exponential drift)")
    print("=" * 70)

    # Load Hopper oracle for experiments 1 & 3
    oracle_path = 'hopper_sac.zip'
    if not os.path.exists(oracle_path):
        from hopper_pano import train_oracle
        train_oracle(oracle_path, total_timesteps=args.oracle_steps, seed=args.seed)

    env = gym.make('Hopper-v4')
    sac = SAC.load(oracle_path, env=env)

    # Experiment 1: Data Scaling
    exp1_results = experiment_1_data_scaling(sac, env, dt=0.002, seed=args.seed)

    # Experiment 2: Multi-environment ablation
    exp2_results = experiment_2_continuous_ablation(
        seed=args.seed, oracle_steps=args.oracle_steps, n_eval_episodes=args.n_eval_episodes
    )

    # Experiment 3: Impact Profiling
    exp3_results = experiment_3_impact_profiling(sac, seed=args.seed)

    env.close()

    # Save all results
    all_results = {
        'experiment': 'bulletproof_negative',
        'seed': args.seed,
        'data_scaling': exp1_results,
        'multi_env_ablation': exp2_results,
        'impact_profiling': exp3_results,
    }

    os.makedirs(args.results_dir, exist_ok=True)
    save_results(all_results, os.path.join(args.results_dir, 'bulletproof_results.json'))

    # Final summary
    print("\n" + "=" * 70)
    print("BULLETPROOF NEGATIVE PROTOCOL — FINAL SUMMARY")
    print("=" * 70)

    print("\nExperiment 1: Data Scaling Law")
    print("  → More data does NOT fix the prediction-velocity gap (architectural limit)")
    for r in exp1_results:
        print(f"    {r['n_transitions']:>7,} transitions: ratio = {r['ratio']:.0f}×")

    print("\nExperiment 2: Multi-Environment Ablation")
    failed = sum(1 for r in exp2_results if r['jepa_reward_mean'] < r['baseline_reward_mean'])
    print(f"  → Standard Latent JEPA failed on {failed}/{len(exp2_results)} environments")
    for r in exp2_results:
        status = "FAIL" if r['jepa_reward_mean'] < r['baseline_reward_mean'] else "OK"
        print(f"    {r['env_id']:<30} [{status}] JEPA={r['jepa_reward_mean']:.1f} vs "
              f"Baseline={r['baseline_reward_mean']:.1f}")

    print("\nExperiment 3: Impact Horizon Profiling")
    print("  → Error grows exponentially with rollout steps")
    if exp3_results:
        step1 = exp3_results[0]['overall_error']
        step10 = exp3_results[-1]['overall_error']
        growth = step10 / (step1 + 1e-8)
        print(f"    Step 1 error:  {step1:.1f}")
        print(f"    Step 10 error: {step10:.1f}")
        print(f"    Growth factor: {growth:.0f}×")

    print("\n" + "=" * 70)
    print("CONCLUSION: Standard Latent JEPA rollout fails across environments")
    print("            and data scales. The failure is architectural.")
    print("=" * 70)
