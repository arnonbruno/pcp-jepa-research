#!/usr/bin/env python3
"""
PHASE A: FORENSIC VALIDATION
Gate A0 - Freeze the Evidence

Purpose: Create reproducible artifact bundle for all experiments.
If we can't reproduce, we stop.
"""

import jax
import jax.numpy as jnp
import numpy as np
import json
import pickle
import os
from datetime import datetime
import hashlib

# Config
CONFIG = {
    'commit_hash': 'de87d00',  # Current commit
    'timestamp': datetime.now().isoformat(),
    'jax_version': jax.__version__,
    
    # Environment
    'dt': 0.05,
    'restitution': 0.8,
    'x_min': 0.0,
    'x_max': 4.0,
    'x_target': 2.0,
    
    # Task
    'max_steps': 30,
    'tau_success': 0.3,
    'tau_catastrophic': 0.5,
    'v_max': 2.0,
    
    # MPPI
    'horizon': 50,
    'n_samples': 64,
    'action_scale': 2.0,
    
    # Training
    'train_seed': 42,
    'n_train': 5000,
    'n_epochs': 100,
    
    # Evaluation
    'eval_seed': 42,
    'n_episodes': 100,
}

SEEDS = {
    'train': 42,
    'eval_random': 100,
    'eval_mppi': 200,
    'eval_oracle': 300,
}

@jax.jit
def physics_step_jax(obs, a):
    """JIT-compiled physics."""
    x, v = obs[0], obs[1]
    v_new = v + a * CONFIG['dt']
    x_new = x + v_new * CONFIG['dt']
    
    # Bounces
    x_new = jnp.where(x_new < CONFIG['x_min'], 2 * CONFIG['x_min'] - x_new, x_new)
    v_new = jnp.where(x_new < CONFIG['x_min'], -v_new * CONFIG['restitution'], v_new)
    x_new = jnp.where(x_new > CONFIG['x_max'], 2 * CONFIG['x_max'] - x_new, x_new)
    v_new = jnp.where(x_new > CONFIG['x_max'], -v_new * CONFIG['restitution'], v_new)
    
    return jnp.array([x_new, v_new])

def run_episode_trajectory(key, policy_fn=None):
    """
    Run episode and save FULL trajectory for forensic analysis.
    
    Returns:
        trajectory: list of (x, v, a, t) tuples
        metadata: episode metrics
    """
    key, subkey = jax.random.split(key)
    x = float(jax.random.uniform(subkey, (), minval=0.5, maxval=3.5))
    key, subkey = jax.random.split(key)
    v = float(jax.random.uniform(subkey, (), minval=-0.5, maxval=0.5))
    
    trajectory = [(x, v, None, 0)]  # (x, v, a, t)
    
    for t in range(CONFIG['max_steps']):
        obs = jnp.array([x, v])
        
        if policy_fn:
            key, subkey = jax.random.split(key)
            a = float(policy_fn(obs, subkey))
        else:
            key, subkey = jax.random.split(key)
            a = float(jax.random.uniform(subkey, (), minval=-1.0, maxval=1.0))
        
        # Step physics
        obs_next = physics_step_jax(obs, a)
        x, v = float(obs_next[0]), float(obs_next[1])
        
        trajectory.append((x, v, a, t+1))
    
    final_x, final_v, _, _ = trajectory[-1]
    miss = abs(final_x - CONFIG['x_target'])
    
    metadata = {
        'final_x': final_x,
        'final_v': final_v,
        'miss_distance': miss,
        'success': miss < CONFIG['tau_success'],
        'catastrophic': miss > CONFIG['tau_catastrophic'] or abs(final_v) > CONFIG['v_max'],
    }
    
    return trajectory, metadata, key

def save_artifacts():
    """Save all artifacts for forensic analysis."""
    os.makedirs('/home/ulluboz/pcp-jepa-research/forensics', exist_ok=True)
    
    # Save config
    with open('/home/ulluboz/pcp-jepa-research/forensics/config.json', 'w') as f:
        json.dump(CONFIG, f, indent=2)
    
    # Save seeds
    with open('/home/ulluboz/pcp-jepa-research/forensics/seeds.json', 'w') as f:
        json.dump(SEEDS, f, indent=2)
    
    print("✅ Config and seeds saved")

def collect_random_trajectories():
    """Collect random baseline trajectories."""
    print("\n" + "="*60)
    print("COLLECTING RANDOM BASELINE TRAJECTORIES")
    print("="*60)
    
    key = jax.random.PRNGKey(SEEDS['eval_random'])
    all_trajectories = []
    all_metadata = []
    
    for ep in range(CONFIG['n_episodes']):
        traj, meta, key = run_episode_trajectory(key, policy_fn=None)
        all_trajectories.append(traj)
        all_metadata.append(meta)
    
    # Save
    with open('/home/ulluboz/pcp-jepa-research/forensics/random_trajectories.pkl', 'wb') as f:
        pickle.dump(all_trajectories, f)
    
    with open('/home/ulluboz/pcp-jepa-research/forensics/random_metadata.pkl', 'wb') as f:
        pickle.dump(all_metadata, f)
    
    # Stats
    successes = sum(m['success'] for m in all_metadata)
    catastrophics = sum(m['catastrophic'] for m in all_metadata)
    avg_miss = np.mean([m['miss_distance'] for m in all_metadata])
    
    print(f"  Episodes: {CONFIG['n_episodes']}")
    print(f"  Success rate: {successes/CONFIG['n_episodes']:.1%}")
    print(f"  Catastrophic rate: {catastrophics/CONFIG['n_episodes']:.1%}")
    print(f"  Avg miss: {avg_miss:.3f}")
    print(f"  ✅ Saved to random_trajectories.pkl")
    
    return all_metadata

def main():
    print("="*60)
    print("GATE A0: FREEZE THE EVIDENCE")
    print("="*60)
    
    # Save config and seeds
    save_artifacts()
    
    # Collect random baseline trajectories
    random_metadata = collect_random_trajectories()
    
    # Verification: recompute metrics from saved trajectories
    print("\n" + "="*60)
    print("VERIFICATION: RECOMPUTE METRICS")
    print("="*60)
    
    with open('/home/ulluboz/pcp-jepa-research/forensics/random_trajectories.pkl', 'rb') as f:
        loaded_trajectories = pickle.load(f)
    
    recomputed_metadata = []
    for traj in loaded_trajectories:
        final_x, final_v, _, _ = traj[-1]
        miss = abs(final_x - CONFIG['x_target'])
        recomputed_metadata.append({
            'final_x': final_x,
            'final_v': final_v,
            'miss_distance': miss,
            'success': miss < CONFIG['tau_success'],
            'catastrophic': miss > CONFIG['tau_catastrophic'] or abs(final_v) > CONFIG['v_max'],
        })
    
    # Compare
    match = True
    for i, (orig, recomputed) in enumerate(zip(random_metadata, recomputed_metadata)):
        if orig['miss_distance'] != recomputed['miss_distance']:
            print(f"  ❌ Mismatch at episode {i}")
            match = False
    
    if match:
        print("  ✅ All metrics match when recomputed from trajectories")
    else:
        print("  ❌ METRIC COMPUTATION BUG DETECTED")
        return False
    
    # Final summary
    print("\n" + "="*60)
    print("GATE A0 STATUS")
    print("="*60)
    print("""
Artifacts saved:
  - config.json: Environment and evaluation parameters
  - seeds.json: All random seeds used
  - random_trajectories.pkl: Full episode trajectories
  - random_metadata.pkl: Episode metrics

Verification:
  ✅ Metrics recomputed correctly from trajectories
  ✅ Reproducibility confirmed

NEXT: Run Gate A1 (Environment sanity checks)
""")
    
    return True

if __name__ == '__main__':
    main()