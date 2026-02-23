#!/usr/bin/env python3
"""
PHASE A: FORENSIC VALIDATION
Gate A1 - Environment Sanity Checks

A1.1: Reachability check - can we actually reach the target?
A1.2: Simple controller baseline - does ANY controller beat random?
A1.3: Metric sanity - are metrics computed correctly?
"""

import jax
import jax.numpy as jnp
import numpy as np
import json
import pickle
import os
from tqdm import tqdm

# Load config from Gate A0
with open('/home/ulluboz/pcp-jepa-research/forensics/config.json', 'r') as f:
    CONFIG = json.load(f)

# ============================================================
# A1.1: REACHABILITY CHECK
# ============================================================

def reachability_check():
    """
    For each (H, action_scale), compute reachable x-range via brute-force sampling.
    
    Pass condition: target x* is inside the 95% reachable interval for at least
    one tested H/scale configuration.
    """
    print("\n" + "="*60)
    print("A1.1: REACHABILITY CHECK")
    print("="*60)
    
    @jax.jit
    def physics_step(obs, a, dt=0.05, restitution=0.8):
        x, v = obs[0], obs[1]
        v_new = v + a * dt
        x_new = x + v_new * dt
        x_new = jnp.where(x_new < 0, -x_new, x_new)
        v_new = jnp.where(x_new < 0, -v_new * restitution, v_new)
        x_new = jnp.where(x_new > 4, 8 - x_new, x_new)
        v_new = jnp.where(x_new > 4, -v_new * restitution, v_new)
        return jnp.array([x_new, v_new])
    
def create_rollout_fn(horizon, physics_step_fn):
    """Create JIT-compiled rollout function with static horizon."""
    @jax.jit
    def rollout_batch(obs, actions):
        def single_rollout(action_seq):
            current = obs
            for t in range(horizon):  # horizon is now static (closure)
                current = physics_step_fn(current, action_seq[t])
            return current[0]
        return jax.vmap(single_rollout)(actions)
    return rollout_batch
    
    # Test configurations
    configs = [
        {'horizon': 10, 'action_scale': 1.0},
        {'horizon': 20, 'action_scale': 1.0},
        {'horizon': 30, 'action_scale': 1.0},
        {'horizon': 50, 'action_scale': 1.0},
        {'horizon': 50, 'action_scale': 2.0},
        {'horizon': 100, 'action_scale': 1.0},
        {'horizon': 100, 'action_scale': 2.0},
    ]
    
    # Starting positions to test
    start_positions = [0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5]
    target = CONFIG['x_target']
    n_samples = 50000  # Large sample for accurate reachability
    
    results = []
    reachable_configs = []
    
    print(f"\nTarget: x = {target}")
    print(f"Sample size: {n_samples} trajectories per config")
    print(f"\n{'Horizon':<8} {'Scale':<6} {'Start':<6} {'Min x':<8} {'Max x':<8} {'Can Reach?':<10}")
    print("-" * 60)
    
    key = jax.random.PRNGKey(0)
    
    for cfg in configs:
        H = cfg['horizon']
        scale = cfg['action_scale']
        
        for x0 in start_positions:
            obs = jnp.array([x0, 0.0])
            
            # Sample action sequences
            key, subkey = jax.random.split(key)
            actions = jax.random.uniform(subkey, (n_samples, H, 1), 
                                         minval=-scale, maxval=scale)
            
            # Rollout
            final_x = rollout_batch(obs, actions, H)
            
            # Compute reachable interval (95%)
            x_min = float(jnp.percentile(final_x, 2.5))
            x_max = float(jnp.percentile(final_x, 97.5))
            
            can_reach = x_min <= target <= x_max
            
            print(f"{H:<8} {scale:<6.1f} {x0:<6.1f} {x_min:<8.2f} {x_max:<8.2f} {'✅' if can_reach else '❌':<10}")
            
            results.append({
                'horizon': H,
                'action_scale': scale,
                'start': x0,
                'x_min': x_min,
                'x_max': x_max,
                'can_reach': can_reach
            })
            
            if can_reach:
                reachable_configs.append(cfg)
    
    # Summary
    unique_reachable = {f"{c['horizon']}_{c['action_scale']}" for c in reachable_configs}
    
    print("\n" + "="*60)
    print("REACHABILITY SUMMARY")
    print("="*60)
    print(f"\nConfigs that can reach target from SOME starting position:")
    for cfg_str in sorted(unique_reachable):
        print(f"  - H={cfg_str.split('_')[0]}, scale={cfg_str.split('_')[1]}")
    
    if unique_reachable:
        print("\n✅ PASS: Target is reachable with some H/scale configuration")
        return True, results
    else:
        print("\n❌ FAIL: Target is NOT reachable with any tested configuration")
        print("   → Task is fundamentally unsolvable, MPPI comparisons meaningless")
        return False, results


# ============================================================
# A1.2: SIMPLE CONTROLLER BASELINE
# ============================================================

def simple_controller_baseline():
    """
    Test if ANY controller beats random on this task.
    
    Controllers:
    1. P controller: a = clip(k * (x* - x), -a_max, a_max)
    2. Bang-bang: a = sign(x* - x) with damping near target
    
    Pass condition: at least one beats random by clear margin.
    """
    print("\n" + "="*60)
    print("A1.2: SIMPLE CONTROLLER BASELINE")
    print("="*60)
    
    @jax.jit
    def physics_step(obs, a):
        x, v = obs[0], obs[1]
        v_new = v + a * 0.05
        x_new = x + v_new * 0.05
        x_new = jnp.where(x_new < 0, -x_new, x_new)
        v_new = jnp.where(x_new < 0, -v_new * 0.8, v_new)
        x_new = jnp.where(x_new > 4, 8 - x_new, x_new)
        v_new = jnp.where(x_new > 4, -v_new * 0.8, v_new)
        return jnp.array([x_new, v_new])
    
    def p_controller(obs, k=2.0, a_max=1.0):
        """Proportional controller."""
        x, v = float(obs[0]), float(obs[1])
        a = np.clip(k * (CONFIG['x_target'] - x), -a_max, a_max)
        return a
    
    def bangbang_controller(obs, damping_threshold=0.5, a_max=1.0):
        """Bang-bang controller with damping near target."""
        x, v = float(obs[0]), float(obs[1])
        error = CONFIG['x_target'] - x
        
        if abs(error) < damping_threshold:
            # Damping mode - reduce velocity
            a = -np.sign(v) * a_max * 0.5
        else:
            # Full thrust toward target
            a = np.sign(error) * a_max
        
        return a
    
    def run_episode(policy_fn, key):
        key, subkey = jax.random.split(key)
        x = float(jax.random.uniform(subkey, (), minval=0.5, maxval=3.5))
        key, subkey = jax.random.split(key)
        v = float(jax.random.uniform(subkey, (), minval=-0.5, maxval=0.5))
        
        for _ in range(CONFIG['max_steps']):
            obs = jnp.array([x, v])
            a = policy_fn(obs)
            obs_next = physics_step(obs, a)
            x, v = float(obs_next[0]), float(obs_next[1])
        
        miss = abs(x - CONFIG['x_target'])
        return miss < CONFIG['tau_success'], miss
    
    # Test policies
    n_episodes = CONFIG['n_episodes']
    
    policies = {
        'random': lambda obs, key: float(jax.random.uniform(key, (), minval=-1.0, maxval=1.0)),
        'p_k1.0': lambda obs, key: p_controller(obs, k=1.0),
        'p_k2.0': lambda obs, key: p_controller(obs, k=2.0),
        'p_k5.0': lambda obs, key: p_controller(obs, k=5.0),
        'bangbang': lambda obs, key: bangbang_controller(obs),
    }
    
    results = {}
    
    for name, policy_fn in policies.items():
        print(f"\nTesting {name}...")
        key = jax.random.PRNGKey(CONFIG['eval_seed'])
        
        successes = 0
        misses = []
        
        for _ in range(n_episodes):
            key, subkey = jax.random.split(key)
            success, miss = run_episode(policy_fn, subkey)
            if success:
                successes += 1
            misses.append(miss)
        
        sr = successes / n_episodes
        avg_miss = np.mean(misses)
        
        results[name] = {
            'success_rate': sr,
            'avg_miss': avg_miss
        }
        
        print(f"  Success: {sr:.1%}, Avg miss: {avg_miss:.3f}")
    
    # Compare to random
    random_sr = results['random']['success_rate']
    better_policies = [name for name, r in results.items() 
                       if r['success_rate'] > random_sr + 0.05]  # At least 5% better
    
    print("\n" + "="*60)
    print("CONTROLLER COMPARISON")
    print("="*60)
    print(f"\nRandom baseline: {random_sr:.1%}")
    print(f"\nPolicies beating random by >5%:")
    for name in better_policies:
        imp = results[name]['success_rate'] - random_sr
        print(f"  - {name}: {results[name]['success_rate']:.1%} (+{imp:.1%})")
    
    if better_policies:
        print("\n✅ PASS: At least one simple controller beats random")
        return True, results
    else:
        print("\n❌ FAIL: No simple controller beats random")
        print("   → Task may be luck-dominated or success threshold wrong")
        return False, results


# ============================================================
# A1.3: METRIC SANITY
# ============================================================

def metric_sanity_check():
    """
    Verify all metric computations on saved trajectories.
    
    Pass condition: metrics consistent when recomputed offline.
    """
    print("\n" + "="*60)
    print("A1.3: METRIC SANITY CHECK")
    print("="*60)
    
    # Load saved trajectories
    with open('/home/ulluboz/pcp-jepa-research/forensics/random_trajectories.pkl', 'rb') as f:
        trajectories = pickle.load(f)
    
    with open('/home/ulluboz/pcp-jepa-research/forensics/random_metadata.pkl', 'rb') as f:
        original_metadata = pickle.load(f)
    
    print(f"\nLoaded {len(trajectories)} trajectories")
    
    # Recompute all metrics
    errors = []
    
    for i, (traj, orig_meta) in enumerate(zip(trajectories, original_metadata)):
        # Get final state
        final_x, final_v, final_a, final_t = traj[-1]
        
        # Recompute
        miss = abs(final_x - CONFIG['x_target'])
        success = miss < CONFIG['tau_success']
        catastrophic = miss > CONFIG['tau_catastrophic'] or abs(final_v) > CONFIG['v_max']
        
        # Check
        if abs(miss - orig_meta['miss_distance']) > 1e-6:
            errors.append(f"Episode {i}: miss mismatch {miss} vs {orig_meta['miss_distance']}")
        if success != orig_meta['success']:
            errors.append(f"Episode {i}: success mismatch")
        if catastrophic != orig_meta['catastrophic']:
            errors.append(f"Episode {i}: catastrophic mismatch")
    
    if errors:
        print("\n❌ METRIC ERRORS FOUND:")
        for err in errors[:10]:  # Show first 10
            print(f"  - {err}")
        return False
    else:
        print(f"\n✅ PASS: All {len(trajectories)} trajectories have consistent metrics")
        
        # Additional sanity: check trajectory structure
        print("\nTrajectory structure check:")
        print(f"  - Trajectory length: {len(trajectories[0])} (expected {CONFIG['max_steps'] + 1})")
        print(f"  - Final time index: {trajectories[0][-1][3]} (expected {CONFIG['max_steps']})")
        
        return True


# ============================================================
# MAIN
# ============================================================

def main():
    print("="*60)
    print("GATE A1: ENVIRONMENT SANITY CHECKS")
    print("="*60)
    
    results = {}
    
    # A1.1: Reachability
    reachable, reach_results = reachability_check()
    results['A1.1_reachable'] = reachable
    
    # A1.2: Simple controllers
    has_better_controller, controller_results = simple_controller_baseline()
    results['A1.2_better_controller'] = has_better_controller
    
    # A1.3: Metric sanity
    metrics_ok = metric_sanity_check()
    results['A1.3_metrics_ok'] = metrics_ok
    
    # Gate A1 status
    print("\n" + "="*60)
    print("GATE A1 STATUS")
    print("="*60)
    
    all_pass = reachable and has_better_controller and metrics_ok
    
    print(f"""
A1.1 Reachability: {'✅ PASS' if reachable else '❌ FAIL'}
A1.2 Simple controllers: {'✅ PASS' if has_better_controller else '❌ FAIL'}
A1.3 Metric sanity: {'✅ PASS' if metrics_ok else '❌ FAIL'}

GATE A1: {'✅ PASS' if all_pass else '❌ FAIL'}
""")
    
    if not all_pass:
        print("STOP: Environment or task definition has issues.")
        print("Fix these before proceeding to MPPI validation.")
    else:
        print("Environment is valid. Proceed to Gate A2 (MPPI verification).")
    
    # Save results
    os.makedirs('/home/ulluboz/pcp-jepa-research/forensics', exist_ok=True)
    with open('/home/ulluboz/pcp-jepa-research/forensics/gate_A1_results.json', 'w') as f:
        json.dump({
            'pass': all_pass,
            'results': results,
            'reachability': reach_results,
            'controllers': controller_results
        }, f, indent=2)
    
    return all_pass

if __name__ == '__main__':
    main()