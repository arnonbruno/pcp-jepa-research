#!/usr/bin/env python3
"""
PHASE A: FORENSIC VALIDATION
Gate A1 - Environment Sanity Checks (Fixed JIT)

A1.1: Reachability check
A1.2: Simple controller baseline
A1.3: Metric sanity
"""

import jax
import jax.numpy as jnp
import numpy as np
import json
import pickle
import os

# Load config
with open('/home/ulluboz/pcp-jepa-research/forensics/config.json', 'r') as f:
    CONFIG = json.load(f)

# ============================================================
# PHYSICS (non-JIT for simplicity)
# ============================================================

def physics_step(obs, a, dt=0.05, restitution=0.8):
    """Physics step (non-JIT for flexibility)."""
    x, v = float(obs[0]), float(obs[1])
    a = float(a)
    
    v_new = v + a * dt
    x_new = x + v_new * dt
    
    # Bounces
    if x_new < 0:
        x_new = -x_new
        v_new = -v_new * restitution
    if x_new > 4:
        x_new = 8 - x_new
        v_new = -v_new * restitution
    
    return np.array([x_new, v_new])

# ============================================================
# A1.1: REACHABILITY CHECK
# ============================================================

def reachability_check():
    """Compute reachable x-range via sampling."""
    print("\n" + "="*60)
    print("A1.1: REACHABILITY CHECK")
    print("="*60)
    
    configs = [
        {'horizon': 10, 'action_scale': 1.0},
        {'horizon': 20, 'action_scale': 1.0},
        {'horizon': 30, 'action_scale': 1.0},
        {'horizon': 50, 'action_scale': 1.0},
        {'horizon': 50, 'action_scale': 2.0},
        {'horizon': 100, 'action_scale': 2.0},
    ]
    
    start_positions = [0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5]
    target = CONFIG['x_target']
    n_samples = 10000
    
    results = []
    reachable_configs = []
    
    print(f"\nTarget: x = {target}")
    print(f"\n{'Horizon':<8} {'Scale':<6} {'Start':<6} {'Min x':<8} {'Max x':<8} {'Reach?':<8}")
    print("-" * 55)
    
    np.random.seed(42)
    
    for cfg in configs:
        H = cfg['horizon']
        scale = cfg['action_scale']
        
        for x0 in start_positions:
            final_xs = []
            
            for _ in range(n_samples):
                x, v = x0, 0.0
                for t in range(H):
                    a = np.random.uniform(-scale, scale)
                    obs = np.array([x, v])
                    obs = physics_step(obs, a)
                    x, v = obs[0], obs[1]
                final_xs.append(x)
            
            x_min = np.percentile(final_xs, 2.5)
            x_max = np.percentile(final_xs, 97.5)
            can_reach = x_min <= target <= x_max
            
            print(f"{H:<8} {scale:<6.1f} {x0:<6.1f} {x_min:<8.2f} {x_max:<8.2f} {'✅' if can_reach else '❌':<8}")
            
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
    
    unique_reachable = {f"H={c['horizon']}, scale={c['action_scale']}" for c in reachable_configs}
    
    print("\n" + "="*60)
    print("REACHABILITY SUMMARY")
    print("="*60)
    
    if unique_reachable:
        print(f"\nConfigs that CAN reach target:")
        for cfg_str in sorted(unique_reachable):
            print(f"  - {cfg_str}")
        print("\n✅ PASS: Target is reachable")
        return True, results
    else:
        print("\n❌ FAIL: Target NOT reachable with any config")
        return False, results


# ============================================================
# A1.2: SIMPLE CONTROLLER BASELINE
# ============================================================

def simple_controller_baseline():
    """Test if simple controllers beat random."""
    print("\n" + "="*60)
    print("A1.2: SIMPLE CONTROLLER BASELINE")
    print("="*60)
    
    def p_controller(obs, k=2.0, a_max=1.0):
        x, v = float(obs[0]), float(obs[1])
        a = np.clip(k * (CONFIG['x_target'] - x), -a_max, a_max)
        return a
    
    def bangbang_controller(obs, damping_threshold=0.5, a_max=1.0):
        x, v = float(obs[0]), float(obs[1])
        error = CONFIG['x_target'] - x
        
        if abs(error) < damping_threshold:
            a = -np.sign(v) * a_max * 0.5
        else:
            a = np.sign(error) * a_max
        
        return a
    
    def run_episode(policy_fn, seed):
        np.random.seed(seed)
        x = np.random.uniform(0.5, 3.5)
        v = np.random.uniform(-0.5, 0.5)
        
        for _ in range(CONFIG['max_steps']):
            obs = np.array([x, v])
            a = policy_fn(obs)
            obs = physics_step(obs, a)
            x, v = obs[0], obs[1]
        
        miss = abs(x - CONFIG['x_target'])
        return miss < CONFIG['tau_success'], miss
    
    n_episodes = CONFIG['n_episodes']
    
    policies = {
        'random': lambda obs: np.random.uniform(-1.0, 1.0),
        'p_k1.0': lambda obs: p_controller(obs, k=1.0),
        'p_k2.0': lambda obs: p_controller(obs, k=2.0),
        'p_k5.0': lambda obs: p_controller(obs, k=5.0),
        'bangbang': lambda obs: bangbang_controller(obs),
    }
    
    results = {}
    
    for name, policy_fn in policies.items():
        print(f"\nTesting {name}...")
        
        successes = 0
        misses = []
        
        for i in range(n_episodes):
            success, miss = run_episode(policy_fn, seed=i)
            if success:
                successes += 1
            misses.append(miss)
        
        sr = successes / n_episodes
        avg_miss = np.mean(misses)
        
        results[name] = {'success_rate': sr, 'avg_miss': avg_miss}
        print(f"  Success: {sr:.1%}, Avg miss: {avg_miss:.3f}")
    
    random_sr = results['random']['success_rate']
    better = [name for name, r in results.items() if r['success_rate'] > random_sr + 0.05]
    
    print("\n" + "="*60)
    print("CONTROLLER COMPARISON")
    print("="*60)
    print(f"\nRandom baseline: {random_sr:.1%}")
    print(f"\nPolicies beating random by >5%:")
    for name in better:
        print(f"  - {name}: {results[name]['success_rate']:.1%} (+{results[name]['success_rate'] - random_sr:.1%})")
    
    if better:
        print("\n✅ PASS: Simple controllers beat random")
        return True, results
    else:
        print("\n❌ FAIL: No controller beats random")
        return False, results


# ============================================================
# A1.3: METRIC SANITY
# ============================================================

def metric_sanity_check():
    """Verify metrics from saved trajectories."""
    print("\n" + "="*60)
    print("A1.3: METRIC SANITY CHECK")
    print("="*60)
    
    with open('/home/ulluboz/pcp-jepa-research/forensics/random_trajectories.pkl', 'rb') as f:
        trajectories = pickle.load(f)
    
    with open('/home/ulluboz/pcp-jepa-research/forensics/random_metadata.pkl', 'rb') as f:
        original_metadata = pickle.load(f)
    
    print(f"\nLoaded {len(trajectories)} trajectories")
    
    errors = []
    
    for i, (traj, orig_meta) in enumerate(zip(trajectories, original_metadata)):
        final_x, final_v, final_a, final_t = traj[-1]
        
        miss = abs(final_x - CONFIG['x_target'])
        success = miss < CONFIG['tau_success']
        catastrophic = miss > CONFIG['tau_catastrophic'] or abs(final_v) > CONFIG['v_max']
        
        if abs(miss - orig_meta['miss_distance']) > 1e-6:
            errors.append(f"Episode {i}: miss mismatch")
        if success != orig_meta['success']:
            errors.append(f"Episode {i}: success mismatch")
    
    if errors:
        print(f"\n❌ {len(errors)} errors found")
        return False
    else:
        print(f"\n✅ All {len(trajectories)} trajectories verified")
        return True


# ============================================================
# MAIN
# ============================================================

def main():
    print("="*60)
    print("GATE A1: ENVIRONMENT SANITY CHECKS")
    print("="*60)
    
    # A1.1
    reachable, reach_results = reachability_check()
    
    # A1.2
    has_controller, ctrl_results = simple_controller_baseline()
    
    # A1.3
    metrics_ok = metric_sanity_check()
    
    # Summary
    all_pass = reachable and has_controller and metrics_ok
    
    print("\n" + "="*60)
    print("GATE A1 STATUS")
    print("="*60)
    print(f"""
A1.1 Reachability: {'✅ PASS' if reachable else '❌ FAIL'}
A1.2 Controllers: {'✅ PASS' if has_controller else '❌ FAIL'}
A1.3 Metrics: {'✅ PASS' if metrics_ok else '❌ FAIL'}

GATE A1: {'✅ PASS' if all_pass else '❌ FAIL'}
""")
    
    os.makedirs('/home/ulluboz/pcp-jepa-research/forensics', exist_ok=True)
    with open('/home/ulluboz/pcp-jepa-research/forensics/gate_A1_results.json', 'w') as f:
        json.dump({
            'pass': all_pass,
            'reachable': reachable,
            'has_better_controller': has_controller,
            'metrics_ok': metrics_ok
        }, f, indent=2)
    
    return all_pass

if __name__ == '__main__':
    main()