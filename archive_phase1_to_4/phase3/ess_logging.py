#!/usr/bin/env python3
"""
PHASE 3: ESS / Weight-Collapse Logging
Measures MPPI behavior near event boundaries to understand the mechanism.

Logs per MPPI iteration:
- J_min, J_mean, J_std (cost statistics)
- w_max (max weight) 
- ESS = 1 / sum(w^2) (effective sample size)
- ESS_frac = ESS / N
- entropy = -sum(w * log(w+eps))
- topk_mass for k in {1, 5, 10}

Event proximity signals:
- event_now: 1 if impact at current step
- dist_to_next_event: steps until next impact
- prox: exp(-dist_to_next_event / τ) with τ=5
"""

import numpy as np
import json
import os
from collections import defaultdict

# ============== ENVIRONMENT ==============
class BouncingBall:
    """1D bouncing ball environment - matching gate_A1 physics."""
    
    def __init__(self, x0=None, v0=None, g=0.0, e=0.8, dt=0.05):
        self.g = g  # 0.0 to match gate_A1 (no gravity)
        self.e = e
        self.dt = dt
        self.x_target = 2.0
        self.tau = 0.3  # Match gate_A1
        
    def reset(self, seed=None):
        if seed is not None:
            np.random.seed(seed)
        # Match gate_A1: start positions from [0.5-3.5] with v=0.0
        start_positions = [0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5]
        self.x = start_positions[seed % len(start_positions)]
        self.v = 0.0
        return np.array([self.x, self.v])
    
    def step(self, a):
        """Step physics. a is scalar action (force)."""
        # Apply gravity + action
        self.v += (-self.g + a) * self.dt
        self.x += self.v * self.dt
        
        # Ground bounce
        if self.x < 0:
            self.x = -self.x * self.e  # reflect and dampen
            self.v = -self.v * self.e
            
        # Wall bounce (at x=3)
        if self.x > 3:
            self.x = 3 - (self.x - 3) * self.e
            self.v = -self.v * self.e
            
        return np.array([self.x, self.v])
    
    def predict_rollout(self, actions):
        """Predict trajectory without executing. Returns states and event info."""
        x, v = self.x, self.v
        states = [np.array([x, v])]
        event_flags = []
        
        for a in actions:
            v += (-self.g + a) * self.dt
            x += v * self.dt
            
            # Check for bounce
            event = 0
            if x < 0:
                event = 1
                x = -x * self.e
                v = -v * self.e
            elif x > 3:
                event = 1
                x = 3 - (x - 3) * self.e
                v = -v * self.e
                
            states.append(np.array([x, v]))
            event_flags.append(event)
            
        return states, event_flags
    
    def dist_to_next_event(self, action=0, lookahead=50):
        """Compute steps until next event (bounce)."""
        x, v = self.x, self.v
        
        for step in range(lookahead):
            v_new = v + (-self.g + action) * self.dt
            x_new = x + v_new * self.dt
            
            # Check for event
            if x_new < 0 or x_new > 3:
                return step + 1
                
            x, v = x_new, v_new
            
        return lookahead  # no event within horizon


# ============== TRUE PHYSICS (ORACLE) ==============
def true_physics_step(x, v, a, g=9.81, e=0.8, dt=0.05):
    """Oracle physics for planning."""
    v_new = v + (-g + a) * dt
    x_new = x + v_new * dt
    
    event = 0
    if x_new < 0:
        event = 1
        x_new = -x_new * e
        v_new = -v_new * e
    elif x_new > 3:
        event = 1
        x_new = 3 - (x_new - 3) * e
        v_new = -v_new * e
        
    return x_new, v_new, event


def true_physics_rollout(x, v, actions, g=9.81, e=0.8, dt=0.05):
    """Rollout with true physics. Returns states, events, costs."""
    states = [(x, v)]
    events = []
    costs = []
    
    for a in actions:
        x, v, ev = true_physics_step(x, v, a, g, e, dt)
        states.append((x, v))
        events.append(ev)
        
        # Cost: distance to target
        cost = (x - 2.0) ** 2
        costs.append(cost)
        
    # Terminal cost
    final_x = states[-1][0]
    terminal_cost = (final_x - 2.0) ** 2
    total_cost = sum(costs) + terminal_cost
    
    return states, events, total_cost


# ============== MPPI WITH ESS LOGGING ==============
def mppi_with_ess_logging(x, v, horizon=30, n_samples=64, temperature=1.0, 
                          action_scale=2.0, beta=0.8):
    """
    MPPI with comprehensive ESS logging.
    Returns action and diagnostic dict.
    """
    
    # Sample actions
    actions = np.random.uniform(-action_scale, action_scale, (n_samples, horizon))
    
    # Compute costs for each sample
    costs = []
    all_events = []
    all_states = []
    
    for i in range(n_samples):
        states, events, total_cost = true_physics_rollout(x, v, actions[i])
        costs.append(total_cost)
        all_events.append(events)
        all_states.append(states)
        
    costs = np.array(costs)
    
    # MPPI weights
    min_cost = costs.min()
    costs_shifted = costs - min_cost
    weights = np.exp(-costs_shifted / temperature)
    weights = weights / weights.sum()
    
    # ========== ESS METRICS ==========
    w_max = weights.max()
    ess = 1.0 / (weights ** 2).sum()
    ess_frac = ess / n_samples
    
    # Entropy
    eps = 1e-10
    entropy = -np.sum(weights * np.log(weights + eps))
    
    # Top-k mass
    sorted_weights = np.sort(weights)[::-1]
    top1_mass = sorted_weights[0]
    top5_mass = sorted_weights[:5].sum()
    top10_mass = sorted_weights[:10].sum()
    
    # Cost statistics
    J_min = costs.min()
    J_mean = costs.mean()
    J_std = costs.std()
    
    # Bounce fraction
    bounce_fracs = [sum(ev) for ev in all_events]
    mean_bounce_frac = np.mean(bounce_fracs)
    
    # Collapse flags
    collapse_ess = ess_frac < 0.05
    collapse_weight = w_max > 0.5
    
    # Package diagnostics
    diagnostics = {
        'J_min': float(J_min),
        'J_mean': float(J_mean),
        'J_std': float(J_std),
        'w_max': float(w_max),
        'ess': float(ess),
        'ess_frac': float(ess_frac),
        'entropy': float(entropy),
        'top1_mass': float(top1_mass),
        'top5_mass': float(top5_mass),
        'top10_mass': float(top10_mass),
        'mean_bounce_frac': float(mean_bounce_frac),
        'collapse_ess': bool(collapse_ess),
        'collapse_weight': bool(collapse_weight),
    }
    
    # Compute action
    actions_weighted = actions.T @ weights
    action = actions_weighted[0]  # first action only
    
    return action, diagnostics


# ============== RUN EXPERIMENT ==============
def run_ess_experiment(n_episodes=100, horizon=30, n_samples=64):
    """Run MPPI with ESS logging across episodes."""
    
    env = BouncingBall()
    
    # Storage
    all_diagnostics = []
    episode_results = []
    
    for ep in range(n_episodes):
        obs = env.reset(seed=ep)
        x, v = obs[0], obs[1]
        
        episode_diagnostics = []
        success = False
        catastrophics = 0
        
        for step in range(50):  # max steps
            # Event proximity
            dist_event = env.dist_to_next_event()
            prox = np.exp(-dist_event / 5.0)
            
            # MPPI with logging
            action, diag = mppi_with_ess_logging(
                x, v, horizon=horizon, n_samples=n_samples
            )
            
            # Add event proximity to diagnostics
            diag['dist_to_event'] = dist_event
            diag['prox'] = prox
            diag['step'] = step
            
            # Check for event at current step (need to step to know)
            # We'll log this after stepping
            episode_diagnostics.append(diag)
            
            # Execute
            obs = env.step(action)
            x, v = obs[0], obs[1]
            
            # Check success/catastrophic
            if abs(x - env.x_target) < env.tau:
                success = True
            if x < -1 or x > 4:
                catastrophics += 1
                
        episode_results.append({
            'success': success,
            'catastrophics': catastrophics,
            'final_x': x,
        })
        
        all_diagnostics.extend(episode_diagnostics)
        
        if (ep + 1) % 20 == 0:
            print(f"Episode {ep+1}/{n_episodes} complete")
            
    return all_diagnostics, episode_results


# ============== ANALYZE RESULTS ==============
def analyze_ess_results(diagnostics, results):
    """Analyze ESS behavior near events."""
    
    # Convert to arrays
    prox = np.array([d['prox'] for d in diagnostics])
    ess_frac = np.array([d['ess_frac'] for d in diagnostics])
    w_max = np.array([d['w_max'] for d in diagnostics])
    J_std = np.array([d['J_std'] for d in diagnostics])
    dist_event = np.array([d['dist_to_event'] for d in diagnostics])
    
    # Bin by proximity
    bins = [0, 0.2, 0.4, 0.6, 0.8, 1.0]
    bin_labels = ['0-0.2', '0.2-0.4', '0.4-0.6', '0.6-0.8', '0.8-1.0']
    
    results_by_bin = {}
    for i, label in enumerate(bin_labels):
        mask = (prox >= bins[i]) & (prox < bins[i+1])
        if mask.sum() > 0:
            results_by_bin[label] = {
                'ess_frac_mean': ess_frac[mask].mean(),
                'ess_frac_std': ess_frac[mask].std(),
                'w_max_mean': w_max[mask].mean(),
                'J_std_mean': J_std[mask].mean(),
                'count': mask.sum(),
            }
    
    # Success by ESS
    # Need to correlate ESS with eventual success - this is per-episode
    # For now, just report aggregate
    
    analysis = {
        'results_by_proximity_bin': results_by_bin,
        'overall': {
            'ess_frac_mean': ess_frac.mean(),
            'ess_frac_std': ess_frac.std(),
            'w_max_mean': w_max.mean(),
            'J_std_mean': J_std.mean(),
        },
        'n_samples': len(diagnostics),
    }
    
    return analysis


def main():
    print("="*70)
    print("PHASE 3: ESS / WEIGHT-COLLAPSE LOGGING")
    print("="*70)
    
    # Run experiment
    print("\nRunning MPPI with ESS logging...")
    diagnostics, results = run_ess_experiment(n_episodes=100, horizon=30, n_samples=64)
    
    print(f"\nCollected {len(diagnostics)} diagnostic samples")
    
    # Analyze
    print("\nAnalyzing ESS behavior...")
    analysis = analyze_ess_results(diagnostics, results)
    
    # Print results
    print("\n" + "="*70)
    print("ESS ANALYSIS BY EVENT PROXIMITY")
    print("="*70)
    
    print("\nProximity bin | ESS_frac | w_max | J_std | Count")
    print("-"*55)
    for bin_label, stats in analysis['results_by_proximity_bin'].items():
        print(f"{bin_label:14} | {stats['ess_frac_mean']:.3f} ± {stats['ess_frac_std']:.3f} | "
              f"{stats['w_max_mean']:.3f} | {stats['J_std_mean']:.1f} | {stats['count']}")
    
    print(f"\nOverall: ESS_frac={analysis['overall']['ess_frac_mean']:.3f}, "
          f"w_max={analysis['overall']['w_max_mean']:.3f}")
    
    # Success rate
    success_rate = sum(r['success'] for r in results) / len(results)
    catastrophic_rate = sum(r['catastrophics'] for r in results) / len(results)
    
    print(f"\nEpisode results:")
    print(f"  Success rate: {success_rate:.1%}")
    print(f"  Catastrophic rate: {catastrophic_rate:.1%}")
    
    # Save results
    output_dir = '/home/ulluboz/pcp-jepa-research/results/phase3'
    os.makedirs(output_dir, exist_ok=True)
    
    # Save full diagnostics
    with open(f'{output_dir}/ess_diagnostics.json', 'w') as f:
        # Convert to serializable
        serializable = []
        for d in diagnostics:
            serializable.append({k: float(v) if isinstance(v, (np.floating, np.integer)) else v 
                               for k, v in d.items()})
        json.dump(serializable, f, indent=2)
    
    # Save analysis - convert numpy types
    def convert_numpy(obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, (np.floating, np.integer)):
            return float(obj)
        elif isinstance(obj, dict):
            return {k: convert_numpy(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [convert_numpy(i) for i in obj]
        return obj
    
    with open(f'{output_dir}/ess_analysis.json', 'w') as f:
        json.dump(convert_numpy(analysis), f, indent=2)
        
    print(f"\nSaved to {output_dir}/")
    
    # Expected signature check
    print("\n" + "="*70)
    print("MECHANISM CHECK")
    print("="*70)
    
    # Check if ESS drops near events
    low_prox = analysis['results_by_proximity_bin'].get('0-0.2', {})
    high_prox = analysis['results_by_proximity_bin'].get('0.8-1.0', {})
    
    if low_prox and high_prox:
        ess_drop = low_prox['ess_frac_mean'] - high_prox['ess_frac_mean']
        print(f"ESS drop near events (low prox → high prox): {ess_drop:.3f}")
        
        if ess_drop > 0.1:
            print("✅ SIGNATURE CONFIRMED: ESS collapses near events")
        else:
            print("⚠️ No clear ESS collapse signature (may need more data)")
    
    return analysis


if __name__ == '__main__':
    main()