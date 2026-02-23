#!/usr/bin/env python3
"""
PHASE 3: ESS / Weight-Collapse Logging - WITH GRAVITY
The harder physics where MPPI actually struggles.
"""

import numpy as np
import json
import os

# ============== ENVIRONMENT WITH GRAVITY ==============
class BouncingBallGravity:
    """1D bouncing ball with GRAVITY - the hard physics."""
    
    def __init__(self):
        self.g = 9.81  # GRAVITY
        self.e = 0.8
        self.dt = 0.05
        self.x_target = 2.0
        self.tau = 0.3
        
    def reset(self, seed=None):
        if seed is not None:
            np.random.seed(seed)
        # Start positions
        start_positions = [0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5]
        self.x = start_positions[seed % len(start_positions)]
        self.v = 0.0
        return np.array([self.x, self.v])
    
    def step(self, a):
        self.v += (-self.g + a) * self.dt
        self.x += self.v * self.dt
        
        if self.x < 0:
            self.x = -self.x * self.e
            self.v = -self.v * self.e
        if self.x > 3:
            self.x = 3 - (self.x - 3) * self.e
            self.v = -self.v * self.e
            
        return np.array([self.x, self.v])
    
    def dist_to_next_event(self, action=0, lookahead=50):
        """Compute steps until next event."""
        x, v = self.x, self.v
        
        for step in range(lookahead):
            v_new = v + (-self.g + action) * self.dt
            x_new = x + v_new * self.dt
            
            if x_new < 0 or x_new > 3:
                return step + 1
                
            x, v = x_new, v_new
            
        return lookahead


# ============== TRUE PHYSICS WITH GRAVITY ==============
def true_physics_step(x, v, a, g=9.81, e=0.8, dt=0.05):
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
    states = [(x, v)]
    events = []
    costs = []
    
    for a in actions:
        x, v, ev = true_physics_step(x, v, a, g, e, dt)
        states.append((x, v))
        events.append(ev)
        
        cost = (x - 2.0) ** 2
        costs.append(cost)
        
    final_x = states[-1][0]
    terminal_cost = (final_x - 2.0) ** 2
    total_cost = sum(costs) + terminal_cost
    
    return states, events, total_cost


# ============== MPPI WITH ESS LOGGING ==============
def mppi_with_ess_logging(x, v, horizon=30, n_samples=64, temperature=1.0, 
                          action_scale=2.0, g=9.81):
    """MPPI with comprehensive ESS logging - WITH GRAVITY."""
    
    actions = np.random.uniform(-action_scale, action_scale, (n_samples, horizon))
    
    costs = []
    all_events = []
    
    for i in range(n_samples):
        _, _, total_cost = true_physics_rollout(x, v, actions[i], g=g)
        costs.append(total_cost)
        
    costs = np.array(costs)
    
    # MPPI weights
    min_cost = costs.min()
    costs_shifted = costs - min_cost
    weights = np.exp(-costs_shifted / temperature)
    weights = weights / weights.sum()
    
    # ESS metrics
    w_max = weights.max()
    ess = 1.0 / (weights ** 2).sum()
    ess_frac = ess / n_samples
    
    eps = 1e-10
    entropy = -np.sum(weights * np.log(weights + eps))
    
    sorted_weights = np.sort(weights)[::-1]
    top1_mass = sorted_weights[0]
    top5_mass = sorted_weights[:5].sum()
    
    J_min = costs.min()
    J_mean = costs.mean()
    J_std = costs.std()
    
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
    }
    
    actions_weighted = actions.T @ weights
    action = actions_weighted[0]
    
    return action, diagnostics


def run_ess_experiment_gravity(n_episodes=100, horizon=30, n_samples=64):
    """Run MPPI with ESS logging - WITH GRAVITY."""
    
    env = BouncingBallGravity()
    
    all_diagnostics = []
    episode_results = []
    
    for ep in range(n_episodes):
        obs = env.reset(seed=ep)
        x, v = obs[0], obs[1]
        
        episode_diagnostics = []
        success = False
        
        for step in range(50):
            dist_event = env.dist_to_next_event()
            prox = np.exp(-dist_event / 5.0)
            
            action, diag = mppi_with_ess_logging(
                x, v, horizon=horizon, n_samples=n_samples, g=9.81
            )
            
            diag['dist_to_event'] = dist_event
            diag['prox'] = prox
            diag['step'] = step
            
            episode_diagnostics.append(diag)
            
            obs = env.step(action)
            x, v = obs[0], obs[1]
            
            if abs(x - env.x_target) < env.tau:
                success = True
                
        episode_results.append({'success': success, 'final_x': x})
        all_diagnostics.extend(episode_diagnostics)
        
        if (ep + 1) % 20 == 0:
            print(f"Episode {ep+1}/{n_episodes}")
            
    return all_diagnostics, episode_results


def analyze_ess_results(diagnostics, results):
    """Analyze ESS behavior near events."""
    
    prox = np.array([d['prox'] for d in diagnostics])
    ess_frac = np.array([d['ess_frac'] for d in diagnostics])
    w_max = np.array([d['w_max'] for d in diagnostics])
    J_std = np.array([d['J_std'] for d in diagnostics])
    
    bins = [0, 0.2, 0.4, 0.6, 0.8, 1.0]
    bin_labels = ['0-0.2', '0.2-0.4', '0.4-0.6', '0.6-0.8', '0.8-1.0']
    
    results_by_bin = {}
    for i, label in enumerate(bin_labels):
        mask = (prox >= bins[i]) & (prox < bins[i+1])
        if mask.sum() > 0:
            results_by_bin[label] = {
                'ess_frac_mean': float(ess_frac[mask].mean()),
                'ess_frac_std': float(ess_frac[mask].std()),
                'w_max_mean': float(w_max[mask].mean()),
                'J_std_mean': float(J_std[mask].mean()),
                'count': int(mask.sum()),
            }
    
    return {
        'results_by_proximity_bin': results_by_bin,
        'overall': {
            'ess_frac_mean': float(ess_frac.mean()),
            'w_max_mean': float(w_max.mean()),
        },
    }


def main():
    print("="*70)
    print("PHASE 3: ESS LOGGING WITH GRAVITY")
    print("="*70)
    
    print("\nRunning MPPI with ESS logging (GRAVITY)...")
    diagnostics, results = run_ess_experiment_gravity(n_episodes=100, horizon=30, n_samples=64)
    
    print(f"\nCollected {len(diagnostics)} samples")
    
    analysis = analyze_ess_results(diagnostics, results)
    
    print("\n" + "="*70)
    print("ESS ANALYSIS BY EVENT PROXIMITY (WITH GRAVITY)")
    print("="*70)
    
    print("\nProximity bin | ESS_frac | w_max | Count")
    print("-"*50)
    for label, stats in analysis['results_by_proximity_bin'].items():
        print(f"{label:14} | {stats['ess_frac_mean']:.3f} ± {stats['ess_frac_std']:.3f} | "
              f"{stats['w_max_mean']:.3f} | {stats['count']}")
    
    success_rate = sum(r['success'] for r in results) / len(results)
    print(f"\nSuccess rate: {success_rate:.1%}")
    
    # Save
    output_dir = '/home/ulluboz/pcp-jepa-research/results/phase3'
    os.makedirs(output_dir, exist_ok=True)
    
    with open(f'{output_dir}/ess_analysis_gravity.json', 'w') as f:
        json.dump(analysis, f, indent=2)
        
    print(f"\nSaved to {output_dir}/ess_analysis_gravity.json")
    
    # Check for ESS collapse near events
    low_prox = analysis['results_by_proximity_bin'].get('0-0.2', {})
    high_prox = analysis['results_by_proximity_bin'].get('0.8-1.0', {})
    
    if low_prox and high_prox:
        ess_drop = low_prox['ess_frac_mean'] - high_prox['ess_frac_mean']
        print(f"\nESS drop near events: {ess_drop:+.3f}")
        if ess_drop > 0.1:
            print("✅ ESS COLLAPSES near events (mechanism confirmed!)")
        elif ess_drop < -0.1:
            print("⚠️ ESS INCREASES near events (unexpected)")


if __name__ == '__main__':
    main()
