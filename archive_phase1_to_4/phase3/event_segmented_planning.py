#!/usr/bin/env python3
"""
PHASE 3: Event-Segmented Planning (The Fix)

MPPI struggles when rollouts straddle discontinuities. 
Event-segmented planning: plan only until next event, execute, then replan.

Compares 4 planners:
1. MPPI (standard) with H=50
2. MPPI with event-segmented (adaptive horizon)
3. P-controller / bang-bang (strong feedback baseline)
4. Random

Metrics: success rate, miss distance, catastrophics, ESS near events
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
        self.v += (-self.g + a) * self.dt
        self.x += self.v * self.dt
        
        if self.x < 0:
            self.x = -self.x * self.e
            self.v = -self.v * self.e
        if self.x > 3:
            self.x = 3 - (self.x - 3) * self.e
            self.v = -self.v * self.e
            
        return np.array([self.x, self.v])
    
    def predict_rollout(self, actions):
        """Predict trajectory without executing."""
        x, v = self.x, self.v
        states = [np.array([x, v])]
        event_flags = []
        
        for a in actions:
            v += (-self.g + a) * self.dt
            x += v * self.dt
            
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
        """Compute steps until next event."""
        x, v = self.x, self.v
        
        for step in range(lookahead):
            v_new = v + (-self.g + action) * self.dt
            x_new = x + v_new * self.dt
            
            if x_new < 0 or x_new > 3:
                return step + 1
                
            x, v = x_new, v_new
            
        return lookahead


# ============== TRUE PHYSICS (matching gate_A1) ==============
def true_physics_step(x, v, a, g=0.0, e=0.8, dt=0.05):
    # Match gate_A1: no gravity, just a*dt for velocity
    v_new = v + a * dt
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


def true_physics_rollout(x, v, actions, g=0.0, e=0.8, dt=0.05):
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


# ============== PLANNERS ==============

def random_policy():
    """Random action."""
    return np.random.uniform(-2.0, 2.0)


def p_controller(x, target=2.0, k=2.0):
    """P controller: a = k * (target - x), clipped to a_max"""
    a = k * (target - x)
    return np.clip(a, -1.0, 1.0)  # Match gate_A1: a_max=1.0


def bang_bang(x, target=2.0, scale=1.0):
    """Bang-bang: max thrust toward target."""
    if x < target:
        return scale
    else:
        return -scale


def mppi_standard(x, v, horizon=50, n_samples=64, temperature=1.0, action_scale=2.0):
    """Standard MPPI with fixed horizon."""
    
    actions = np.random.uniform(-action_scale, action_scale, (n_samples, horizon))
    
    costs = []
    for i in range(n_samples):
        _, _, total_cost = true_physics_rollout(x, v, actions[i])
        costs.append(total_cost)
        
    costs = np.array(costs)
    
    # MPPI weights
    min_cost = costs.min()
    costs_shifted = costs - min_cost
    weights = np.exp(-costs_shifted / temperature)
    weights = weights / weights.sum()
    
    # Compute action
    actions_weighted = actions.T @ weights
    action = actions_weighted[0]
    
    return action


def mppi_event_segmented(x, v, horizon=50, n_samples=64, temperature=1.0, 
                         action_scale=2.0, env=None):
    """
    Event-segmented MPPI:
    1. Predict time-to-next-event
    2. Plan only until event (capped)
    3. Execute first action
    4. Replan at next step
    """
    
    if env is None:
        env = BouncingBall()
        env.x, env.v = x, v
    
    # Compute adaptive horizon based on next event
    dist_event = env.dist_to_next_event()
    
    # Adaptive horizon: plan to just before the event
    # Min horizon = 3 steps, max = horizon
    k_e = min(max(dist_event - 1, 3), horizon)
    
    # Run MPPI with short horizon (event-free segment)
    actions = np.random.uniform(-action_scale, action_scale, (n_samples, int(k_e)))
    
    costs = []
    for i in range(n_samples):
        _, _, total_cost = true_physics_rollout(x, v, actions[i])
        costs.append(total_cost)
        
    costs = np.array(costs)
    
    # MPPI weights
    min_cost = costs.min()
    costs_shifted = costs - min_cost
    weights = np.exp(-costs_shifted / temperature)
    weights = weights / weights.sum()
    
    # Compute action
    actions_weighted = actions.T @ weights
    action = actions_weighted[0]
    
    # Return action + diagnostics
    diagnostics = {
        'adaptive_horizon': k_e,
        'dist_to_event': dist_event,
    }
    
    return action, diagnostics


def run_episode(env, planner, max_steps=50, **planner_kwargs):
    """Run one episode with given planner."""
    
    obs = env.reset(seed=planner_kwargs.get('seed', 0))
    x, v = obs[0], obs[1]
    
    # Track metrics
    success = False
    catastrophics = 0
    final_x = None
    trajectory = [(x, v)]
    ess_history = []
    
    for step in range(max_steps):
        # Get action from planner
        if planner == 'random':
            action = random_policy()
            diagnostics = {}
        elif planner == 'p_controller':
            action = p_controller(x)
            diagnostics = {}
        elif planner == 'bang_bang':
            action = bang_bang(x)
            diagnostics = {}
        elif planner == 'mppi_standard':
            action = mppi_standard(x, v, horizon=planner_kwargs.get('horizon', 50))
            diagnostics = {}
        elif planner == 'mppi_segmented':
            action, diagnostics = mppi_event_segmented(x, v, horizon=planner_kwargs.get('horizon', 50),
                                                        env=env)
        else:
            raise ValueError(f"Unknown planner: {planner}")
            
        # Track adaptive horizon if available
        if 'adaptive_horizon' in diagnostics:
            ess_history.append(diagnostics['adaptive_horizon'])
        
        # Execute
        obs = env.step(action)
        x, v = obs[0], obs[1]
        trajectory.append((x, v))
        
        # Check success/catastrophic
        if abs(x - env.x_target) < env.tau:
            success = True
        if x < -1 or x > 4:
            catastrophics += 1
            
    final_x = x
    
    return {
        'success': success,
        'catastrophics': catastrophics,
        'final_x': final_x,
        'miss': abs(final_x - env.x_target),
        'trajectory': trajectory,
        'adaptive_horizons': ess_history,
    }


def run_comparison(n_episodes=100):
    """Compare all planners."""
    
    planners = ['random', 'p_controller', 'bang_bang', 'mppi_standard', 'mppi_segmented']
    results = {p: [] for p in planners}
    
    print("="*70)
    print("EVENT-SEGMENTED PLANNING COMPARISON")
    print("="*70)
    
    for ep in range(n_episodes):
        for planner in planners:
            env = BouncingBall()
            kwargs = {'seed': ep}
            
            if planner in ['mppi_standard', 'mppi_segmented']:
                kwargs['horizon'] = 50
                
            result = run_episode(env, planner, **kwargs)
            results[planner].append(result)
            
        if (ep + 1) % 20 == 0:
            print(f"Episode {ep+1}/{n_episodes}")
            
    return results


def summarize_results(results):
    """Summarize and print results."""
    
    print("\n" + "="*70)
    print("RESULTS SUMMARY")
    print("="*70)
    
    summary = {}
    
    for planner, episodes in results.items():
        successes = sum(e['success'] for e in episodes)
        catastrophics = sum(e['catastrophics'] for e in episodes)
        misses = [e['miss'] for e in episodes]
        
        summary[planner] = {
            'success_rate': successes / len(episodes),
            'catastrophic_rate': catastrophics / len(episodes),
            'miss_mean': np.mean(misses),
            'miss_std': np.std(misses),
        }
        
        print(f"\n{planner}:")
        print(f"  Success rate: {summary[planner]['success_rate']:.1%}")
        print(f"  Catastrophic rate: {summary[planner]['catastrophic_rate']:.1%}")
        print(f"  Miss distance: {summary[planner]['miss_mean']:.3f} ± {summary[planner]['miss_std']:.3f}")
        
    return summary


def main():
    # Run comparison
    results = run_comparison(n_episodes=100)
    
    # Summarize
    summary = summarize_results(results)
    
    # Compare MPPI variants
    print("\n" + "="*70)
    print("MPPI STANDARD vs SEGMENTED")
    print("="*70)
    
    mppi_std = summary['mppi_standard']['success_rate']
    mppi_seg = summary['mppi_segmented']['success_rate']
    
    print(f"\nMPPI Standard:   {mppi_std:.1%}")
    print(f"MPPI Segmented:   {mppi_seg:.1%}")
    print(f"Improvement:      {mppi_seg - mppi_std:+.1%}")
    
    if mppi_seg > mppi_std:
        print("\n✅ Event-segmented MPPI improves success rate!")
    else:
        print("\n⚠️ No improvement from segmentation")
    
    # Save results
    output_dir = '/home/ulluboz/pcp-jepa-research/results/phase3'
    os.makedirs(output_dir, exist_ok=True)
    
    # Convert to serializable
    serializable = {}
    for planner, episodes in results.items():
        serializable[planner] = []
        for ep in episodes:
            serializable[planner].append({
                'success': ep['success'],
                'catastrophics': ep['catastrophics'],
                'final_x': float(ep['final_x']),
                'miss': float(ep['miss']),
            })
    
    with open(f'{output_dir}/event_segmented_comparison.json', 'w') as f:
        json.dump(serializable, f, indent=2)
        
    with open(f'{output_dir}/event_segmented_summary.json', 'w') as f:
        json.dump(summary, f, indent=2)
        
    print(f"\nSaved to {output_dir}/")
    
    return summary


if __name__ == '__main__':
    main()