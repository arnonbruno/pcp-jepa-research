#!/usr/bin/env python3
"""
PHASE 2: Perception Gap - Baselines

Evaluates performance of different controllers across observation regimes:
- O0: Full state (upper bound)
- O1: Partial state (position only)
- O2: Pixels (64x64)

This quantifies how hard the "JEPA bridge" really is.
"""

import numpy as np
from observation_regimes import BouncingBallGravity, ObservationModel, generate_trajectories


# ============================================================================
# CONTROLLERS
# ============================================================================

class Controller:
    """Base controller."""
    def __init__(self):
        pass
    
    def reset(self):
        pass
    
    def act(self, obs, target=2.0):
        raise NotImplementedError


class PDController(Controller):
    """PD on full state."""
    def __init__(self, k1=1.5, k2=-2.0):
        self.k1 = k1
        self.k2 = k2
    
    def act(self, obs, target=2.0):
        # obs = [x, v]
        x, v = obs[0], obs[1]
        a = self.k1 * (target - x) + self.k2 * (-v)
        return np.clip(a, -2.0, 2.0)


class PartialObsController(Controller):
    """PD on partial state - needs velocity estimation."""
    def __init__(self, estimator=None):
        self.estimator = estimator  # velocity estimator
    
    def act(self, obs, target=2.0):
        x = obs[0]  # only position
        
        # Estimate velocity (simple: finite difference or zero)
        if self.estimator is not None:
            v = self.estimator.estimate(obs)
        else:
            v = 0.0  # naive baseline
        
        a = 1.5 * (target - x) + (-2.0) * (-v)
        return np.clip(a, -2.0, 2.0)


class VelocityEstimator:
    """Simple velocity estimator from position sequence."""
    
    def __init__(self, window=3):
        self.window = window
        self.pos_history = []
    
    def estimate(self, obs):
        """Estimate velocity from recent positions."""
        self.pos_history.append(obs[0])
        if len(self.pos_history) > self.window:
            self.pos_history.pop(0)
        
        if len(self.pos_history) < 2:
            return 0.0
        
        # Finite difference
        dt = 0.05
        v = (self.pos_history[-1] - self.pos_history[0]) / (len(self.pos_history) * dt)
        return v
    
    def reset(self):
        self.pos_history = []


class OpenLoopPlanner:
    """Simple CEM planner using learned dynamics."""
    
    def __init__(self, n_samples=32, horizon=10):
        self.n_samples = n_samples
        self.horizon = horizon
    
    def act(self, obs, target=2.0, model=None):
        # Sample actions and evaluate
        best_a = 0.0
        best_cost = float('inf')
        
        for _ in range(self.n_samples):
            a = np.random.uniform(-2.0, 2.0)
            
            # Simple rollouts
            x, v = obs[0], obs[1]
            cost = 0
            for _ in range(self.horizon):
                a_clip = np.clip(a, -2.0, 2.0)
                v += (-9.81 + a_clip) * 0.05
                x += v * 0.05
                if x < 0:
                    x = -x * 0.8
                    v = -v * 0.8
                if x > 3:
                    x = 3 - (x - 3) * 0.8
                    v = -v * 0.8
                cost += (x - target) ** 2
            
            if cost < best_cost:
                best_cost = cost
                best_a = a
        
        return np.clip(best_a, -2.0, 2.0)


# ============================================================================
# EVALUATION
# ============================================================================

def evaluate_controller(
    controller,
    observation='full',
    n_episodes=200,
    restitution=0.8,
    seed=0,
    use_estimator=False
):
    """Evaluate a controller in a given observation regime."""
    
    np.random.seed(seed)
    obs_model = ObservationModel(regime=observation)
    env = BouncingBallGravity(restitution=restitution)
    
    # Special handling for estimators
    if hasattr(controller, 'estimator') and controller.estimator is not None:
        estimator = controller.estimator
    else:
        estimator = None
    
    successes = 0
    terminal_misses = []
    
    for ep in range(n_episodes):
        state = env.reset(seed=ep + seed)
        x, v = state[0], state[1]
        
        if estimator:
            estimator.reset()
        
        for step in range(30):
            # Get observation
            observation_arr = obs_model.observe(state)
            
            # Controller action
            if hasattr(controller, 'estimator') and observation == 'partial':
                # Need to handle partial observation specially
                if estimator:
                    # Use history for velocity estimation
                    v_est = estimator.estimate(observation_arr)
                    full_obs = np.array([observation_arr[0], v_est])
                else:
                    full_obs = np.array([observation_arr[0], 0.0])
                a = controller.act(full_obs)
            else:
                a = controller.act(observation_arr)
            
            # Step
            state, _ = env.step(a)
            x, v = state[0], state[1]
            
            if abs(x - env.x_target) < env.tau:
                successes += 1
                break
        
        terminal_misses.append(abs(x - env.x_target))
    
    return {
        'success_rate': successes / n_episodes,
        'terminal_miss_mean': np.mean(terminal_misses),
        'terminal_miss_std': np.std(terminal_misses),
    }


def run_perception_gap_experiment():
    """Main experiment: success rate vs observation regime."""
    print("="*70)
    print("PHASE 2: PERCEPTION GAP")
    print("="*70)
    
    # Configuration
    n_episodes = 200
    restitution = 0.8  # Hard regime
    
    results = []
    
    # Controllers to test
    controllers = [
        ('PD (full state)', 'full', PDController()),
        ('PD (no v)', 'partial', PartialObsController(estimator=None)),
        ('PD + vel est', 'partial', PartialObsController(estimator=VelocityEstimator())),
        ('Open-loop CEM', 'full', OpenLoopPlanner()),
    ]
    
    for name, obs_regime, controller in controllers:
        print(f"\n{name} ({obs_regime}):")
        
        result = evaluate_controller(
            controller,
            observation=obs_regime,
            n_episodes=n_episodes,
            restitution=restitution,
            seed=42
        )
        
        print(f"  Success: {result['success_rate']:.1%}")
        print(f"  Terminal miss: {result['terminal_miss_mean']:.3f} ± {result['terminal_miss_std']:.3f}")
        
        results.append({
            'name': name,
            'regime': obs_regime,
            **result
        })
    
    # Summary table
    print("\n" + "="*70)
    print("SUMMARY: Perception Gap")
    print("="*70)
    print(f"{'Controller':<25} {'Regime':<10} {'Success':>10} {'Miss':>12}")
    print("-"*60)
    for r in results:
        print(f"{r['name']:<25} {r['regime']:<10} {r['success_rate']:>9.1%} {r['terminal_miss_mean']:>10.3f}")
    
    # Gap analysis
    pd_full = [r for r in results if 'full' in r['regime'] and 'PD' in r['name']][0]
    pd_partial = [r for r in results if r['regime'] == 'partial' and 'vel est' in r['name']][0]
    
    gap = pd_full['success_rate'] - pd_partial['success_rate']
    print(f"\nPerception gap (full vs partial+est): {gap:.1%}")
    
    return results


def sweep_restitution_perception():
    """How does perception gap vary with restitution?"""
    print("\n" + "="*70)
    print("SWEEP: Perception Gap vs Restitution")
    print("="*70)
    
    controllers = [
        ('PD_full', 'full', PDController()),
        ('PD_partial', 'partial', PartialObsController(estimator=VelocityEstimator())),
    ]
    
    for e in [0.3, 0.5, 0.7, 0.8, 0.9, 0.95]:
        print(f"\ne = {e}:")
        
        for name, obs, ctrl in controllers:
            result = evaluate_controller(
                ctrl, observation=obs, n_episodes=200, restitution=e, seed=42
            )
            print(f"  {name}: {result['success_rate']:.1%}")


if __name__ == '__main__':
    results = run_perception_gap_experiment()
    sweep_restitution_perception()
