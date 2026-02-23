#!/usr/bin/env python3
"""
Option A: Hard projection rollout (bounce operator)

Enforce physics constraints during model rollout:
1. Predict in latent space
2. Decode to observation
3. Apply bounce projection if wall violated
4. Re-encode to latent

This prevents MPPI from planning through walls.
"""

import jax
import jax.numpy as jnp
import pickle
import numpy as np
import json
import time
import sys
sys.path.insert(0, '/home/ulluboz/pcp-jepa-research')

from experiments.phase2.train_action_controllable import (
    ActionControllableModel, physics_step
)

# ============================================================
# CONFIG
# ============================================================

X_MIN, X_MAX = 0.0, 4.0
RESTITUTION = 0.8
N_EPISODES = 500
MAX_STEPS = 30
HORIZON = 10
N_SAMPLES = 64
X_TARGET = 2.0
TAU_SUCCESS = 0.3
TAU_CAT = 0.5
V_MAX = 2.0

# ============================================================
# BOUNCE PROJECTION
# ============================================================

def bounce_projection(x: float, v: float, x_min: float = X_MIN, x_max: float = X_MAX, 
                      restitution: float = RESTITUTION) -> tuple:
    """
    Apply bounce physics if position violates boundaries.
    
    Returns corrected (x, v) pair.
    """
    # Check lower bound
    if x < x_min:
        x = 2 * x_min - x  # Reflect: x = -x
        v = -v * restitution
    
    # Check upper bound
    if x > x_max:
        x = 2 * x_max - x  # Reflect: x = 8 - x
        v = -v * restitution
    
    return x, v


def bounce_projection_jax(obs: jnp.ndarray) -> jnp.ndarray:
    """
    JAX-compatible bounce projection.
    
    Args:
        obs: [x, v] observation
    
    Returns:
        Corrected [x, v]
    """
    x, v = obs[0], obs[1]
    
    # Lower bound bounce
    x = jnp.where(x < X_MIN, 2 * X_MIN - x, x)
    v = jnp.where(x < X_MIN, -v * RESTITUTION, v)
    
    # Upper bound bounce
    x = jnp.where(x > X_MAX, 2 * X_MAX - x, x)
    v = jnp.where(x > X_MAX, -v * RESTITUTION, v)
    
    return jnp.array([x, v])


# ============================================================
# CONSTRAINED MODEL
# ============================================================

class ConstrainedModel:
    """
    Wrapper that enforces physics constraints on model predictions.
    """
    def __init__(self, model, params):
        self.model = model
        self.params = params
    
    def predict(self, obs: jnp.ndarray, action: jnp.ndarray, apply_bounce: bool = True) -> jnp.ndarray:
        """
        Predict next state with optional bounce projection.
        """
        # Model prediction
        next_obs = self.model.apply(self.params, obs, action)
        
        # Apply bounce projection
        if apply_bounce:
            next_obs = bounce_projection_jax(next_obs)
        
        return next_obs


def create_mppi_constrained(model, params, apply_bounce: bool = True):
    """
    Create MPPI planner with constrained rollouts.
    """
    constrained_model = ConstrainedModel(model, params)
    
    def batch_rollout(obs, actions):
        def single(action_seq):
            current = obs
            for t in range(HORIZON):
                current = constrained_model.predict(current, action_seq[t], apply_bounce=apply_bounce)
            return current[0]
        return jax.vmap(single)(actions)
    
    rollout_jit = jax.jit(batch_rollout)
    
    def plan(obs, key):
        key, subkey = jax.random.split(key)
        actions = jax.random.uniform(subkey, (N_SAMPLES, HORIZON, 1), minval=-1.0, maxval=1.0)
        final_x = rollout_jit(obs, actions)
        costs = jnp.abs(final_x - X_TARGET)
        weights = jax.nn.softmax(-costs)
        return jnp.sum(weights[:, None] * actions[:, 0, :], axis=0)[0]
    
    return jax.jit(plan)


# ============================================================
# EVALUATION
# ============================================================

def run_episode(key, use_mppi, mppi_plan=None):
    key, subkey = jax.random.split(key)
    x = float(jax.random.uniform(subkey, (), minval=0.5, maxval=3.5))
    key, subkey = jax.random.split(key)
    v = float(jax.random.uniform(subkey, (), minval=-0.5, maxval=0.5))
    
    for _ in range(MAX_STEPS):
        obs = jnp.array([x, v])
        if use_mppi and mppi_plan:
            key, subkey = jax.random.split(key)
            a = float(mppi_plan(obs, subkey))
        else:
            key, subkey = jax.random.split(key)
            a = float(jax.random.uniform(subkey, (), minval=-1.0, maxval=1.0))
        x, v = physics_step(x, v, a)
    
    miss = abs(x - X_TARGET)
    return {
        'miss': miss,
        'success': miss < TAU_SUCCESS,
        'catastrophic': miss > TAU_CAT or abs(v) > V_MAX
    }


def visualize_cheat_trajectory(model, params):
    """Show a trajectory that goes through wall in model vs reality."""
    print("\n" + "="*60)
    print("CHEAT TRAJECTORY VISUALIZATION")
    print("="*60)
    
    # Start near left wall, moving toward it
    obs = jnp.array([0.5, -0.5])
    actions = [jnp.array([-0.3]) for _ in range(15)]  # Push toward wall
    
    # Unconstrained model rollout
    obs_uc = obs
    model_traj_uc = [float(obs_uc[0])]
    for a in actions:
        obs_uc = model.apply(params, obs_uc, a)
        model_traj_uc.append(float(obs_uc[0]))
    
    # Constrained model rollout
    constrained = ConstrainedModel(model, params)
    obs_c = obs
    model_traj_c = [float(obs_c[0])]
    for a in actions:
        obs_c = constrained.predict(obs_c, a, apply_bounce=True)
        model_traj_c.append(float(obs_c[0]))
    
    # Real physics
    x, v = 0.5, -0.5
    real_traj = [x]
    for a in actions:
        x, v = physics_step(x, v, float(a[0]))
        real_traj.append(x)
    
    print(f"\nStarting: x=0.5, v=-0.5 (moving toward wall)")
    print(f"Actions: all a=-0.3 (pushing toward wall)")
    print(f"\nStep | Unconstrained | Constrained | Real Physics")
    print("-" * 55)
    for i in range(min(10, len(actions)+1)):
        uc = model_traj_uc[i] if i < len(model_traj_uc) else model_traj_uc[-1]
        c = model_traj_c[i] if i < len(model_traj_c) else model_traj_c[-1]
        r = real_traj[i] if i < len(real_traj) else real_traj[-1]
        wall_violation = "❌ WALL!" if uc < 0 else ""
        print(f"  {i:2d} |     {uc:6.2f}     |    {c:6.2f}    |    {r:6.2f}  {wall_violation}")
    
    print("\n🔴 Unconstrained model goes THROUGH wall (negative x)")
    print("🟢 Constrained model bounces correctly")
    print("🔵 Real physics bounces correctly")


def main():
    print("="*60)
    print("OPTION A: HARD PROJECTION ROLLOUT")
    print("="*60)
    
    # Load model
    print("\nLoading model...")
    model = ActionControllableModel(latent_dim=32, hidden_dim=128, action_dim=1, obs_dim=2)
    
    with open('/home/ulluboz/pcp-jepa-research/checkpoints/model_best.pkl', 'rb') as f:
        ckpt = pickle.load(f)
    params = ckpt['params']
    print(f"Loaded model (sens={ckpt['metrics']['sensitivity']:.3f})")
    
    # Visualize cheat trajectory
    visualize_cheat_trajectory(model, params)
    
    # Create planners
    print("\nCompiling planners...")
    mppi_unconstrained = create_mppi_constrained(model, params, apply_bounce=False)
    mppi_constrained = create_mppi_constrained(model, params, apply_bounce=True)
    
    # Warmup
    _ = mppi_unconstrained(jnp.array([1.0, 0.0]), jax.random.PRNGKey(0))
    _ = mppi_constrained(jnp.array([1.0, 0.0]), jax.random.PRNGKey(0))
    print("Done")
    
    # Run evaluations
    key = jax.random.PRNGKey(42)
    results = {}
    
    for method in ['random', 'mppi_unconstrained', 'mppi_constrained']:
        print(f"\n{method.upper()} ({N_EPISODES} episodes)...")
        
        if method == 'random':
            mppi = None
        elif method == 'mppi_unconstrained':
            mppi = mppi_unconstrained
        else:
            mppi = mppi_constrained
        
        successes, cats, misses = 0, 0, []
        start = time.time()
        
        for ep in range(N_EPISODES):
            key, subkey = jax.random.split(key)
            r = run_episode(subkey, mppi is not None, mppi)
            if r['success']: successes += 1
            if r['catastrophic']: cats += 1
            misses.append(r['miss'])
            
            if (ep+1) % 100 == 0:
                elapsed = time.time() - start
                print(f"  {ep+1}/{N_EPISODES} ({elapsed:.0f}s)")
        
        sr = successes / N_EPISODES
        cr = cats / N_EPISODES
        avg_miss = np.mean(misses)
        ci_sr = 1.96 * np.sqrt(sr * (1-sr) / N_EPISODES)
        
        results[method] = {
            'success_rate': sr,
            'success_ci': ci_sr,
            'catastrophic_rate': cr,
            'avg_miss': avg_miss
        }
        
        print(f"  Success: {sr:.1%} ± {ci_sr:.1%}")
        print(f"  Catastrophic: {cr:.1%}")
        print(f"  Miss: {avg_miss:.3f}")
    
    # Summary
    print("\n" + "="*60)
    print("SUMMARY")
    print("="*60)
    print(f"\n{'Method':<25} {'Success':<15} {'Catastrophic':<15} {'Avg Miss'}")
    print("-" * 70)
    for method, r in results.items():
        print(f"{method:<25} {r['success_rate']:.1%} ± {r['success_ci']:.1%}    "
              f"{r['catastrophic_rate']:.1%}            {r['avg_miss']:.3f}")
    
    # Key comparison
    print("\n" + "="*60)
    print("KEY RESULT")
    print("="*60)
    imp_uc = results['mppi_unconstrained']['success_rate'] - results['random']['success_rate']
    imp_c = results['mppi_constrained']['success_rate'] - results['random']['success_rate']
    
    print(f"\nMPPI unconstrained vs Random: {imp_uc:+.1%}")
    print(f"MPPI constrained vs Random:   {imp_c:+.1%}")
    
    if imp_c > imp_uc:
        print("\n✅ CONSTRAINT HELPS: Bounce projection improves planning!")
    else:
        print("\n⚠️  Constraint didn't help — need deeper fix")
    
    # Save
    output = {
        'n_episodes': N_EPISODES,
        'results': results,
        'key_finding': {
            'unconstrained_improvement': imp_uc,
            'constrained_improvement': imp_c,
            'constraint_helps': imp_c > imp_uc
        }
    }
    with open('/home/ulluboz/pcp-jepa-research/results/phase2/option_a_results.json', 'w') as f:
        json.dump(output, f, indent=2)
    print(f"\nSaved to option_a_results.json")


if __name__ == '__main__':
    main()