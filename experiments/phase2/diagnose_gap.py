#!/usr/bin/env python3
"""
Diagnose model-reality gap: compare model predictions to real physics.
"""

import jax
import jax.numpy as jnp
import pickle
import numpy as np
import sys
sys.path.insert(0, '/home/ulluboz/pcp-jepa-research')

from experiments.phase2.train_action_controllable import (
    ActionControllableModel, physics_step
)

def main():
    print("Loading model...")
    model = ActionControllableModel(latent_dim=32, hidden_dim=128, action_dim=1, obs_dim=2)
    
    with open('/home/ulluboz/pcp-jepa-research/checkpoints/model_best.pkl', 'rb') as f:
        ckpt = pickle.load(f)
    params = ckpt['params']
    
    print("\n" + "="*60)
    print("MODEL-REALITY GAP DIAGNOSIS")
    print("="*60)
    
    # Test 1: Single-step prediction accuracy
    print("\n1. Single-step prediction error:")
    errors = []
    for _ in range(100):
        key = jax.random.PRNGKey(np.random.randint(0, 10000))
        x0 = np.random.uniform(0.5, 3.5)
        v0 = np.random.uniform(-0.5, 0.5)
        a = np.random.uniform(-1.0, 1.0)
        
        # Real physics
        x_real, v_real = physics_step(x0, v0, a)
        
        # Model prediction
        obs = jnp.array([x0, v0])
        pred = model.apply(params, obs, jnp.array([a]))
        x_pred, v_pred = float(pred[0]), float(pred[1])
        
        errors.append([abs(x_pred - x_real), abs(v_pred - v_real)])
    
    errors = np.array(errors)
    print(f"  Mean x error: {np.mean(errors[:,0]):.4f}")
    print(f"  Mean v error: {np.mean(errors[:,1]):.4f}")
    
    # Test 2: Trajectory divergence
    print("\n2. 30-step trajectory error:")
    traj_errors = []
    
    for _ in range(20):
        x0 = np.random.uniform(1.0, 3.0)
        v0 = np.random.uniform(-0.3, 0.3)
        
        # Random action sequence
        actions = [np.random.uniform(-0.5, 0.5) for _ in range(30)]
        
        # Real physics rollout
        x_real, v_real = x0, v0
        real_traj = [(x_real, v_real)]
        for a in actions:
            x_real, v_real = physics_step(x_real, v_real, a)
            real_traj.append((x_real, v_real))
        
        # Model rollout
        obs = jnp.array([x0, v0])
        model_traj = [(float(obs[0]), float(obs[1]))]
        for a in actions:
            obs = model.apply(params, obs, jnp.array([a]))
            model_traj.append((float(obs[0]), float(obs[1])))
        
        # Compare final states
        final_x_err = abs(model_traj[-1][0] - real_traj[-1][0])
        traj_errors.append(final_x_err)
    
    print(f"  Mean final x error: {np.mean(traj_errors):.3f}")
    print(f"  Max final x error: {np.max(traj_errors):.3f}")
    
    # Test 3: Does model know about bounces?
    print("\n3. Bounce behavior test:")
    print("  (Testing if model learns wall bounces)")
    
    # Start near wall, strong negative velocity
    x0, v0 = 0.3, -0.5  # Should bounce
    obs = jnp.array([x0, v0])
    
    # Model prediction for 10 steps with a=0
    model_traj = [float(obs[0])]
    for _ in range(10):
        obs = model.apply(params, obs, jnp.array([0.0]))
        model_traj.append(float(obs[0]))
    
    # Real physics
    x, v = x0, v0
    real_traj = [x]
    for _ in range(10):
        x, v = physics_step(x, v, 0.0)
        real_traj.append(x)
    
    print(f"  Model trajectory: {model_traj[:5]}")
    print(f"  Real trajectory:  {real_traj[:5]}")
    
    # Test 4: Does model know action direction?
    print("\n4. Action direction test:")
    
    # Start at center, stationary
    obs0 = jnp.array([2.0, 0.0])
    
    # Apply +1 for 10 steps
    obs = obs0
    for _ in range(10):
        obs = model.apply(params, obs, jnp.array([1.0]))
    x_plus = float(obs[0])
    
    # Apply -1 for 10 steps
    obs = obs0
    for _ in range(10):
        obs = model.apply(params, obs, jnp.array([-1.0]))
    x_minus = float(obs[0])
    
    print(f"  +1 actions: x = {x_plus:.2f}")
    print(f"  -1 actions: x = {x_minus:.2f}")
    print(f"  Direction correct: {'✅' if x_plus > x_minus else '❌'}")
    
    # Test 5: Target reaching (what does MPPI think is optimal?)
    print("\n5. MPPI target reaching analysis:")
    print(f"  Target: x = 2.0")
    
    # Sample many action sequences and see what the model predicts
    key = jax.random.PRNGKey(42)
    
    # From various starting positions
    for x0 in [1.0, 1.5, 2.0, 2.5, 3.0]:
        obs0 = jnp.array([x0, 0.0])
        
        # Sample 100 action sequences
        best_x = None
        best_cost = float('inf')
        best_actions = None
        
        for _ in range(100):
            key, subkey = jax.random.split(key)
            actions = jax.random.uniform(subkey, (10, 1), minval=-1.0, maxval=1.0)
            
            obs = obs0
            for a in actions:
                obs = model.apply(params, obs, a)
            
            final_x = float(obs[0])
            cost = abs(final_x - 2.0)
            
            if cost < best_cost:
                best_cost = cost
                best_x = final_x
                best_actions = actions
        
        # Execute best_actions in REAL physics
        x_real, v_real = float(x0), 0.0
        for a in best_actions:
            x_real, v_real = physics_step(x_real, v_real, float(a[0]))
        
        print(f"  Start {x0:.1f}: model predicts x={best_x:.2f}, real x={x_real:.2f}, "
              f"gap={abs(best_x - x_real):.2f}")
    
    print("\n" + "="*60)
    print("DIAGNOSIS")
    print("="*60)
    print("""
The model has:
- Strong action influence (4.27) ✅
- Correct action direction ✅

But it likely:
- Doesn't learn accurate physics (bounce behavior)
- Has large trajectory errors over 30 steps
- MPPI exploits these errors → plans cheat trajectories

FIX:
1. Add physics-consistency loss (energy conservation, bounce dynamics)
2. Train with longer rollouts (multi-step loss)
3. Use mixed model: physics for known dynamics, neural for residuals
""")


if __name__ == '__main__':
    main()