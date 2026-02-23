#!/usr/bin/env python3
"""Verify trained action-controllable model."""

import jax
import jax.numpy as jnp
import pickle
import numpy as np
import sys
sys.path.insert(0, '/home/ulluboz/pcp-jepa-research')

from experiments.phase2.train_action_controllable import (
    ActionControllableModel, Config, physics_step
)

def main():
    cfg = Config()
    key = jax.random.PRNGKey(42)
    
    # Load checkpoint
    checkpoint_path = '/home/ulluboz/pcp-jepa-research/checkpoints/model_best.pkl'
    print(f"Loading checkpoint from {checkpoint_path}")
    
    with open(checkpoint_path, 'rb') as f:
        checkpoint = pickle.load(f)
    
    params = checkpoint['params']
    print(f"Loaded model from epoch {checkpoint['epoch']}")
    print(f"Metrics: {checkpoint['metrics']}")
    
    # Create model
    model = ActionControllableModel(
        latent_dim=cfg.latent_dim,
        hidden_dim=cfg.hidden_dim,
        action_dim=cfg.action_dim,
        obs_dim=cfg.obs_dim
    )
    
    # ============================================================
    # VERIFICATION 1: Action Influence
    # ============================================================
    print("\n" + "=" * 60)
    print("VERIFICATION: Action Influence Test")
    print("=" * 60)
    
    test_states = [
        (1.0, 0.0),
        (2.0, 0.5),
        (3.0, -0.3),
    ]
    
    all_delta_x = []
    for x0, v0 in test_states:
        obs0 = jnp.array([x0, v0])
        
        # +1 actions
        obs = obs0
        for t in range(30):
            obs = model.apply(params, obs, jnp.array([1.0]))
        x_plus = float(obs[0])
        
        # -1 actions
        obs = obs0
        for t in range(30):
            obs = model.apply(params, obs, jnp.array([-1.0]))
        x_minus = float(obs[0])
        
        delta_x = abs(x_plus - x_minus)
        all_delta_x.append(delta_x)
        
        print(f"  Start ({x0:.1f}, {v0:.1f}): +1→{x_plus:.2f}, -1→{x_minus:.2f}, Δx={delta_x:.2f}")
    
    avg_delta = np.mean(all_delta_x)
    print(f"\n  Average Δx = {avg_delta:.2f}")
    print(f"  {'✅ PASS' if avg_delta >= 2.0 else '❌ FAIL'}: Target ≥2.0")
    
    # ============================================================
    # VERIFICATION 2: Multi-step Stability
    # ============================================================
    print("\n" + "=" * 60)
    print("VERIFICATION: Multi-step Stability")
    print("=" * 60)
    
    errors = []
    for i in range(10):
        key, subkey = jax.random.split(key)
        x0 = float(jax.random.uniform(subkey, (), minval=0.5, maxval=3.5))
        key, subkey = jax.random.split(key)
        v0 = float(jax.random.uniform(subkey, (), minval=-0.5, maxval=0.5))
        key, subkey = jax.random.split(key)
        actions = jax.random.uniform(subkey, (30, 1), minval=-1.0, maxval=1.0)
        
        # Ground truth
        x_true = [x0]
        x, v = x0, v0
        for t in range(30):
            x, v = physics_step(x, v, float(actions[t, 0]))
            x_true.append(x)
        
        # Model
        obs = jnp.array([x0, v0])
        x_pred = [x0]
        for t in range(30):
            obs = model.apply(params, obs, actions[t])
            x_pred.append(float(obs[0]))
        
        sample_errors = [abs(x_pred[t] - x_true[t]) for t in range(1, 31)]
        errors.append(sample_errors)
    
    errors = np.array(errors)
    mean_errors = np.mean(errors, axis=0)
    
    print(f"  1-step error: {mean_errors[0]:.4f}")
    print(f"  10-step error: {mean_errors[9]:.4f}")
    print(f"  20-step error: {mean_errors[19]:.4f}")
    print(f"  30-step error: {mean_errors[29]:.4f}")
    
    # ============================================================
    # VERIFICATION 3: Jacobian Sensitivity
    # ============================================================
    print("\n" + "=" * 60)
    print("VERIFICATION: Action Sensitivity (Jacobian)")
    print("=" * 60)
    
    sensitivities = []
    for i in range(100):
        key, subkey = jax.random.split(key)
        x0 = float(jax.random.uniform(subkey, (), minval=0.5, maxval=3.5))
        key, subkey = jax.random.split(key)
        v0 = float(jax.random.uniform(subkey, (), minval=-0.5, maxval=0.5))
        key, subkey = jax.random.split(key)
        a = float(jax.random.uniform(subkey, (), minval=-1.0, maxval=1.0))
        
        obs = jnp.array([x0, v0])
        
        def predict_x(a_scalar):
            return model.apply(params, obs, jnp.array([a_scalar]))[0]
        
        dx_da = float(jax.grad(predict_x)(a))
        sensitivities.append(abs(dx_da))
    
    avg_sens = np.mean(sensitivities)
    min_sens = np.min(sensitivities)
    
    print(f"  Average |∂x/∂a| = {avg_sens:.4f}")
    print(f"  Min |∂x/∂a| = {min_sens:.4f}")
    
    # ============================================================
    # SUMMARY
    # ============================================================
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print(f"  Action influence: {avg_delta:.2f} (target ≥2.0) {'✅' if avg_delta >= 2.0 else '❌'}")
    print(f"  30-step error: {mean_errors[29]:.2f}")
    print(f"  Avg sensitivity: {avg_sens:.3f}")
    
    if avg_delta >= 2.0:
        print("\n✅ Model is ready for MPPI planning!")
    else:
        print("\n⚠️  Model needs more action sensitivity")


if __name__ == '__main__':
    main()
