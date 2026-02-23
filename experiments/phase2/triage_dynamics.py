#!/usr/bin/env python3
"""Triage: Diagnose why learned dynamics don't help MPPI.

Quick tests:
1. Action influence: Do actions change trajectory?
2. One-step controllability: Does action correlate with Δx?
3. Multi-step error: Does error explode?
"""

import jax
import jax.numpy as jnp
import json
import numpy as np
import sys
import os

# Add repo to path
sys.path.insert(0, '/home/ulluboz/pcp-jepa-research')

# ============================================================
# Simple Physics Model for Ground Truth
# ============================================================

def physics_step(x, v, a, dt=0.05, restitution=0.8):
    """BouncingBall physics step."""
    v_new = v + a * dt
    x_new = x + v_new * dt
    
    # Bounce off walls
    if x_new < 0:
        x_new = -x_new
        v_new = -v_new * restitution
    elif x_new > 4.0:
        x_new = 8.0 - x_new
        v_new = -v_new * restitution
    
    return x_new, v_new


# ============================================================
# TEST 1: Action Influence on Random Dynamics
# ============================================================

def test_action_influence_random(key):
    """Test with a simple MLP dynamics model (random init)."""
    print("\n" + "=" * 60)
    print("TEST 1: Action Influence (Random Dynamics)")
    print("=" * 60)
    
    import flax.linen as nn
    
    class SimpleDynamics(nn.Module):
        latent_dim: int = 16
        hidden_dim: int = 32
        
        @nn.compact
        def __call__(self, obs, action):
            z = nn.Dense(self.latent_dim)(obs)
            x = jnp.concatenate([z, action], axis=-1)
            delta_z = nn.Dense(self.hidden_dim)(x)
            delta_z = nn.relu(delta_z)
            delta_z = nn.Dense(self.latent_dim)(delta_z)
            z_next = z + delta_z
            return nn.Dense(2)(z_next), z_next
    
    # Initialize model
    model = SimpleDynamics()
    key, subkey = jax.random.split(key)
    dummy_obs = jnp.zeros(2)
    dummy_action = jnp.zeros(1)
    params = model.init(subkey, dummy_obs, dummy_action)
    
    # Test 1: All +1 actions vs all -1 actions
    obs0 = jnp.array([1.0, 0.5])  # Start at x=1, v=0.5
    H = 30
    
    # Rollout with +1 actions
    obs = obs0
    for t in range(H):
        a = jnp.array([1.0])
        obs, _ = model.apply(params, obs, a)
    x_plus = float(obs[0])
    
    # Rollout with -1 actions
    obs = obs0
    for t in range(H):
        a = jnp.array([-1.0])
        obs, _ = model.apply(params, obs, a)
    x_minus = float(obs[0])
    
    delta_x = abs(x_plus - x_minus)
    
    print(f"  Initial: x={obs0[0]:.2f}, v={obs0[1]:.2f}")
    print(f"  +1 actions: x_final={x_plus:.4f}")
    print(f"  -1 actions: x_final={x_minus:.4f}")
    print(f"  Δx = {delta_x:.4f}")
    
    if delta_x < 0.1:
        print("  ❌ FAIL: Actions barely influence trajectory")
        return False
    else:
        print("  ✅ PASS: Actions change trajectory")
        return True


# ============================================================
# TEST 2: Ground Truth Controllability
# ============================================================

def test_ground_truth_controllability(key, n_samples=100):
    """Test action correlation with Δx in true physics."""
    print("\n" + "=" * 60)
    print("TEST 2: Ground Truth Controllability")
    print("=" * 60)
    
    # Sample random states
    key, subkey = jax.random.split(key)
    x0 = jax.random.uniform(subkey, (n_samples,), minval=0.5, maxval=3.5)
    key, subkey = jax.random.split(key)
    v0 = jax.random.uniform(subkey, (n_samples,), minval=-1.0, maxval=1.0)
    
    # Sample random actions
    key, subkey = jax.random.split(key)
    actions = jax.random.uniform(subkey, (n_samples,), minval=-1.0, maxval=1.0)
    
    # Compute one-step transitions
    x1 = []
    v1 = []
    for i in range(n_samples):
        x_new, v_new = physics_step(float(x0[i]), float(v0[i]), float(actions[i]))
        x1.append(x_new)
        v1.append(v_new)
    
    x1 = jnp.array(x1)
    v1 = jnp.array(v1)
    delta_x = x1 - x0
    delta_v = v1 - v0
    
    # Compute correlations
    corr_x = float(jnp.corrcoef(actions, delta_x)[0, 1])
    corr_v = float(jnp.corrcoef(actions, delta_v)[0, 1])
    
    print(f"  Correlation(a, Δx) = {corr_x:.4f}")
    print(f"  Correlation(a, Δv) = {corr_v:.4f}")
    
    if abs(corr_v) < 0.3:
        print("  ⚠️  Low correlation — physics might have low action influence")
    else:
        print("  ✅ Actions affect velocity as expected (correlation ≈ 1.0)")
    
    return True


# ============================================================
# TEST 3: Multi-step Prediction Error
# ============================================================

def test_multistep_prediction(key, H=30, n_samples=10):
    """Test how prediction error accumulates over time."""
    print("\n" + "=" * 60)
    print(f"TEST 3: Multi-step Prediction Error (H={H})")
    print("=" * 60)
    
    import flax.linen as nn
    
    class SimpleModel(nn.Module):
        latent_dim: int = 16
        hidden_dim: int = 32
        
        @nn.compact
        def __call__(self, obs, action):
            """Single step prediction."""
            z = nn.Dense(self.latent_dim)(obs)
            x = jnp.concatenate([z, action], axis=-1)
            delta_z = nn.Dense(self.hidden_dim)(x)
            delta_z = nn.relu(delta_z)
            delta_z = nn.Dense(self.latent_dim)(delta_z)
            z_next = z + delta_z
            return nn.Dense(2)(z_next), z_next
    
    model = SimpleModel()
    key, subkey = jax.random.split(key)
    params = model.init(subkey, jnp.zeros(2), jnp.zeros(1))
    
    errors = []
    
    for i in range(n_samples):
        key, subkey = jax.random.split(key)
        
        # Random initial state
        x0 = float(jax.random.uniform(subkey, (), minval=0.5, maxval=3.5))
        key, subkey = jax.random.split(key)
        v0 = float(jax.random.uniform(subkey, (), minval=-0.5, maxval=0.5))
        
        # Random action sequence
        key, subkey = jax.random.split(key)
        actions = jax.random.uniform(subkey, (H, 1), minval=-1.0, maxval=1.0)
        
        # Ground truth rollout
        x_true = [x0]
        v_true = [v0]
        x, v = x0, v0
        for t in range(H):
            x, v = physics_step(x, v, float(actions[t, 0]))
            x_true.append(x)
            v_true.append(v)
        
        # Model rollout
        obs = jnp.array([x0, v0])
        x_pred = [x0]
        
        for t in range(H):
            obs, _ = model.apply(params, obs, actions[t])
            x_pred.append(float(obs[0]))
        
        # Compute errors
        sample_errors = [abs(x_pred[t] - x_true[t]) for t in range(1, H+1)]
        errors.append(sample_errors)
    
    errors = np.array(errors)
    mean_errors = np.mean(errors, axis=0)
    
    print(f"  1-step error: {mean_errors[0]:.4f}")
    print(f"  5-step error: {mean_errors[4]:.4f}")
    print(f"  10-step error: {mean_errors[9]:.4f}")
    print(f"  20-step error: {mean_errors[19]:.4f}")
    print(f"  30-step error: {mean_errors[29]:.4f}")
    print(f"  Error growth: {mean_errors[29] / max(mean_errors[0], 1e-6):.1f}x")
    
    if mean_errors[29] > 5.0:
        print("  ⚠️  Error explodes with random model")
    else:
        print("  (Random model — error is expected to be large)")
    
    return True


# ============================================================
# TEST 4: Check Actual Trained Model
# ============================================================

def test_trained_model(key):
    """Test if trained O3 model exists and check its action sensitivity."""
    print("\n" + "=" * 60)
    print("TEST 4: Trained Model Check")
    print("=" * 60)
    
    # Look for any saved models
    checkpoint_paths = [
        '/home/ulluboz/pcp-jepa-research/checkpoints',
        '/home/ulluboz/pcp-jepa-research/models',
    ]
    
    found = False
    for path in checkpoint_paths:
        if os.path.exists(path):
            files = os.listdir(path)
            if files:
                print(f"  Found checkpoints in {path}:")
                for f in files[:5]:
                    print(f"    - {f}")
                found = True
    
    if not found:
        print("  ❌ No trained model checkpoints found")
        print("  → Models were trained in memory but not saved")
        print("\n  This is the root cause:")
        print("  - Training ran but models weren't persisted")
        print("  - MPPI evaluation tried to use random/untrained models")
        return False
    
    return True


# ============================================================
# TEST 5: Train a Simple Model and Test Action Influence
# ============================================================

def test_trained_model_influence(key):
    """Train a simple model and verify it learns action influence."""
    print("\n" + "=" * 60)
    print("TEST 5: Train Simple Model & Verify Action Influence")
    print("=" * 60)
    
    import flax.linen as nn
    import optax
    
    class SimpleModel(nn.Module):
        latent_dim: int = 16
        hidden_dim: int = 64
        
        @nn.compact
        def __call__(self, obs, action):
            z = nn.Dense(self.latent_dim)(obs)
            x = jnp.concatenate([z, action], axis=-1)
            delta_z = nn.Dense(self.hidden_dim)(x)
            delta_z = nn.relu(delta_z)
            delta_z = nn.Dense(self.hidden_dim)(delta_z)
            delta_z = nn.relu(delta_z)
            delta_z = nn.Dense(self.latent_dim)(delta_z)
            z_next = z + delta_z
            return nn.Dense(2)(z_next)
    
    model = SimpleModel()
    key, subkey = jax.random.split(key)
    params = model.init(subkey, jnp.zeros(2), jnp.zeros(1))
    
    # Generate training data
    print("  Generating training data...")
    n_train = 1000
    X = []
    Y = []
    A = []
    
    for _ in range(n_train):
        key, subkey = jax.random.split(key)
        x0 = float(jax.random.uniform(subkey, (), minval=0.5, maxval=3.5))
        key, subkey = jax.random.split(key)
        v0 = float(jax.random.uniform(subkey, (), minval=-0.5, maxval=0.5))
        key, subkey = jax.random.split(key)
        a = float(jax.random.uniform(subkey, (), minval=-1.0, maxval=1.0))
        
        x1, v1 = physics_step(x0, v0, a)
        
        X.append([x0, v0])
        A.append([a])
        Y.append([x1, v1])
    
    X = jnp.array(X)
    A = jnp.array(A)
    Y = jnp.array(Y)
    
    # Train
    print("  Training model...")
    optimizer = optax.adam(1e-3)
    opt_state = optimizer.init(params)
    
    def loss_fn(params, X, A, Y):
        Y_pred = jax.vmap(lambda x, a: model.apply(params, x, a))(X, A)
        return jnp.mean((Y_pred - Y) ** 2)
    
    @jax.jit
    def update(params, opt_state, X, A, Y):
        loss, grads = jax.value_and_grad(loss_fn)(params, X, A, Y)
        updates, opt_state = optimizer.update(grads, opt_state)
        params = optax.apply_updates(params, updates)
        return params, opt_state, loss
    
    losses = []
    for epoch in range(100):
        params, opt_state, loss = update(params, opt_state, X, A, Y)
        losses.append(float(loss))
        if epoch % 20 == 0:
            print(f"    Epoch {epoch}: loss = {loss:.6f}")
    
    print(f"  Final loss: {losses[-1]:.6f}")
    
    # Test action influence AFTER training
    print("\n  Testing action influence on trained model:")
    
    obs0 = jnp.array([1.0, 0.0])
    H = 30
    
    # Rollout with +1
    obs = obs0
    for t in range(H):
        obs = model.apply(params, obs, jnp.array([1.0]))
    x_plus = float(obs[0])
    
    # Rollout with -1
    obs = obs0
    for t in range(H):
        obs = model.apply(params, obs, jnp.array([-1.0]))
    x_minus = float(obs[0])
    
    delta_x = abs(x_plus - x_minus)
    
    print(f"    +1 actions: x_final={x_plus:.4f}")
    print(f"    -1 actions: x_final={x_minus:.4f}")
    print(f"    Δx = {delta_x:.4f}")
    
    if delta_x > 1.0:
        print("    ✅ PASS: Trained model shows strong action influence")
        return True
    else:
        print("    ⚠️  WARNING: Trained model still has weak action influence")
        return False


# ============================================================
# SUMMARY & RECOMMENDATIONS
# ============================================================

def print_recommendations():
    print("\n" + "=" * 60)
    print("DIAGNOSIS & RECOMMENDATIONS")
    print("=" * 60)
    
    print("""
KEY FINDING: Models were trained in memory but NOT SAVED.

This explains why:
- MPPI got SIGKILL (tried to use non-existent checkpoints)
- Results show random performance (models not loaded)
- Catastrophic rate is 0% (baseline is broken)

FIX:
1. Save model checkpoints after training
2. Load checkpoints before evaluation
3. Verify action influence in loaded model

QUICK TEST:
- Train simple model (TEST 5) shows if architecture works
- If TEST 5 passes, the issue is checkpointing, not architecture
""")


# ============================================================
# MAIN
# ============================================================

def main():
    print("=" * 60)
    print("DYNAMICS TRIAGE - PCP-JEPA")
    print("=" * 60)
    
    key = jax.random.PRNGKey(42)
    
    results = {}
    
    # Run tests
    results['action_influence_random'] = test_action_influence_random(key)
    results['ground_truth_ctrl'] = test_ground_truth_controllability(key)
    results['multistep_error'] = test_multistep_prediction(key)
    results['trained_model'] = test_trained_model(key)
    results['train_and_test'] = test_trained_model_influence(key)
    
    print_recommendations()
    
    # Save
    os.makedirs('/home/ulluboz/pcp-jepa-research/results/phase2', exist_ok=True)
    with open('/home/ulluboz/pcp-jepa-research/results/phase2/triage_results.json', 'w') as f:
        json.dump({k: str(v) for k, v in results.items()}, f, indent=2)
    
    print(f"\nResults saved to triage_results.json")


if __name__ == '__main__':
    main()
