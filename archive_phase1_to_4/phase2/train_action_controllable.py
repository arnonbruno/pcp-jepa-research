#!/usr/bin/env python3
"""
Train Action-Controllable World Model

Key insight: Low prediction loss does NOT imply controllability.
This pipeline enforces action sensitivity through:
1. Jacobian-based action sensitivity loss
2. Contrastive action effect loss
3. FiLM conditioning (actions can't be washed out)
4. Proper data distribution (persistently exciting actions)
"""

import jax
import jax.numpy as jnp
import flax.linen as nn
import optax
import numpy as np
import json
import os
import pickle
from dataclasses import dataclass
from typing import Dict, Tuple, Optional
from functools import partial

# ============================================================
# CONFIG
# ============================================================

@dataclass
class Config:
    # Model
    latent_dim: int = 32
    hidden_dim: int = 128
    action_dim: int = 1
    obs_dim: int = 2
    
    # Training
    n_train: int = 5000
    n_val: int = 500
    batch_size: int = 256
    epochs: int = 200
    lr: float = 1e-3
    
    # Action sensitivity
    epsilon_sens: float = 0.1  # Minimum |dx/da|
    lambda_sens: float = 0.5  # Weight for sensitivity loss
    lambda_contrast: float = 0.3  # Weight for contrastive loss
    delta_contrast: float = 0.2  # Minimum separation for contrastive
    
    # Physics
    dt: float = 0.05
    restitution: float = 0.8
    x_bounds: Tuple[float, float] = (0.0, 4.0)
    
    # Checkpoints
    checkpoint_dir: str = '/home/ulluboz/pcp-jepa-research/checkpoints'
    save_every: int = 20
    
    # Evaluation
    horizon: int = 30
    n_eval_episodes: int = 20


# ============================================================
# FIEM-CONDITIONED DYNAMICS MODEL
# ============================================================

class FiLMLayer(nn.Module):
    """Feature-wise Linear Modulation layer."""
    features: int
    
    @nn.compact
    def __call__(self, x, gamma, beta):
        # gamma: scale, beta: shift
        return gamma * x + beta


class ActionControllableModel(nn.Module):
    """
    World model with FiLM conditioning for strong action influence.
    
    Architecture:
    - Encoder: obs -> latent
    - Action encoder: a -> (gamma, beta) for FiLM layers
    - Dynamics: latent -> latent (with FiLM conditioning)
    - Decoder: latent -> obs
    """
    latent_dim: int
    hidden_dim: int
    action_dim: int
    obs_dim: int
    
    @nn.compact
    def encode(self, obs: jnp.ndarray) -> jnp.ndarray:
        """Encode observation to latent."""
        z = nn.Dense(self.hidden_dim)(obs)
        z = nn.relu(z)
        z = nn.Dense(self.latent_dim)(z)
        return z
    
    @nn.compact
    def decode(self, z: jnp.ndarray) -> jnp.ndarray:
        """Decode latent to observation."""
        x = nn.Dense(self.hidden_dim)(z)
        x = nn.relu(x)
        x = nn.Dense(self.obs_dim)(x)
        return x
    
    @nn.compact
    def action_to_film(self, a: jnp.ndarray) -> Tuple[jnp.ndarray, jnp.ndarray]:
        """Convert action to FiLM parameters (gamma, beta)."""
        # Action embedding
        a_emb = nn.Dense(self.hidden_dim)(a)
        a_emb = nn.relu(a_emb)
        
        # Generate gamma and beta for 2 FiLM layers
        gamma1 = nn.Dense(self.hidden_dim, name='gamma1')(a_emb)
        beta1 = nn.Dense(self.hidden_dim, name='beta1')(a_emb)
        
        gamma2 = nn.Dense(self.latent_dim, name='gamma2')(a_emb)
        beta2 = nn.Dense(self.latent_dim, name='beta2')(a_emb)
        
        return (gamma1, beta1), (gamma2, beta2)
    
    @nn.compact
    def dynamics_step(self, z: jnp.ndarray, a: jnp.ndarray) -> jnp.ndarray:
        """
        One-step latent dynamics with FiLM conditioning.
        
        This makes it HARD for the network to ignore actions.
        """
        # Get FiLM parameters
        (gamma1, beta1), (gamma2, beta2) = self.action_to_film(a)
        
        # First layer with FiLM
        h = nn.Dense(self.hidden_dim)(z)
        h = nn.relu(h)
        h = gamma1 * h + beta1  # FiLM conditioning
        
        # Second layer with FiLM
        h = nn.Dense(self.hidden_dim)(h)
        h = nn.relu(h)
        
        # Predict delta_z
        delta_z = nn.Dense(self.latent_dim)(h)
        delta_z = gamma2 * delta_z + beta2  # FiLM on output
        
        return z + delta_z
    
    @nn.compact
    def __call__(self, obs: jnp.ndarray, a: jnp.ndarray) -> jnp.ndarray:
        """Full forward pass: obs -> next_obs."""
        z = self.encode(obs)
        z_next = self.dynamics_step(z, a)
        return self.decode(z_next)
    
    @nn.compact
    def predict_next(self, obs: jnp.ndarray, a: jnp.ndarray) -> jnp.ndarray:
        """Alias for __call__."""
        return self(obs, a)
    
    @nn.compact
    def predict_with_jacobian(self, obs: jnp.ndarray, a: jnp.ndarray) -> Tuple[jnp.ndarray, jnp.ndarray]:
        """Predict next state and compute ∂x_next/∂a."""
        z = self.encode(obs)
        z_next = self.dynamics_step(z, a)
        x_next = self.decode(z_next)
        
        # Jacobian wrt action
        def predict_x_next(a_):
            z_next_ = self.dynamics_step(z, a_)
            return self.decode(z_next_)[0]  # x coordinate only
        
        dx_da = jax.grad(predict_x_next)(a)
        
        return x_next, dx_da


# ============================================================
# PHYSICS SIMULATOR (Ground Truth)
# ============================================================

def physics_step(x: float, v: float, a: float, 
                 dt: float = 0.05, 
                 restitution: float = 0.8,
                 x_min: float = 0.0,
                 x_max: float = 4.0) -> Tuple[float, float]:
    """BouncingBall physics with walls."""
    v_new = v + a * dt
    x_new = x + v_new * dt
    
    # Bounce off walls
    if x_new < x_min:
        x_new = 2 * x_min - x_new  # = -x_new
        v_new = -v_new * restitution
    elif x_new > x_max:
        x_new = 2 * x_max - x_new  # = 8 - x_new
        v_new = -v_new * restitution
    
    return x_new, v_new


# ============================================================
# DATA GENERATION (Persistently Exciting Actions)
# ============================================================

def generate_dataset(cfg: Config, key: jax.random.PRNGKey) -> Dict:
    """
    Generate training data with diverse action sequences.
    
    Key: "Persistently exciting" actions ensure model can't
    learn to ignore them as noise.
    """
    data = {'train': [], 'val': []}
    
    for split, n_samples in [('train', cfg.n_train), ('val', cfg.n_val)]:
        observations = []
        actions = []
        next_observations = []
        
        for i in range(n_samples):
            key, subkey = jax.random.split(key)
            
            # Random initial state
            x0 = float(jax.random.uniform(subkey, (), minval=0.5, maxval=3.5))
            key, subkey = jax.random.split(key)
            v0 = float(jax.random.uniform(subkey, (), minval=-0.5, maxval=0.5))
            
            # Diverse action distribution
            # Mix of: constant +1, constant -1, ramps, random
            key, subkey = jax.random.split(key)
            action_type = jax.random.randint(subkey, (), 0, 4)
            
            if action_type == 0:
                # Constant +1
                a = 1.0
            elif action_type == 1:
                # Constant -1
                a = -1.0
            elif action_type == 2:
                # Ramp (based on velocity)
                a = -np.sign(v0) * 0.5
            else:
                # Random
                key, subkey = jax.random.split(key)
                a = float(jax.random.uniform(subkey, (), minval=-1.0, maxval=1.0))
            
            # Apply physics
            x1, v1 = physics_step(x0, v0, a, cfg.dt, cfg.restitution,
                                  cfg.x_bounds[0], cfg.x_bounds[1])
            
            observations.append([x0, v0])
            actions.append([a])
            next_observations.append([x1, v1])
        
        data[split] = {
            'obs': jnp.array(observations),
            'actions': jnp.array(actions),
            'next_obs': jnp.array(next_observations)
        }
    
    return data


# ============================================================
# LOSSES
# ============================================================

def prediction_loss(model, params, obs, actions, next_obs):
    """Standard MSE prediction loss."""
    pred = jax.vmap(lambda o, a: model.apply(params, o, a))(obs, actions)
    return jnp.mean((pred - next_obs) ** 2)


def action_sensitivity_loss(model, params, obs, actions, epsilon=0.1):
    """
    Jacobian-based action sensitivity loss.
    
    L_sens = E[ReLU(ε - |∂x_next/∂a|)]
    
    Forces |dx/da| >= ε, ensuring actions affect predictions.
    """
    def compute_sensitivity(o, a):
        # We need to compute gradient through model
        def predict_x(a_):
            return model.apply(params, o, a_)[0]  # x coordinate
        
        dx_da = jax.grad(predict_x)(a)
        return jnp.abs(dx_da)
    
    sensitivities = jax.vmap(compute_sensitivity)(obs, actions)
    
    # Hinge loss: penalize if sensitivity < epsilon
    loss = jnp.mean(jax.nn.relu(epsilon - sensitivities))
    
    return loss, jnp.mean(sensitivities)


def contrastive_action_loss(model, params, obs, actions, key, delta=0.2):
    """
    Contrastive loss: same state, different actions should give different predictions.
    
    L_ce = E[ReLU(δ - |x_next(a) - x_next(a')|)]
    
    a and a' are two different actions from the same state.
    """
    batch_size = obs.shape[0]
    
    # Sample different actions
    key, subkey = jax.random.split(key)
    actions_alt = jax.random.uniform(subkey, (batch_size, 1), minval=-1.0, maxval=1.0)
    
    # Predict with both actions
    pred_orig = jax.vmap(lambda o, a: model.apply(params, o, a))(obs, actions)
    pred_alt = jax.vmap(lambda o, a: model.apply(params, o, a))(obs, actions_alt)
    
    # Distance in x coordinate
    separation = jnp.abs(pred_orig[:, 0] - pred_alt[:, 0])
    
    # Hinge loss: penalize if separation < delta
    loss = jnp.mean(jax.nn.relu(delta - separation))
    
    return loss, jnp.mean(separation)


def total_loss(model, params, obs, actions, next_obs, key, cfg):
    """Combined loss with weights."""
    # Prediction loss
    l_pred = prediction_loss(model, params, obs, actions, next_obs)
    
    # Action sensitivity loss
    l_sens, sens_metric = action_sensitivity_loss(
        model, params, obs, actions, cfg.epsilon_sens
    )
    
    # Contrastive loss
    l_contrast, sep_metric = contrastive_action_loss(
        model, params, obs, actions, key, cfg.delta_contrast
    )
    
    # Total
    total = (l_pred + 
             cfg.lambda_sens * l_sens + 
             cfg.lambda_contrast * l_contrast)
    
    # Return JAX arrays (not Python floats) for JIT compatibility
    metrics = {
        'loss_pred': l_pred,
        'loss_sens': l_sens,
        'loss_contrast': l_contrast,
        'loss_total': total,
        'sensitivity': sens_metric,
        'separation': sep_metric
    }
    
    return total, metrics


# ============================================================
# TRAINING LOOP
# ============================================================

def train(cfg: Config, key: jax.random.PRNGKey):
    """Train the action-controllable world model."""
    
    print("=" * 60)
    print("TRAINING ACTION-CONTROLLABLE WORLD MODEL")
    print("=" * 60)
    
    # Create model
    model = ActionControllableModel(
        latent_dim=cfg.latent_dim,
        hidden_dim=cfg.hidden_dim,
        action_dim=cfg.action_dim,
        obs_dim=cfg.obs_dim
    )
    
    # Initialize
    key, subkey = jax.random.split(key)
    params = model.init(subkey, jnp.zeros(cfg.obs_dim), jnp.zeros(cfg.action_dim))
    
    # Generate data
    print("\nGenerating data...")
    data = generate_dataset(cfg, key)
    print(f"  Train: {cfg.n_train} samples")
    print(f"  Val: {cfg.n_val} samples")
    
    # Optimizer
    optimizer = optax.adam(cfg.lr)
    opt_state = optimizer.init(params)
    
    # Training function
    @jax.jit
    def update(params, opt_state, obs, actions, next_obs, key):
        (loss, metrics), grads = jax.value_and_grad(
            lambda p: total_loss(model, p, obs, actions, next_obs, key, cfg),
            has_aux=True
        )(params)
        updates, opt_state = optimizer.update(grads, opt_state)
        params = optax.apply_updates(params, updates)
        return params, opt_state, loss, metrics
    
    # Training loop
    print("\nTraining...")
    history = {'train': [], 'val': []}
    best_val_loss = float('inf')
    best_params = None
    
    train_obs = data['train']['obs']
    train_actions = data['train']['actions']
    train_next_obs = data['train']['next_obs']
    
    val_obs = data['val']['obs']
    val_actions = data['val']['actions']
    val_next_obs = data['val']['next_obs']
    
    for epoch in range(cfg.epochs):
        # Shuffle
        key, subkey = jax.random.split(key)
        perm = jax.random.permutation(subkey, cfg.n_train)
        
        train_obs = train_obs[perm]
        train_actions = train_actions[perm]
        train_next_obs = train_next_obs[perm]
        
        # Mini-batches
        epoch_metrics = []
        for i in range(0, cfg.n_train, cfg.batch_size):
            batch_obs = train_obs[i:i+cfg.batch_size]
            batch_actions = train_actions[i:i+cfg.batch_size]
            batch_next_obs = train_next_obs[i:i+cfg.batch_size]
            
            key, subkey = jax.random.split(key)
            params, opt_state, loss, metrics = update(
                params, opt_state, batch_obs, batch_actions, batch_next_obs, subkey
            )
            epoch_metrics.append(metrics)
        
        # Average metrics (convert to floats here, outside JIT)
        avg_metrics = {k: float(np.mean([float(m[k]) for m in epoch_metrics])) 
                       for k in epoch_metrics[0].keys()}
        history['train'].append(avg_metrics)
        
        # Validation
        key, subkey = jax.random.split(key)
        val_loss, val_metrics = total_loss(
            model, params, val_obs, val_actions, val_next_obs, subkey, cfg
        )
        # Convert to floats for history
        val_metrics_float = {k: float(v) for k, v in val_metrics.items()}
        history['val'].append(val_metrics_float)
        
        # Log
        if epoch % 10 == 0 or epoch == cfg.epochs - 1:
            print(f"  Epoch {epoch:3d}: "
                  f"loss={avg_metrics['loss_total']:.4f} "
                  f"(pred={avg_metrics['loss_pred']:.4f}, "
                  f"sens={avg_metrics['sensitivity']:.3f}, "
                  f"sep={avg_metrics['separation']:.3f})")
        
        # Save best
        if val_metrics_float['loss_total'] < best_val_loss:
            best_val_loss = val_metrics_float['loss_total']
            best_params = params
        
        # Save checkpoint
        if epoch % cfg.save_every == 0:
            os.makedirs(cfg.checkpoint_dir, exist_ok=True)
            checkpoint_path = os.path.join(cfg.checkpoint_dir, f'model_epoch{epoch}.pkl')
            with open(checkpoint_path, 'wb') as f:
                pickle.dump({'params': params, 'epoch': epoch, 'metrics': val_metrics_float}, f)
    
    # Save final checkpoint
    final_path = os.path.join(cfg.checkpoint_dir, 'model_final.pkl')
    with open(final_path, 'wb') as f:
        pickle.dump({'params': best_params, 'epoch': cfg.epochs, 'metrics': val_metrics_float}, f)
    print(f"\nSaved final checkpoint to {final_path}")
    
    # Save best checkpoint
    best_path = os.path.join(cfg.checkpoint_dir, 'model_best.pkl')
    with open(best_path, 'wb') as f:
        pickle.dump({'params': best_params, 'epoch': cfg.epochs, 'metrics': val_metrics_float}, f)
    print(f"Saved best checkpoint to {best_path}")
    
    return model, best_params, history


# ============================================================
# VERIFICATION TESTS
# ============================================================

def verify_action_influence(model, params, key, H=30):
    """Test if actions significantly change trajectory."""
    print("\n" + "=" * 60)
    print("VERIFICATION: Action Influence Test")
    print("=" * 60)
    
    # Test from multiple initial states
    test_states = [
        (1.0, 0.0),   # Left side, stationary
        (2.0, 0.5),   # Center, moving right
        (3.0, -0.3),  # Right side, moving left
    ]
    
    all_delta_x = []
    
    for x0, v0 in test_states:
        obs0 = jnp.array([x0, v0])
        
        # Rollout with +1 actions
        obs = obs0
        for t in range(H):
            obs = model.apply(params, obs, jnp.array([1.0]))
        x_plus = float(obs[0])
        
        # Rollout with -1 actions
        obs = obs0
        for t in range(H):
            obs = model.apply(params, obs, jnp.array([-1.0]))
        x_minus = float(obs[0])
        
        delta_x = abs(x_plus - x_minus)
        all_delta_x.append(delta_x)
        
        print(f"  Start ({x0:.1f}, {v0:.1f}): "
              f"+1→{x_plus:.2f}, -1→{x_minus:.2f}, Δx={delta_x:.2f}")
    
    avg_delta = np.mean(all_delta_x)
    print(f"\n  Average Δx = {avg_delta:.2f}")
    
    if avg_delta >= 2.0:
        print("  ✅ PASS: Strong action influence (target: ≥2.0)")
        return True
    else:
        print(f"  ⚠️  WEAK: Δx = {avg_delta:.2f} (target: ≥2.0)")
        return False


def verify_multistep_stability(model, params, key, H=30, n_samples=10):
    """Test if prediction error stays bounded."""
    print("\n" + "=" * 60)
    print("VERIFICATION: Multi-step Stability")
    print("=" * 60)
    
    errors = []
    
    for i in range(n_samples):
        key, subkey = jax.random.split(key)
        x0 = float(jax.random.uniform(subkey, (), minval=0.5, maxval=3.5))
        key, subkey = jax.random.split(key)
        v0 = float(jax.random.uniform(subkey, (), minval=-0.5, maxval=0.5))
        key, subkey = jax.random.split(key)
        actions = jax.random.uniform(subkey, (H, 1), minval=-1.0, maxval=1.0)
        
        # Ground truth
        x_true = [x0]
        v_true = [v0]
        x, v = x0, v0
        for t in range(H):
            x, v = physics_step(x, v, float(actions[t, 0]))
            x_true.append(x)
            v_true.append(v)
        
        # Model prediction
        obs = jnp.array([x0, v0])
        x_pred = [x0]
        for t in range(H):
            obs = model.apply(params, obs, actions[t])
            x_pred.append(float(obs[0]))
        
        # Compute error
        sample_errors = [abs(x_pred[t] - x_true[t]) for t in range(1, H+1)]
        errors.append(sample_errors)
    
    errors = np.array(errors)
    mean_errors = np.mean(errors, axis=0)
    
    print(f"  1-step error: {mean_errors[0]:.4f}")
    print(f"  10-step error: {mean_errors[9]:.4f}")
    print(f"  20-step error: {mean_errors[19]:.4f}")
    print(f"  30-step error: {mean_errors[29]:.4f}")
    
    if mean_errors[29] < 1.0:
        print("  ✅ PASS: Error stays bounded (<1.0)")
        return True
    else:
        print(f"  ⚠️  WARNING: Error = {mean_errors[29]:.2f} at H=30")
        return False


def verify_sensitivity(model, params, key, n_samples=100):
    """Test Jacobian-based action sensitivity."""
    print("\n" + "=" * 60)
    print("VERIFICATION: Action Sensitivity (Jacobian)")
    print("=" * 60)
    
    sensitivities = []
    
    for i in range(n_samples):
        key, subkey = jax.random.split(key)
        x0 = float(jax.random.uniform(subkey, (), minval=0.5, maxval=3.5))
        key, subkey = jax.random.split(key)
        v0 = float(jax.random.uniform(subkey, (), minval=-0.5, maxval=0.5))
        key, subkey = jax.random.split(key)
        a = float(jax.random.uniform(subkey, (), minval=-1.0, maxval=1.0))
        
        obs = jnp.array([x0, v0])
        action = jnp.array([a])
        
        # Compute ∂x_next/∂a
        def predict_x(a_):
            return model.apply(params, obs, a_[None])[0]  # a_ is scalar, need to wrap
        
        dx_da = float(jax.grad(predict_x)(action[0]))  # Pass scalar
        sensitivities.append(abs(dx_da))
    
    avg_sens = np.mean(sensitivities)
    min_sens = np.min(sensitivities)
    
    print(f"  Average |∂x/∂a| = {avg_sens:.4f}")
    print(f"  Min |∂x/∂a| = {min_sens:.4f}")
    
    if min_sens >= 0.05:
        print("  ✅ PASS: All samples have non-trivial action sensitivity")
        return True
    else:
        print("  ⚠️  WARNING: Some samples have near-zero sensitivity")
        return False


# ============================================================
# MAIN
# ============================================================

def main():
    cfg = Config()
    key = jax.random.PRNGKey(42)
    
    # Train
    model, params, history = train(cfg, key)
    
    # Verify
    print("\n" + "=" * 60)
    print("POST-TRAINING VERIFICATION")
    print("=" * 60)
    
    results = {}
    results['action_influence'] = verify_action_influence(model, params, key)
    results['multistep_stability'] = verify_multistep_stability(model, params, key)
    results['sensitivity'] = verify_sensitivity(model, params, key)
    
    # Summary
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    
    passed = sum(1 for v in results.values() if v)
    total = len(results)
    print(f"Passed: {passed}/{total}")
    
    if passed == total:
        print("\n✅ Model is ready for MPPI planning!")
    else:
        print("\n⚠️  Some tests failed. Consider:")
        print("  - Increasing lambda_sens / lambda_contrast")
        print("  - Using more diverse action data")
        print("  - Longer training")
    
    # Save results
    results_path = '/home/ulluboz/pcp-jepa-research/results/phase2/train_verification.json'
    with open(results_path, 'w') as f:
        json.dump({
            'config': cfg.__dict__,
            'verification': {k: str(v) for k, v in results.items()},
            'final_metrics': history['val'][-1] if history['val'] else None
        }, f, indent=2)
    print(f"\nResults saved to {results_path}")


if __name__ == '__main__':
    main()
