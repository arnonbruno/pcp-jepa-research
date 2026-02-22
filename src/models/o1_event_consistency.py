"""
O1: Event-Consistency Objective

Hypothesis: Making the latent explicitly event-consistent extends horizon
by preventing wrong-mode rollouts.

Model additions:
- Event head: p_t = σ(h(z_t))
- Event-conditioned dynamics: f(z_t, a_t, p_t)

Losses:
1. Event classification: L_evt = BCE(p_t, ŷ_t)
2. Event timing: Soft-DTW on event probabilities
3. Event-consistent rollouts: L_seq = Σ BCE(p_{t+i}^{rollout}, ŷ_{t+i})
"""

import jax
import jax.numpy as jnp
import flax.linen as nn
from typing import Dict, Optional, Tuple, List
from dataclasses import dataclass
import optax


# ============================================================================
# O1 Model Components
# ============================================================================

class EventHead(nn.Module):
    """
    Predict event probability from latent state.
    
    p_t = σ(h(z_t))
    
    Can also predict event type (multi-class) if needed.
    """
    hidden_dim: int = 64
    num_event_types: int = 1  # 1 = binary, >1 = multi-class
    
    @nn.compact
    def __call__(self, z: jnp.ndarray) -> Dict[str, jnp.ndarray]:
        """
        Args:
            z: [..., latent_dim] latent state
        
        Returns:
            Dictionary with 'event_prob' and optionally 'event_type_logits'
        """
        h = nn.Dense(self.hidden_dim)(z)
        h = nn.relu(h)
        
        # Event probability (binary)
        event_logit = nn.Dense(1)(h)
        event_prob = jax.nn.sigmoid(event_logit)
        
        outputs = {
            'event_logit': event_logit.squeeze(-1),
            'event_prob': event_prob.squeeze(-1),
        }
        
        # Event type (optional)
        if self.num_event_types > 1:
            type_logits = nn.Dense(self.num_event_types)(h)
            outputs['event_type_logits'] = type_logits
        
        return outputs


class EventConditionedDynamics(nn.Module):
    """
    Latent dynamics conditioned on event probability.
    
    z_{t+1} = f(z_t, a_t, p_t)
    
    The event probability acts as a "mode indicator" for dynamics.
    """
    latent_dim: int
    hidden_dim: int = 128
    
    @nn.compact
    def __call__(
        self,
        z: jnp.ndarray,
        a: jnp.ndarray,
        event_prob: jnp.ndarray,
    ) -> jnp.ndarray:
        """
        Args:
            z: [..., latent_dim] current latent
            a: [..., action_dim] action
            event_prob: [...] event probability (scalar per batch)
        
        Returns:
            z_next: [..., latent_dim] next latent
        """
        # Concatenate z, a, event_prob
        event_prob = event_prob[..., None]  # [..., 1]
        x = jnp.concatenate([z, a, event_prob], axis=-1)
        
        # Dynamics network
        h = nn.Dense(self.hidden_dim)(x)
        h = nn.relu(h)
        h = nn.Dense(self.hidden_dim)(h)
        h = nn.relu(h)
        
        # Residual update
        delta_z = nn.Dense(self.latent_dim)(h)
        z_next = z + delta_z
        
        return z_next


class O1Model(nn.Module):
    """
    Full O1 model: Encoder + Event Head + Event-Conditioned Dynamics.
    """
    latent_dim: int
    action_dim: int
    hidden_dim: int = 128
    num_event_types: int = 1
    
    def setup(self):
        self.encoder = nn.Dense(self.latent_dim)
        self.event_head = EventHead(hidden_dim=64, num_event_types=self.num_event_types)
        self.dynamics = EventConditionedDynamics(
            latent_dim=self.latent_dim,
            hidden_dim=self.hidden_dim,
        )
        self.decoder = nn.Dense(self.hidden_dim)  # Optional reconstruction
    
    def __call__(
        self,
        obs: jnp.ndarray,
        actions: jnp.ndarray,
        return_event_probs: bool = True,
    ) -> Dict[str, jnp.ndarray]:
        """
        Args:
            obs: [B, T, obs_dim] observations
            actions: [B, T, action_dim] actions
            return_event_probs: whether to return event probabilities
        
        Returns:
            Dictionary with latents, event_probs, predictions
        """
        B, T, _ = obs.shape
        
        # Encode observations
        z = self.encoder(obs)  # [B, T, latent_dim]
        
        # Predict events
        event_outputs = self.event_head(z)  # [B, T]
        event_probs = event_outputs['event_prob']
        
        # Rollout dynamics
        z_pred = []
        z_t = z[:, 0]
        
        for t in range(T - 1):
            a_t = actions[:, t]
            p_t = event_probs[:, t]
            z_t = self.dynamics(z_t, a_t, p_t)
            z_pred.append(z_t)
        
        z_pred = jnp.stack(z_pred, axis=1)  # [B, T-1, latent_dim]
        
        outputs = {
            'z': z,
            'z_pred': z_pred,
            'event_probs': event_probs,
            'event_logits': event_outputs['event_logit'],
        }
        
        if 'event_type_logits' in event_outputs:
            outputs['event_type_logits'] = event_outputs['event_type_logits']
        
        return outputs
    
    def encode(self, obs: jnp.ndarray) -> jnp.ndarray:
        """Encode single observation."""
        return self.encoder(obs)
    
    def predict_event(self, z: jnp.ndarray) -> jnp.ndarray:
        """Predict event probability from latent."""
        return self.event_head(z)['event_prob']
    
    def step(
        self,
        z: jnp.ndarray,
        a: jnp.ndarray,
        event_prob: Optional[jnp.ndarray] = None,
    ) -> Tuple[jnp.ndarray, jnp.ndarray]:
        """
        Single-step prediction.
        
        Args:
            z: [..., latent_dim] latent state
            a: [..., action_dim] action
            event_prob: optional event probability (will predict if None)
        
        Returns:
            z_next, event_prob
        """
        if event_prob is None:
            event_prob = self.predict_event(z)
        
        z_next = self.dynamics(z, a, event_prob)
        return z_next, event_prob
    
    def rollout(
        self,
        z0: jnp.ndarray,
        actions: jnp.ndarray,
    ) -> Tuple[jnp.ndarray, jnp.ndarray]:
        """
        Multi-step rollout.
        
        Args:
            z0: [B, latent_dim] initial latent
            actions: [B, H, action_dim] action sequence
        
        Returns:
            z_trajectory: [B, H, latent_dim]
            event_probs: [B, H]
        """
        B, H, _ = actions.shape
        
        z_trajectory = []
        event_probs = []
        
        z = z0
        for t in range(H):
            p = self.predict_event(z)
            event_probs.append(p)
            z_trajectory.append(z)
            
            z, _ = self.step(z, actions[:, t], p)
        
        return (
            jnp.stack(z_trajectory, axis=1),
            jnp.stack(event_probs, axis=1),
        )


# ============================================================================
# O1 Losses
# ============================================================================

def event_classification_loss(
    event_probs: jnp.ndarray,
    event_labels: jnp.ndarray,
    pos_weight: float = 1.0,
) -> jnp.ndarray:
    """
    L_evt = BCE(p_t, ŷ_t) with class balancing.
    
    Args:
        event_probs: [B, T] predicted event probabilities
        event_labels: [B, T] event window labels ŷ_t
        pos_weight: weight for positive class (for class balancing)
    
    Returns:
        Scalar loss
    """
    # Binary cross-entropy with class weighting
    loss = optax.sigmoid_binary_cross_entropy(
        event_probs,
        event_labels,
    )
    
    # Weight positive examples more
    weights = jnp.where(event_labels > 0.5, pos_weight, 1.0)
    loss = loss * weights
    
    return jnp.mean(loss)


def event_timing_loss_sdtw(
    event_probs: jnp.ndarray,
    event_labels: jnp.ndarray,
    gamma: float = 1.0,
) -> jnp.ndarray:
    """
    Soft-DTW loss for event timing.
    
    Penalizes misalignment in event probability sequences.
    
    Args:
        event_probs: [B, T] predicted event probabilities
        event_labels: [B, T] event labels
        gamma: soft-DTW temperature
    
    Returns:
        Scalar loss
    """
    B, T = event_probs.shape
    
    # Simple approximation: compare cumulative event signals
    # (Full soft-DTW is expensive; this captures timing alignment)
    
    pred_cumsum = jnp.cumsum(event_probs, axis=1)
    true_cumsum = jnp.cumsum(event_labels, axis=1)
    
    # L2 distance between cumulative signals
    loss = jnp.mean((pred_cumsum - true_cumsum) ** 2)
    
    return loss


def event_consistent_rollout_loss(
    model: O1Model,
    params: dict,
    z0: jnp.ndarray,
    actions: jnp.ndarray,
    event_labels: jnp.ndarray,
    horizon: int = 10,
) -> jnp.ndarray:
    """
    L_seq = Σ BCE(p_{t+i}^{rollout}, ŷ_{t+i})
    
    Penalize mismatch in event probability sequence during rollout.
    
    Args:
        model: O1Model
        params: model parameters
        z0: [B, latent_dim] initial latent
        actions: [B, H, action_dim] planned actions
        event_labels: [B, H] event labels for rollout horizon
        horizon: rollout length
    
    Returns:
        Scalar loss
    """
    # Rollout
    def rollout_step(carry, action):
        z = carry
        p = model.apply(params, z, method=model.predict_event)
        z_next = model.apply(params, z, action, p, method=model.step)[0]
        return z_next, p
    
    _, event_probs_seq = jax.lax.scan(
        rollout_step,
        z0,
        actions.transpose(1, 0, 2),  # [H, B, action_dim]
    )
    
    # event_probs_seq: [H, B]
    event_probs_seq = event_probs_seq.transpose(1, 0)  # [B, H]
    
    # BCE loss
    loss = event_classification_loss(event_probs_seq, event_labels)
    
    return loss


def o1_total_loss(
    model: O1Model,
    params: dict,
    obs: jnp.ndarray,
    actions: jnp.ndarray,
    event_labels: jnp.ndarray,
    event_window_labels: jnp.ndarray,
    pos_weight: float = 1.0,
    lambda_timing: float = 0.1,
    lambda_seq: float = 0.1,
) -> Dict[str, jnp.ndarray]:
    """
    Total O1 loss.
    
    L_O1 = L_evt + λ_timing * L_timing + λ_seq * L_seq
    
    Args:
        model: O1Model
        params: model parameters
        obs: [B, T, obs_dim] observations
        actions: [B, T, action_dim] actions
        event_labels: [B, T] event labels y_t
        event_window_labels: [B, T] event window labels ŷ_t
        pos_weight: positive class weight
        lambda_timing: timing loss weight
        lambda_seq: sequence consistency weight
    
    Returns:
        Dictionary with loss components
    """
    # Forward pass
    outputs = model.apply(params, obs, actions)
    
    event_probs = outputs['event_probs']
    
    # L1: Event classification
    L_evt = event_classification_loss(event_probs, event_window_labels, pos_weight)
    
    # L2: Event timing
    L_timing = event_timing_loss_sdtw(event_probs, event_labels)
    
    # L3: Event-consistent rollout (sample initial states)
    B, T, _ = obs.shape
    H = min(10, T - 1)
    
    # Use first H-1 actions for rollout
    z0 = outputs['z'][:, 0]
    rollout_actions = actions[:, :H]
    rollout_labels = event_window_labels[:, 1:H+1]
    
    L_seq = event_consistent_rollout_loss(
        model, params, z0, rollout_actions, rollout_labels, H
    )
    
    # Total
    total = L_evt + lambda_timing * L_timing + lambda_seq * L_seq
    
    return {
        'total': total,
        'L_evt': L_evt,
        'L_timing': L_timing,
        'L_seq': L_seq,
    }


# ============================================================================
# Ablation Variants
# ============================================================================

class O1AblationA(nn.Module):
    """
    O1-a: No event head, no event conditioning.
    
    Baseline: plain predictive latent without event awareness.
    """
    latent_dim: int
    action_dim: int
    hidden_dim: int = 128
    
    def setup(self):
        self.encoder = nn.Dense(self.latent_dim)
        self.dynamics = nn.Sequential([
            nn.Dense(self.hidden_dim),
            nn.relu,
            nn.Dense(self.hidden_dim),
            nn.relu,
            nn.Dense(self.latent_dim),
        ])
    
    def __call__(self, obs, actions):
        z = self.encoder(obs)
        
        # Plain dynamics (no event conditioning)
        z_pred = []
        z_t = z[:, 0]
        for t in range(obs.shape[1] - 1):
            x = jnp.concatenate([z_t, actions[:, t]], axis=-1)
            z_t = z_t + self.dynamics(x)
            z_pred.append(z_t)
        
        return {'z': z, 'z_pred': jnp.stack(z_pred, axis=1)}


class O1AblationB(nn.Module):
    """
    O1-b: Event head but NOT fed into dynamics.
    
    Tests whether event prediction alone (without dynamics conditioning) helps.
    """
    latent_dim: int
    action_dim: int
    hidden_dim: int = 128
    
    def setup(self):
        self.encoder = nn.Dense(self.latent_dim)
        self.event_head = EventHead(hidden_dim=64)
        self.dynamics = nn.Sequential([
            nn.Dense(self.hidden_dim),
            nn.relu,
            nn.Dense(self.hidden_dim),
            nn.relu,
            nn.Dense(self.latent_dim),
        ])
    
    def __call__(self, obs, actions):
        z = self.encoder(obs)
        event_probs = self.event_head(z)['event_prob']
        
        # Dynamics WITHOUT event conditioning
        z_pred = []
        z_t = z[:, 0]
        for t in range(obs.shape[1] - 1):
            x = jnp.concatenate([z_t, actions[:, t]], axis=-1)
            z_t = z_t + self.dynamics(x)
            z_pred.append(z_t)
        
        return {
            'z': z,
            'z_pred': jnp.stack(z_pred, axis=1),
            'event_probs': event_probs,
        }


class O1AblationC(nn.Module):
    """
    O1-c: Event conditioning but NO timing loss.
    
    Tests whether timing loss is necessary.
    """
    latent_dim: int
    action_dim: int
    hidden_dim: int = 128
    
    def setup(self):
        self.encoder = nn.Dense(self.latent_dim)
        self.event_head = EventHead(hidden_dim=64)
        self.dynamics = EventConditionedDynamics(self.latent_dim, self.hidden_dim)
    
    def __call__(self, obs, actions):
        z = self.encoder(obs)
        event_probs = self.event_head(z)['event_prob']
        
        z_pred = []
        z_t = z[:, 0]
        for t in range(obs.shape[1] - 1):
            z_t = self.dynamics(z_t, actions[:, t], event_probs[:, t])
            z_pred.append(z_t)
        
        return {
            'z': z,
            'z_pred': jnp.stack(z_pred, axis=1),
            'event_probs': event_probs,
        }


# ============================================================================
# Training
# ============================================================================

def create_o1_model(
    latent_dim: int,
    action_dim: int,
    obs_dim: int,
    hidden_dim: int = 128,
    key: jax.random.PRNGKey = None,
) -> Tuple[O1Model, dict]:
    """Create and initialize O1 model."""
    model = O1Model(
        latent_dim=latent_dim,
        action_dim=action_dim,
        hidden_dim=hidden_dim,
    )
    
    # Dummy inputs for initialization
    obs = jnp.zeros((1, 10, obs_dim))
    actions = jnp.zeros((1, 10, action_dim))
    
    params = model.init(key, obs, actions)
    
    return model, params


def train_step_o1(
    model: O1Model,
    state,
    obs: jnp.ndarray,
    actions: jnp.ndarray,
    event_labels: jnp.ndarray,
    event_window_labels: jnp.ndarray,
    pos_weight: float,
):
    """Single training step for O1."""
    
    def loss_fn(params):
        losses = o1_total_loss(
            model, params, obs, actions,
            event_labels, event_window_labels, pos_weight,
        )
        return losses['total'], losses
    
    (loss, losses), grads = jax.value_and_grad(loss_fn, has_aux=True)(state.params)
    state = state.apply_gradients(grads=grads)
    
    return state, losses