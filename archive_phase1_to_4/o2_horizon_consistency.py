"""
O2: Horizon-Consistency / Planning-Consistency Objective

Hypothesis: Even with event tokens, latents can be "bad for planning."
Directly optimizing planning-consistency yields latents with stable
long-horizon geometry.

Key challenge: "optimal actions" aren't given.
Solution: Self-improvement with counterfactual evaluation.

Mechanism:
1. Plan actions: a* = MPC(z_t)
2. Execute in simulator → J_true(a*)
3. Compare to model → J_model(a*)
4. Penalize mismatch, especially near events
"""

import jax
import jax.numpy as jnp
import flax.linen as nn
from typing import Dict, Optional, Tuple, List, Callable
from dataclasses import dataclass


# ============================================================================
# O2 Model Components
# ============================================================================

class CostHead(nn.Module):
    """
    Predict cost from latent state.
    
    Used for planning-consistency loss.
    """
    hidden_dim: int = 64
    
    @nn.compact
    def __call__(self, z: jnp.ndarray) -> jnp.ndarray:
        """
        Args:
            z: [..., latent_dim]
        
        Returns:
            cost: [...] scalar cost
        """
        h = nn.Dense(self.hidden_dim)(z)
        h = nn.relu(h)
        cost = nn.Dense(1)(h)
        return cost.squeeze(-1)


class O2Model(nn.Module):
    """
    Full O2 model: Encoder + Dynamics + Cost Head.
    
    The cost head enables planning-consistency training.
    """
    latent_dim: int
    action_dim: int
    obs_dim: int
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
        self.cost_head = CostHead(hidden_dim=64)
    
    def __call__(
        self,
        obs: jnp.ndarray,
        actions: jnp.ndarray,
    ) -> Dict[str, jnp.ndarray]:
        """
        Args:
            obs: [B, T, obs_dim]
            actions: [B, T, action_dim]
        
        Returns:
            Dictionary with latents, predictions, costs
        """
        B, T, _ = obs.shape
        
        # Encode
        z = self.encoder(obs)
        
        # Predict costs
        costs = self.cost_head(z)
        
        # Predict next latents
        z_pred = []
        z_t = z[:, 0]
        for t in range(T - 1):
            x = jnp.concatenate([z_t, actions[:, t]], axis=-1)
            delta_z = self.dynamics(x)
            z_t = z_t + delta_z
            z_pred.append(z_t)
        
        z_pred = jnp.stack(z_pred, axis=1)
        
        # Predicted costs
        costs_pred = self.cost_head(z_pred)
        
        return {
            'z': z,
            'z_pred': z_pred,
            'costs': costs,
            'costs_pred': costs_pred,
        }
    
    def encode(self, obs: jnp.ndarray) -> jnp.ndarray:
        """Encode observation."""
        return self.encoder(obs)
    
    def predict_cost(self, z: jnp.ndarray) -> jnp.ndarray:
        """Predict cost from latent."""
        return self.cost_head(z)
    
    def predict_next(self, z: jnp.ndarray, a: jnp.ndarray) -> jnp.ndarray:
        """Predict next latent."""
        x = jnp.concatenate([z, a], axis=-1)
        delta_z = self.dynamics(x)
        return z + delta_z


# ============================================================================
# O2 Losses
# ============================================================================

def trajectory_cost_mismatch_loss(
    J_true: jnp.ndarray,
    J_model: jnp.ndarray,
) -> jnp.ndarray:
    """
    L_cost = (J_true(a*) - J_model(a*))^2
    
    Penalize mismatch between true and predicted trajectory costs.
    
    Args:
        J_true: [B] true costs from simulator
        J_model: [B] model-predicted costs
    
    Returns:
        Scalar loss
    """
    return jnp.mean((J_true - J_model) ** 2)


def event_window_weighted_loss(
    losses: jnp.ndarray,
    event_window_labels: jnp.ndarray,
    weight: float = 2.0,
) -> jnp.ndarray:
    """
    Reweight losses near event windows.
    
    Events dominate failures, so weight them more.
    
    Args:
        losses: [B, T] per-timestep losses
        event_window_labels: [B, T] binary labels
        weight: multiplier for event windows
    
    Returns:
        Weighted loss
    """
    weights = jnp.where(event_window_labels > 0.5, weight, 1.0)
    weighted_losses = losses * weights
    return jnp.mean(weighted_losses)


def gradient_alignment_loss(
    model: O2Model,
    params: dict,
    z: jnp.ndarray,
    a: jnp.ndarray,
    true_gradient: jnp.ndarray,
    eps: float = 1e-4,
) -> jnp.ndarray:
    """
    Encourage ∂J_model/∂a to match true gradient.
    
    Optional, advanced loss. Use only if stable.
    
    Args:
        model: O2Model
        params: model parameters
        z: latent state
        a: action
        true_gradient: estimated gradient from simulator
        eps: finite difference step
    
    Returns:
        Scalar loss
    """
    def cost_fn(a):
        z_next = model.apply(params, z, a, method=model.predict_next)
        return model.apply(params, z_next, method=model.predict_cost)
    
    # Model gradient
    model_grad = jax.grad(cost_fn)(a)
    
    # Alignment loss
    alignment = jnp.sum(model_grad * true_gradient) / (
        jnp.linalg.norm(model_grad) * jnp.linalg.norm(true_gradient) + 1e-8
    )
    
    # We want high alignment (close to 1), so minimize 1 - alignment
    return 1.0 - alignment


# ============================================================================
# Planning Loop for O2
# ============================================================================

class MPPIPlanner:
    """
    MPPI planner for O2 training.
    
    Used to generate planned actions for self-improvement.
    """
    
    def __init__(
        self,
        model: O2Model,
        params: dict,
        horizon: int = 50,
        num_samples: int = 100,
        temperature: float = 1.0,
        action_dim: int = None,
    ):
        self.model = model
        self.params = params
        self.horizon = horizon
        self.num_samples = num_samples
        self.temperature = temperature
        self.action_dim = action_dim
    
    def plan(
        self,
        z0: jnp.ndarray,
        key: jax.random.PRNGKey,
    ) -> jnp.ndarray:
        """
        Plan action sequence using MPPI.
        
        Args:
            z0: [latent_dim] initial latent
            key: random key
        
        Returns:
            actions: [H, action_dim] planned actions
        """
        # Initialize with zeros
        actions = jnp.zeros((self.horizon, self.action_dim))
        
        # Sample perturbations
        k, key = jax.random.split(key)
        perturbations = jax.random.normal(
            k, (self.num_samples, self.horizon, self.action_dim)
        ) * 0.3
        
        # Evaluate samples
        sample_actions = actions + perturbations
        costs = []
        
        for i in range(self.num_samples):
            trajectory_cost = self._evaluate_trajectory(z0, sample_actions[i])
            costs.append(trajectory_cost)
        
        costs = jnp.array(costs)
        
        # MPPI weights (softmax with temperature)
        weights = jax.nn.softmax(-costs / self.temperature)
        
        # Update actions
        actions = jnp.sum(weights[:, None, None] * sample_actions, axis=0)
        
        return actions
    
    def _evaluate_trajectory(
        self,
        z0: jnp.ndarray,
        actions: jnp.ndarray,
    ) -> float:
        """Evaluate trajectory cost."""
        z = z0
        total_cost = 0.0
        
        for t in range(self.horizon):
            cost = self.model.apply(self.params, z, method=self.model.predict_cost)
            total_cost = total_cost + cost
            
            z = self.model.apply(self.params, z, actions[t], method=self.model.predict_next)
        
        return total_cost


# ============================================================================
# O2 Training Loop
# ============================================================================

def o2_training_step(
    model: O2Model,
    params: dict,
    obs: jnp.ndarray,
    actions: jnp.ndarray,
    true_costs: jnp.ndarray,
    event_window_labels: jnp.ndarray,
    planner: MPPIPlanner,
    key: jax.random.PRNGKey,
    use_event_weighting: bool = True,
    use_gradient_alignment: bool = False,
) -> Dict[str, jnp.ndarray]:
    """
    Single O2 training step with self-improvement.
    
    Steps:
    1. Encode observations
    2. Plan actions with current model
    3. Execute in simulator (or use recorded data)
    4. Compare costs and compute losses
    
    Args:
        model: O2Model
        params: model parameters
        obs: [B, T, obs_dim] observations
        actions: [B, T, action_dim] actions (from planner or recorded)
        true_costs: [B, T] true costs from simulator
        event_window_labels: [B, T] event labels
        planner: MPPIPlanner for generating actions
        key: random key
        use_event_weighting: whether to weight event windows more
        use_gradient_alignment: whether to use gradient alignment
    
    Returns:
        Dictionary with losses
    """
    B, T, _ = obs.shape
    
    # Forward pass
    outputs = model.apply(params, obs, actions)
    
    # Model-predicted costs
    J_model = outputs['costs']  # [B, T]
    J_model_pred = outputs['costs_pred']  # [B, T-1]
    
    # Trajectory cost mismatch
    J_true_total = jnp.sum(true_costs, axis=1)  # [B]
    J_model_total = jnp.sum(J_model, axis=1)  # [B]
    
    L_cost = trajectory_cost_mismatch_loss(J_true_total, J_model_total)
    
    # Event-window weighted per-step loss
    if use_event_weighting:
        step_losses = (true_costs[:, :-1] - J_model_pred) ** 2
        L_weighted = event_window_weighted_loss(step_losses, event_window_labels[:, :-1])
    else:
        L_weighted = L_cost
    
    # Gradient alignment (optional)
    if use_gradient_alignment:
        # Estimate true gradient via finite difference
        # (In practice, this would come from simulator)
        L_grad = 0.0  # Placeholder
    else:
        L_grad = 0.0
    
    total = L_weighted + 0.1 * L_grad
    
    return {
        'total': total,
        'L_cost': L_cost,
        'L_weighted': L_weighted,
        'L_grad': L_grad,
    }


# ============================================================================
# Ablation Variants
# ============================================================================

class O2AblationA(nn.Module):
    """
    O2-a: No event-window weighting.
    
    Tests whether event-specific weighting matters.
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
        self.cost_head = CostHead(hidden_dim=64)
    
    def __call__(self, obs, actions):
        z = self.encoder(obs)
        costs = self.cost_head(z)
        
        z_pred = []
        z_t = z[:, 0]
        for t in range(obs.shape[1] - 1):
            x = jnp.concatenate([z_t, actions[:, t]], axis=-1)
            z_t = z_t + self.dynamics(x)
            z_pred.append(z_t)
        
        return {
            'z': z,
            'z_pred': jnp.stack(z_pred, axis=1),
            'costs': costs,
        }


class O2AblationB(nn.Module):
    """
    O2-b: No cost head (only dynamics).
    
    Tests whether cost prediction matters for planning-consistency.
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
        
        z_pred = []
        z_t = z[:, 0]
        for t in range(obs.shape[1] - 1):
            x = jnp.concatenate([z_t, actions[:, t]], axis=-1)
            z_t = z_t + self.dynamics(x)
            z_pred.append(z_t)
        
        return {
            'z': z,
            'z_pred': jnp.stack(z_pred, axis=1),
        }


class O2AblationC(nn.Module):
    """
    O2-c: Random actions (tests that "planning" signal matters).
    
    Uses random actions instead of MPC-planned actions.
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
        self.cost_head = CostHead(hidden_dim=64)
    
    def __call__(self, obs, actions):
        z = self.encoder(obs)
        costs = self.cost_head(z)
        
        z_pred = []
        z_t = z[:, 0]
        for t in range(obs.shape[1] - 1):
            x = jnp.concatenate([z_t, actions[:, t]], axis=-1)
            z_t = z_t + self.dynamics(x)
            z_pred.append(z_t)
        
        return {
            'z': z,
            'z_pred': jnp.stack(z_pred, axis=1),
            'costs': costs,
        }


# ============================================================================
# Helper Functions
# ============================================================================

def create_o2_model(
    latent_dim: int,
    action_dim: int,
    obs_dim: int,
    hidden_dim: int = 128,
    key: jax.random.PRNGKey = None,
) -> Tuple[O2Model, dict]:
    """Create and initialize O2 model."""
    model = O2Model(
        latent_dim=latent_dim,
        action_dim=action_dim,
        obs_dim=obs_dim,
        hidden_dim=hidden_dim,
    )
    
    obs = jnp.zeros((1, 10, obs_dim))
    actions = jnp.zeros((1, 10, action_dim))
    
    params = model.init(key, obs, actions)
    
    return model, params


def compute_planning_regret(
    J_planned: float,
    J_optimal: float,
) -> float:
    """
    Compute planning regret.
    
    Regret = J(planned) - J(optimal)
    
    This is the core metric for planning-consistency.
    """
    return J_planned - J_optimal