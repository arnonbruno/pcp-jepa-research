"""
O3: Event-Localized Uncertainty

Hypothesis: Planners fail because they are overconfident near events.
Uncertainty must spike ONLY near events so planning becomes risk-aware
without global conservatism.

Model additions:
- Gaussian head: μ(z), Σ(z)
- Risk-aware planner: minimize E[cost] + β * risk

Losses:
1. NLL prediction: L_nll = -log p(z_{t+1} | z_t, a_t)
2. Variance shaping: encourage higher variance near events, lower otherwise
"""

import jax
import jax.numpy as jnp
import flax.linen as nn
from typing import Dict, Optional, Tuple
from dataclasses import dataclass


# ============================================================================
# O3 Model Components
# ============================================================================

class GaussianHead(nn.Module):
    """
    Predict Gaussian distribution from latent state.
    
    μ(z), Σ(z) for probabilistic prediction.
    """
    output_dim: int
    hidden_dim: int = 64
    min_std: float = 0.01
    max_std: float = 10.0
    
    @nn.compact
    def __call__(self, z: jnp.ndarray) -> Dict[str, jnp.ndarray]:
        """
        Args:
            z: [..., latent_dim]
        
        Returns:
            mu: [..., output_dim]
            std: [..., output_dim] (diagonal std)
        """
        h = nn.Dense(self.hidden_dim)(z)
        h = nn.relu(h)
        
        mu = nn.Dense(self.output_dim)(h)
        
        # Predict log-std for numerical stability
        log_std = nn.Dense(self.output_dim)(h)
        log_std = jnp.clip(log_std, -5, 5)
        std = jnp.exp(log_std)
        
        # Clamp to reasonable range
        std = jnp.clip(std, self.min_std, self.max_std)
        
        return {'mu': mu, 'std': std}


class O3Model(nn.Module):
    """
    Full O3 model: Encoder + Gaussian Head + Dynamics with uncertainty.
    
    The key is that uncertainty is shaped to be high near events,
    low elsewhere.
    """
    latent_dim: int
    action_dim: int
    obs_dim: int
    hidden_dim: int = 128
    
    def setup(self):
        self.encoder = nn.Dense(self.latent_dim)
        self.gaussian_head = GaussianHead(self.latent_dim, hidden_dim=64)
        self.dynamics = nn.Sequential([
            nn.Dense(self.hidden_dim),
            nn.relu,
            nn.Dense(self.hidden_dim),
            nn.relu,
            nn.Dense(self.latent_dim),
        ])
        self.std_dynamics = nn.Sequential([
            nn.Dense(self.hidden_dim),
            nn.relu,
            nn.Dense(self.latent_dim),
        ])
    
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
            Dictionary with latents, predictions, uncertainties
        """
        B, T, _ = obs.shape
        
        # Encode
        z = self.encoder(obs)
        
        # Predict uncertainty
        gauss = self.gaussian_head(z)
        mu = gauss['mu']
        std = gauss['std']
        
        # Predict next latents
        z_pred = []
        std_pred = []
        
        z_t = z[:, 0]
        for t in range(T - 1):
            # Mean dynamics
            x = jnp.concatenate([z_t, actions[:, t]], axis=-1)
            delta_z = self.dynamics(x)
            z_t = z_t + delta_z
            z_pred.append(z_t)
            
            # Uncertainty dynamics (how uncertainty propagates)
            log_std_t = self.std_dynamics(x)
            std_t = jnp.exp(jnp.clip(log_std_t, -5, 5))
            std_pred.append(std_t)
        
        return {
            'z': z,
            'z_mu': mu,
            'z_std': std,
            'z_pred': jnp.stack(z_pred, axis=1),
            'std_pred': jnp.stack(std_pred, axis=1),
        }
    
    def predict_with_uncertainty(
        self,
        z: jnp.ndarray,
        a: jnp.ndarray,
    ) -> Tuple[jnp.ndarray, jnp.ndarray]:
        """
        Predict next latent with uncertainty.
        
        Returns:
            mu_next, std_next
        """
        x = jnp.concatenate([z, a], axis=-1)
        delta_z = self.dynamics(x)
        mu_next = z + delta_z
        
        log_std = self.std_dynamics(x)
        std_next = jnp.exp(jnp.clip(log_std, -5, 5))
        
        return mu_next, std_next
    
    def sample_latent(
        self,
        mu: jnp.ndarray,
        std: jnp.ndarray,
        key: jax.random.PRNGKey,
    ) -> jnp.ndarray:
        """Sample from latent distribution."""
        eps = jax.random.normal(key, mu.shape)
        return mu + std * eps


# ============================================================================
# O3 Losses
# ============================================================================

def nll_prediction_loss(
    mu_pred: jnp.ndarray,
    std_pred: jnp.ndarray,
    z_true: jnp.ndarray,
) -> jnp.ndarray:
    """
    L_nll = -log p(z_{t+1} | z_t, a_t)
    
    Negative log-likelihood under Gaussian prediction.
    
    Args:
        mu_pred: [B, T, latent_dim] predicted mean
        std_pred: [B, T, latent_dim] predicted std
        z_true: [B, T, latent_dim] true next latent
    
    Returns:
        Scalar NLL loss
    """
    # Gaussian NLL
    var = std_pred ** 2
    
    nll = 0.5 * jnp.log(var) + 0.5 * (z_true - mu_pred) ** 2 / var
    
    return jnp.mean(nll)


def variance_shaping_loss(
    std: jnp.ndarray,
    event_window_labels: jnp.ndarray,
    sigma_min_evt: float = 0.5,
    sigma_max_non: float = 0.3,
) -> jnp.ndarray:
    """
    Event-local calibration shaping.
    
    Encourage higher predicted variance near events, lower otherwise:
    
    L_varshape = E[ŷ_t * ReLU(σ_min_evt - σ_t) + (1-ŷ_t) * ReLU(σ_t - σ_max_non)]
    
    Args:
        std: [B, T, latent_dim] predicted std
        event_window_labels: [B, T] binary event window labels
        sigma_min_evt: minimum std near events
        sigma_max_non: maximum std away from events
    
    Returns:
        Scalar loss
    """
    # Average std across dimensions
    std_avg = jnp.mean(std, axis=-1)  # [B, T]
    
    # Near events: penalize if std < sigma_min_evt
    event_penalty = event_window_labels * jax.nn.relu(sigma_min_evt - std_avg)
    
    # Away from events: penalize if std > sigma_max_non
    non_event_penalty = (1 - event_window_labels) * jax.nn.relu(std_avg - sigma_max_non)
    
    loss = event_penalty + non_event_penalty
    
    return jnp.mean(loss)


def o3_total_loss(
    model: O3Model,
    params: dict,
    obs: jnp.ndarray,
    actions: jnp.ndarray,
    event_window_labels: jnp.ndarray,
    sigma_min_evt: float = 0.5,
    sigma_max_non: float = 0.3,
    lambda_varshape: float = 1.0,
) -> Dict[str, jnp.ndarray]:
    """
    Total O3 loss.
    
    L_O3 = L_nll + λ_varshape * L_varshape
    
    Args:
        model: O3Model
        params: model parameters
        obs: [B, T, obs_dim]
        actions: [B, T, action_dim]
        event_window_labels: [B, T]
        sigma_min_evt: minimum std near events
        sigma_max_non: maximum std away from events
        lambda_varshape: variance shaping weight
    
    Returns:
        Dictionary with loss components
    """
    # Forward pass
    outputs = model.apply(params, obs, actions)
    
    # L1: NLL prediction
    mu_pred = outputs['z_pred']
    std_pred = outputs['std_pred']
    z_true = outputs['z'][:, 1:]  # Next latent targets
    
    L_nll = nll_prediction_loss(mu_pred, std_pred, z_true)
    
    # L2: Variance shaping
    std = outputs['z_std']
    L_varshape = variance_shaping_loss(
        std, event_window_labels, sigma_min_evt, sigma_max_non
    )
    
    # Total
    total = L_nll + lambda_varshape * L_varshape
    
    return {
        'total': total,
        'L_nll': L_nll,
        'L_varshape': L_varshape,
    }


# ============================================================================
# Risk-Aware Planning
# ============================================================================

class RiskAwarePlanner:
    """
    Risk-aware planner using uncertainty.
    
    Minimizes: E[cost] + β * risk
    
    Where risk can be:
    - Variance of cost
    - CVaR (conditional value at risk)
    - Worst-quantile cost
    """
    
    def __init__(
        self,
        model: O3Model,
        params: dict,
        cost_fn: callable,
        horizon: int = 50,
        num_samples: int = 100,
        risk_weight: float = 1.0,
        risk_type: str = 'variance',  # 'variance', 'cvar', 'worst_quantile'
    ):
        self.model = model
        self.params = params
        self.cost_fn = cost_fn
        self.horizon = horizon
        self.num_samples = num_samples
        self.risk_weight = risk_weight
        self.risk_type = risk_type
    
    def evaluate_action_sequence(
        self,
        z0: jnp.ndarray,
        actions: jnp.ndarray,
        key: jax.random.PRNGKey,
    ) -> Tuple[float, float]:
        """
        Evaluate action sequence with uncertainty.
        
        Returns:
            expected_cost, risk
        """
        H = len(actions)
        
        # Sample trajectories
        z = z0
        costs = []
        
        for t in range(H):
            # Predict with uncertainty
            mu, std = self.model.apply(
                self.params, z, actions[t],
                method=self.model.predict_with_uncertainty,
            )
            
            # Sample
            k, key = jax.random.split(key)
            z = self.model.apply(
                self.params, mu, std, k,
                method=self.model.sample_latent,
            )
            
            # Cost
            cost = self.cost_fn(z)
            costs.append(cost)
        
        costs = jnp.array(costs)
        
        # Expected cost
        expected_cost = jnp.mean(costs)
        
        # Risk measure
        if self.risk_type == 'variance':
            risk = jnp.var(costs)
        elif self.risk_type == 'cvar':
            # CVaR at 5% level
            sorted_costs = jnp.sort(costs)
            worst_5pct = sorted_costs[-int(0.05 * len(costs)):]
            risk = jnp.mean(worst_5pct)
        elif self.risk_type == 'worst_quantile':
            # Worst 10% quantile
            sorted_costs = jnp.sort(costs)
            risk = sorted_costs[int(0.9 * len(costs))]
        else:
            risk = jnp.var(costs)
        
        return expected_cost, risk
    
    def plan(
        self,
        z0: jnp.ndarray,
        key: jax.random.PRNGKey,
    ) -> jnp.ndarray:
        """
        Plan action sequence using MPPI with risk awareness.
        
        Args:
            z0: [latent_dim] initial latent
            key: random key
        
        Returns:
            actions: [H, action_dim] planned actions
        """
        # This is a simplified MPPI implementation
        # In practice, you'd use a more sophisticated planner
        
        # Initialize random actions
        actions = jax.random.normal(key, (self.horizon, self.model.action_dim)) * 0.1
        
        # MPPI iterations
        for _ in range(10):
            # Sample perturbations
            k, key = jax.random.split(key)
            perturbations = jax.random.normal(k, (self.num_samples, self.horizon, self.model.action_dim)) * 0.1
            
            # Evaluate samples
            sample_actions = actions + perturbations
            costs = []
            
            for i in range(self.num_samples):
                k, key = jax.random.split(key)
                exp_cost, risk = self.evaluate_action_sequence(z0, sample_actions[i], k)
                total_cost = exp_cost + self.risk_weight * risk
                costs.append(total_cost)
            
            costs = jnp.array(costs)
            
            # MPPI weights (softmax)
            weights = jax.nn.softmax(-costs)
            
            # Update actions
            actions = jnp.sum(weights[:, None, None] * sample_actions, axis=0)
        
        return actions


# ============================================================================
# Ablation Variants
# ============================================================================

class O3AblationA(nn.Module):
    """
    O3-a: No varshape (plain NLL).
    
    Baseline: standard probabilistic prediction without event-aware uncertainty.
    """
    latent_dim: int
    action_dim: int
    hidden_dim: int = 128
    
    def setup(self):
        self.encoder = nn.Dense(self.latent_dim)
        self.gaussian_head = GaussianHead(self.latent_dim)
        self.dynamics = nn.Sequential([
            nn.Dense(self.hidden_dim),
            nn.relu,
            nn.Dense(self.hidden_dim),
            nn.relu,
            nn.Dense(self.latent_dim),
        ])
    
    def __call__(self, obs, actions):
        z = self.encoder(obs)
        gauss = self.gaussian_head(z)
        
        z_pred = []
        z_t = z[:, 0]
        for t in range(obs.shape[1] - 1):
            x = jnp.concatenate([z_t, actions[:, t]], axis=-1)
            z_t = z_t + self.dynamics(x)
            z_pred.append(z_t)
        
        return {
            'z': z,
            'z_mu': gauss['mu'],
            'z_std': gauss['std'],
            'z_pred': jnp.stack(z_pred, axis=1),
        }


class O3AblationB(nn.Module):
    """
    O3-b: Varshape but risk-neutral planner.
    
    Tests whether uncertainty shaping alone helps without risk-aware planning.
    """
    latent_dim: int
    action_dim: int
    hidden_dim: int = 128
    
    def setup(self):
        self.encoder = nn.Dense(self.latent_dim)
        self.gaussian_head = GaussianHead(self.latent_dim)
        self.dynamics = nn.Sequential([
            nn.Dense(self.hidden_dim),
            nn.relu,
            nn.Dense(self.hidden_dim),
            nn.relu,
            nn.Dense(self.latent_dim),
        ])
        self.std_dynamics = nn.Sequential([
            nn.Dense(self.hidden_dim),
            nn.relu,
            nn.Dense(self.latent_dim),
        ])
    
    def __call__(self, obs, actions):
        z = self.encoder(obs)
        gauss = self.gaussian_head(z)
        
        z_pred = []
        std_pred = []
        z_t = z[:, 0]
        for t in range(obs.shape[1] - 1):
            x = jnp.concatenate([z_t, actions[:, t]], axis=-1)
            z_t = z_t + self.dynamics(x)
            z_pred.append(z_t)
            
            log_std = self.std_dynamics(x)
            std_pred.append(jnp.exp(jnp.clip(log_std, -5, 5)))
        
        return {
            'z': z,
            'z_mu': gauss['mu'],
            'z_std': gauss['std'],
            'z_pred': jnp.stack(z_pred, axis=1),
            'std_pred': jnp.stack(std_pred, axis=1),
        }


# ============================================================================
# Calibration Metrics
# ============================================================================

def compute_calibration(
    mu_pred: jnp.ndarray,
    std_pred: jnp.ndarray,
    z_true: jnp.ndarray,
    event_labels: jnp.ndarray,
) -> Dict[str, float]:
    """
    Compute calibration metrics separately for event vs non-event segments.
    
    Args:
        mu_pred: [B, T, latent_dim] predicted mean
        std_pred: [B, T, latent_dim] predicted std
        z_true: [B, T, latent_dim] true values
        event_labels: [B, T] event labels
    
    Returns:
        Dictionary with calibration metrics
    """
    # Z-scores
    z_scores = (z_true - mu_pred) / std_pred
    
    # Event segments
    event_mask = event_labels > 0.5
    non_event_mask = ~event_mask
    
    # Calibration (fraction within 1σ, 2σ, etc.)
    within_1sigma = jnp.abs(z_scores) < 1.0
    within_2sigma = jnp.abs(z_scores) < 2.0
    
    # Event calibration
    event_within_1 = jnp.mean(jnp.where(event_mask, within_1sigma, 0.0))
    event_within_2 = jnp.mean(jnp.where(event_mask, within_2sigma, 0.0))
    
    # Non-event calibration
    non_event_within_1 = jnp.mean(jnp.where(non_event_mask, within_1sigma, 0.0))
    non_event_within_2 = jnp.mean(jnp.where(non_event_mask, within_2sigma, 0.0))
    
    return {
        'event_within_1sigma': float(event_within_1),
        'event_within_2sigma': float(event_within_2),
        'non_event_within_1sigma': float(non_event_within_1),
        'non_event_within_2sigma': float(non_event_within_2),
        'event_std_mean': float(jnp.mean(jnp.where(event_mask[..., None], std_pred, jnp.nan))),
        'non_event_std_mean': float(jnp.mean(jnp.where(non_event_mask[..., None], std_pred, jnp.nan))),
    }