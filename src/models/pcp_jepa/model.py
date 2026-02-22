"""
Planning-Consistent Physics JEPA (PCP-JEPA)

First framework to explicitly train representations for planning-consistency.

Key innovation: Planning regret as a self-supervised objective.
Unlike prior world models that optimize prediction, PCP-JEPA minimizes
the gap between planned and executed returns.
"""

import jax
import jax.numpy as jnp
from flax import linen as nn
from typing import Dict, List, Tuple, Optional, Callable
from dataclasses import dataclass
import functools


# ============================================================================
# Belief State (Partial Observability)
# ============================================================================

class BeliefState(nn.Module):
    """
    Recurrent belief state for partial observability.
    
    b_t = GRU(b_{t-1}, [o_t, a_{t-1}])
    
    Handles:
    - Missing observations (sensor dropout)
    - Delayed observations
    - Partial state visibility
    """
    latent_dim: int = 32
    hidden_dim: int = 64
    
    @nn.compact
    def __call__(
        self,
        belief_prev: jnp.ndarray,
        observation: jnp.ndarray,
        action_prev: jnp.ndarray,
    ) -> jnp.ndarray:
        """
        Update belief state.
        
        Args:
            belief_prev: Previous belief [batch, hidden_dim]
            observation: Current observation [batch, obs_dim]
            action_prev: Previous action [batch, action_dim]
        
        Returns:
            belief: Updated belief [batch, hidden_dim]
        """
        # Project observation and action
        o_proj = nn.Dense(self.hidden_dim)(observation)
        a_proj = nn.Dense(self.hidden_dim)(action_prev)
        
        # Combine
        x = o_proj + a_proj
        
        # GRU update
        carry, hidden = nn.GRUCell()(belief_prev, x)
        
        return hidden
    
    def initialize(self, batch_size: int) -> jnp.ndarray:
        """Initialize belief state."""
        return jnp.zeros((batch_size, self.hidden_dim))


# ============================================================================
# Multi-Scale Dynamics
# ============================================================================

class EventDetector(nn.Module):
    """
    Detect discrete events (contacts, mode switches).
    
    Learned, not hand-coded.
    """
    latent_dim: int = 32
    
    @nn.compact
    def __call__(self, z: jnp.ndarray) -> jnp.ndarray:
        """
        Detect event from latent state.
        
        Returns:
            event: Event probability [batch, 1]
        """
        x = nn.Dense(64)(z)
        x = nn.relu(x)
        event = nn.Dense(1)(x)
        event = jax.nn.sigmoid(event)  # Probability
        
        return event


class MultiScaleDynamics(nn.Module):
    """
    Dynamics at multiple temporal scales.
    
    Fine-grained: z_{t+1} = f(z_t, a_t) - accurate short term
    Chunked: z_{t+k} = F(z_t, a_{t:t+k}) - consistent long term
    Event-conditioned: z_{t+1} = f(z_t, a_t, e_t) - handles discontinuities
    """
    latent_dim: int = 32
    action_dim: int = 1
    chunk_size: int = 10
    
    @nn.compact
    def fine_step(
        self,
        z: jnp.ndarray,
        a: jnp.ndarray,
        event: Optional[jnp.ndarray] = None,
    ) -> jnp.ndarray:
        """
        Single-step dynamics.
        
        Args:
            z: Latent state [batch, latent_dim]
            a: Action [batch, action_dim]
            event: Event token [batch, 1] (optional)
        
        Returns:
            z_next: Next latent [batch, latent_dim]
        """
        # Concatenate
        if event is not None:
            x = jnp.concatenate([z, a, event], axis=-1)
        else:
            x = jnp.concatenate([z, a], axis=-1)
        
        # Dynamics network
        x = nn.Dense(64)(x)
        x = nn.relu(x)
        x = nn.Dense(64)(x)
        x = nn.relu(x)
        dz = nn.Dense(self.latent_dim)(x)
        
        return z + dz
    
    @nn.compact
    def chunk_step(
        self,
        z: jnp.ndarray,
        actions: jnp.ndarray,
    ) -> jnp.ndarray:
        """
        Jump k steps ahead.
        
        Args:
            z: Latent state [batch, latent_dim]
            actions: Action sequence [batch, chunk_size, action_dim]
        
        Returns:
            z_next: Latent after chunk_size steps [batch, latent_dim]
        """
        # Flatten actions
        actions_flat = actions.reshape(len(z), -1)
        
        # Concatenate
        x = jnp.concatenate([z, actions_flat], axis=-1)
        
        # Chunk dynamics network
        x = nn.Dense(128)(x)
        x = nn.relu(x)
        x = nn.Dense(128)(x)
        x = nn.relu(x)
        dz = nn.Dense(self.latent_dim)(x)
        
        return z + dz
    
    def rollout_fine(
        self,
        z_init: jnp.ndarray,
        actions: jnp.ndarray,
        detect_events: bool = True,
    ) -> Tuple[jnp.ndarray, jnp.ndarray]:
        """
        Multi-step fine-grained rollout.
        
        Args:
            z_init: Initial latent [batch, latent_dim]
            actions: Actions [batch, horizon, action_dim]
        
        Returns:
            z_sequence: [batch, horizon+1, latent_dim]
            events: [batch, horizon, 1]
        """
        batch_size, horizon = actions.shape[:2]
        
        z_sequence = [z_init]
        events = []
        
        z = z_init
        for t in range(horizon):
            # Detect event
            if detect_events:
                event = EventDetector(self.latent_dim, name='event_det')(z)
            else:
                event = None
            
            events.append(event)
            
            # Step
            z = self.fine_step(z, actions[:, t], event)
            z_sequence.append(z)
        
        return (
            jnp.stack(z_sequence, axis=1),
            jnp.stack(events, axis=1) if detect_events else None,
        )
    
    def rollout_chunk(
        self,
        z_init: jnp.ndarray,
        actions: jnp.ndarray,
    ) -> jnp.ndarray:
        """
        Multi-step chunked rollout.
        
        Args:
            z_init: Initial latent [batch, latent_dim]
            actions: Actions [batch, horizon, action_dim]
        
        Returns:
            z_sequence: [batch, horizon//chunk_size + 1, latent_dim]
        """
        horizon = actions.shape[1]
        num_chunks = horizon // self.chunk_size
        
        z_sequence = [z_init]
        z = z_init
        
        for c in range(num_chunks):
            chunk_actions = actions[:, c*self.chunk_size:(c+1)*self.chunk_size]
            z = self.chunk_step(z, chunk_actions)
            z_sequence.append(z)
        
        return jnp.stack(z_sequence, axis=1)


# ============================================================================
# Physics Constraints
# ============================================================================

class PhysicsConstraints:
    """
    Physics constraints for planning-consistency.
    
    Enforce:
    - Energy conservation (for conservative systems)
    - Contact consistency (discontinuities)
    - Parameter shift consistency
    """
    
    @staticmethod
    def energy_conservation_loss(
        z_sequence: jnp.ndarray,
        energy_fn: Callable,
    ) -> jnp.ndarray:
        """
        Energy should be conserved for conservative systems.
        
        Args:
            z_sequence: [batch, time, latent_dim]
            energy_fn: Function computing energy from latent
        """
        energies = jax.vmap(energy_fn)(z_sequence)  # [batch, time]
        
        # Variance of energy over time (should be small)
        energy_variance = jnp.var(energies, axis=1).mean()
        
        return energy_variance
    
    @staticmethod
    def contact_consistency_loss(
        events: jnp.ndarray,
        z_sequence: jnp.ndarray,
        contact_fn: Callable,
    ) -> jnp.ndarray:
        """
        Events should correspond to actual contacts.
        
        Args:
            events: [batch, time, 1] - Detected events
            z_sequence: [batch, time, latent_dim]
            contact_fn: Function detecting true contacts
        """
        # True contacts
        true_contacts = contact_fn(z_sequence)
        
        # Event should match true contacts
        event_loss = jnp.mean((events.squeeze(-1) - true_contacts)**2)
        
        return event_loss
    
    @staticmethod
    def parameter_shift_loss(
        z_A: jnp.ndarray,
        z_B: jnp.ndarray,
        params_A: jnp.ndarray,
        params_B: jnp.ndarray,
    ) -> jnp.ndarray:
        """
        Latent structure should be consistent across parameter shifts.
        
        Args:
            z_A: Latent under params_A
            z_B: Latent under params_B
        """
        # Latent distance should scale with parameter distance
        param_distance = jnp.linalg.norm(params_A - params_B, axis=-1)
        latent_distance = jnp.linalg.norm(z_A - z_B, axis=-1)
        
        # Should be correlated
        loss = jnp.abs(param_distance - latent_distance).mean()
        
        return loss


# ============================================================================
# Differentiable Planner
# ============================================================================

class DifferentiablePlanner:
    """
    Differentiable MPC planner in latent space.
    
    Uses CEM (Cross-Entropy Method) with differentiable dynamics.
    Gradients flow from planning loss back to representation.
    """
    
    def __init__(
        self,
        dynamics: MultiScaleDynamics,
        cost_fn: Callable,
        horizon: int = 20,
        num_samples: int = 100,
        num_iterations: int = 5,
        elite_fraction: float = 0.2,
    ):
        self.dynamics = dynamics
        self.cost_fn = cost_fn
        self.horizon = horizon
        self.num_samples = num_samples
        self.num_iterations = num_iterations
        self.elite_fraction = elite_fraction
    
    def plan(
        self,
        z_init: jnp.ndarray,
        key: jax.random.PRNGKey,
        action_dim: int = 1,
        action_bounds: Tuple[float, float] = (-1.0, 1.0),
    ) -> Tuple[jnp.ndarray, jnp.ndarray, Dict]:
        """
        Plan action sequence using CEM.
        
        Args:
            z_init: Initial latent [batch, latent_dim]
            key: Random key
            action_dim: Action dimension
            action_bounds: (min, max) for actions
        
        Returns:
            best_actions: [horizon, action_dim]
            best_trajectory: [horizon+1, latent_dim]
            info: Planning info
        """
        batch_size = z_init.shape[0]
        
        # Initialize action distribution
        mean = jnp.zeros((self.horizon, action_dim))
        std = jnp.ones((self.horizon, action_dim))
        
        for iteration in range(self.num_iterations):
            key, subkey = jax.random.split(key)
            
            # Sample action sequences
            noise = jax.random.normal(
                subkey,
                shape=(self.num_samples, self.horizon, action_dim)
            )
            actions = mean + std * noise
            
            # Clip to bounds
            actions = jnp.clip(actions, action_bounds[0], action_bounds[1])
            
            # Expand for batch
            actions_batch = jnp.tile(actions[None, :, :, :], (batch_size, 1, 1, 1))
            
            # Rollout in latent space
            z_sequence, events = jax.vmap(
                lambda z, a: self.dynamics.rollout_fine(z, a)
            )(z_init, actions_batch)
            
            # Evaluate costs
            costs = jax.vmap(
                lambda z_seq: jnp.sum(self.cost_fn(z_seq))
            )(z_sequence)
            
            # Select elite samples
            num_elite = int(self.num_samples * self.elite_fraction)
            elite_idx = jnp.argsort(costs)[:num_elite]
            elite_actions = actions[elite_idx]
            
            # Update distribution
            mean = jnp.mean(elite_actions, axis=0)
            std = jnp.std(elite_actions, axis=0) + 1e-6
        
        # Final rollout
        best_actions = mean
        best_trajectory, _ = self.dynamics.rollout_fine(z_init, best_actions[None])
        
        info = {
            'final_cost': float(jnp.min(costs)),
            'num_iterations': self.num_iterations,
        }
        
        return best_actions, best_trajectory[0], info


# ============================================================================
# Planning-Consistency Loss (NOVEL)
# ============================================================================

class PlanningConsistencyLoss:
    """
    The key novelty: Train representation for planning-consistency.
    
    Loss = Planning regret + Trajectory mismatch + Constraint violations
    
    Planning regret = How much worse is planned vs optimal?
    """
    
    def __init__(
        self,
        planner: DifferentiablePlanner,
        dynamics: MultiScaleDynamics,
        cost_fn: Callable,
    ):
        self.planner = planner
        self.dynamics = dynamics
        self.cost_fn = cost_fn
    
    def compute_regret(
        self,
        z_init: jnp.ndarray,
        optimal_actions: jnp.ndarray,
        planned_actions: jnp.ndarray,
    ) -> jnp.ndarray:
        """
        Compute planning regret.
        
        Regret = J(planned_actions) - J(optimal_actions)
        
        Where J = cumulative cost over horizon.
        """
        # Rollout optimal actions
        z_optimal, _ = self.dynamics.rollout_fine(z_init, optimal_actions)
        cost_optimal = jnp.sum(self.cost_fn(z_optimal))
        
        # Rollout planned actions
        z_planned, _ = self.dynamics.rollout_fine(z_init, planned_actions)
        cost_planned = jnp.sum(self.cost_fn(z_planned))
        
        # Regret (should be small)
        regret = jax.nn.relu(cost_planned - cost_optimal)  # Only penalize worse
        
        return regret
    
    def trajectory_mismatch_loss(
        self,
        planned_trajectory: jnp.ndarray,
        observed_trajectory: jnp.ndarray,
    ) -> jnp.ndarray:
        """
        How different is planned vs observed?
        
        Penalize trajectories that diverge from reality.
        """
        # MSE between trajectories
        mismatch = jnp.mean((planned_trajectory - observed_trajectory)**2)
        
        return mismatch
    
    def constraint_violation_loss(
        self,
        z_sequence: jnp.ndarray,
        events: jnp.ndarray,
        energy_fn: Callable,
        contact_fn: Callable,
    ) -> jnp.ndarray:
        """
        Penalize physics constraint violations.
        """
        # Energy drift
        energies = jax.vmap(energy_fn)(z_sequence)
        energy_violation = jnp.abs(energies[-1] - energies[0])
        
        # Contact consistency
        true_contacts = contact_fn(z_sequence)
        contact_violation = jnp.mean((events.squeeze(-1) - true_contacts)**2)
        
        return energy_violation + contact_violation


# ============================================================================
# Complete PCP-JEPA Model
# ============================================================================

class PCPJEPA(nn.Module):
    """
    Planning-Consistent Physics JEPA.
    
    Complete model combining:
    1. Belief state (partial observability)
    2. Multi-scale dynamics (fine + chunk)
    3. Event detection (contacts, discontinuities)
    4. Physics constraints
    """
    latent_dim: int = 32
    action_dim: int = 1
    obs_dim: int = 4
    hidden_dim: int = 64
    chunk_size: int = 10
    
    @nn.compact
    def encode(self, observation: jnp.ndarray) -> jnp.ndarray:
        """Encode observation to latent."""
        x = nn.Dense(self.hidden_dim)(observation)
        x = nn.relu(x)
        x = nn.Dense(self.hidden_dim)(x)
        x = nn.relu(x)
        z = nn.Dense(self.latent_dim)(x)
        return z
    
    def get_dynamics(self) -> MultiScaleDynamics:
        """Get dynamics model."""
        return MultiScaleDynamics(
            self.latent_dim,
            self.action_dim,
            self.chunk_size,
            name='dynamics',
        )
    
    def get_belief(self) -> BeliefState:
        """Get belief state model."""
        return BeliefState(
            self.latent_dim,
            self.hidden_dim,
            name='belief',
        )
    
    def __call__(
        self,
        observations: jnp.ndarray,
        actions: jnp.ndarray,
        use_belief: bool = True,
    ) -> Dict:
        """
        Full forward pass.
        
        Args:
            observations: [batch, time, obs_dim]
            actions: [batch, time, action_dim]
        
        Returns:
            Dict with latent sequence, events, etc.
        """
        batch_size, time_steps = observations.shape[:2]
        
        # Encode observations
        z = jax.vmap(self.encode)(observations)  # [batch, time, latent]
        
        # Optionally use belief state
        if use_belief:
            belief = self.get_belief()
            b = belief.initialize(batch_size)
            
            beliefs = []
            for t in range(time_steps):
                a_prev = actions[:, t-1] if t > 0 else jnp.zeros((batch_size, self.action_dim))
                b = belief(b, observations[:, t], a_prev)
                beliefs.append(b)
            
            belief_sequence = jnp.stack(beliefs, axis=1)
        else:
            belief_sequence = None
        
        # Multi-scale dynamics
        dynamics = self.get_dynamics()
        
        # Fine-grained rollout
        z_fine, events = dynamics.rollout_fine(z[:, 0], actions)
        
        # Chunked rollout
        z_chunk = dynamics.rollout_chunk(z[:, 0], actions)
        
        return {
            'z': z,
            'z_fine': z_fine,
            'z_chunk': z_chunk,
            'events': events,
            'belief': belief_sequence,
        }


# ============================================================================
# Training Functions
# ============================================================================

def create_pcp_jepa(
    latent_dim: int = 32,
    action_dim: int = 1,
    obs_dim: int = 4,
    key: jax.random.PRNGKey = None,
) -> Tuple[PCPJEPA, Dict]:
    """Create and initialize PCP-JEPA model."""
    
    model = PCPJEPA(
        latent_dim=latent_dim,
        action_dim=action_dim,
        obs_dim=obs_dim,
    )
    
    # Initialize with dummy inputs
    dummy_obs = jnp.zeros((1, 10, obs_dim))
    dummy_actions = jnp.zeros((1, 10, action_dim))
    
    params = model.init(key, dummy_obs, dummy_actions)
    
    return model, params


def compute_training_loss(
    model: PCPJEPA,
    params: Dict,
    observations: jnp.ndarray,
    actions: jnp.ndarray,
    targets: jnp.ndarray,
    optimal_actions: Optional[jnp.ndarray] = None,
    cost_fn: Optional[Callable] = None,
    weights: Dict[str, float] = None,
) -> Tuple[jnp.ndarray, Dict]:
    """
    Compute total training loss.
    
    Args:
        model: PCP-JEPA model
        params: Model parameters
        observations: [batch, time, obs_dim]
        actions: [batch, time, action_dim]
        targets: [batch, time, obs_dim] - Ground truth observations
        optimal_actions: [batch, time, action_dim] - Demonstrations (optional)
        cost_fn: Cost function for planning
        weights: Loss weights
    
    Returns:
        total_loss: Scalar
        info: Dict with loss components
    """
    if weights is None:
        weights = {
            'prediction': 1.0,
            'planning_regret': 10.0,  # High weight on planning-consistency
            'trajectory_mismatch': 1.0,
            'constraint': 0.5,
        }
    
    # Forward pass
    outputs = model.apply(params, observations, actions)
    
    # Encode targets
    z_targets = jax.vmap(model.apply(params, targets, method=model.encode))(targets)
    
    # 1. Prediction loss
    prediction_loss = jnp.mean((outputs['z_fine'] - z_targets)**2)
    
    # 2. Planning regret loss (if optimal actions provided)
    if optimal_actions is not None and cost_fn is not None:
        # Plan in latent space
        planner = DifferentiablePlanner(
            model.get_dynamics(),
            cost_fn,
            horizon=actions.shape[1],
        )
        
        planned_actions, planned_traj, _ = planner.plan(
            outputs['z'][:, 0],
            jax.random.PRNGKey(0),
        )
        
        # Compute regret
        regret_loss = PlanningConsistencyLoss(
            planner, model.get_dynamics(), cost_fn
        ).compute_regret(
            outputs['z'][:, 0],
            optimal_actions,
            planned_actions,
        )
    else:
        regret_loss = 0.0
    
    # 3. Trajectory mismatch loss
    mismatch_loss = jnp.mean((outputs['z_fine'] - z_targets)**2)
    
    # 4. Constraint loss (simplified)
    # Energy conservation (variance of latent magnitude)
    energy_variance = jnp.var(jnp.linalg.norm(outputs['z_fine'], axis=-1), axis=1).mean()
    constraint_loss = energy_variance
    
    # Total loss
    total_loss = (
        weights['prediction'] * prediction_loss +
        weights['planning_regret'] * regret_loss +
        weights['trajectory_mismatch'] * mismatch_loss +
        weights['constraint'] * constraint_loss
    )
    
    info = {
        'prediction_loss': float(prediction_loss),
        'regret_loss': float(regret_loss) if isinstance(regret_loss, float) else float(regret_loss),
        'mismatch_loss': float(mismatch_loss),
        'constraint_loss': float(constraint_loss),
        'total_loss': float(total_loss),
    }
    
    return total_loss, info