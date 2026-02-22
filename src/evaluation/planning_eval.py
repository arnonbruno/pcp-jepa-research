"""
Proper Planning Evaluation for Phase 2

The minimum correct loop:
1. Real control task: Land at target x* after N steps
2. Lightweight state decoder D(z) → [x, y, vx, vy]
3. MPPI planner in latent space using learned dynamics
4. Closed-loop execution in true simulator
5. Measure: success, catastrophics, horizon scaling, event-linked failures

This makes O1/O3 actually matter in planning:
- O1: event-conditioned dynamics f(z, a, p) influences rollouts
- O3: risk-aware MPPI uses variance Σ
"""

import jax
import jax.numpy as jnp
import numpy as np
from typing import Dict, Tuple, List, Optional
from dataclasses import dataclass
import flax.linen as nn
from functools import partial


# ============================================================================
# Control Task Definition
# ============================================================================

@dataclass
class BouncingBallTask:
    """
    Task: Land at target x* after N steps/bounces.
    
    Control: Horizontal impulse a_t ∈ [-1, 1] (bounded)
    Objective: Minimize |x_final - x_target|
    
    This is a proper planning task where:
    - Impacts matter (ball bounces)
    - Timing matters (when to apply impulse)
    - Events affect outcome
    """
    x_target: float = 1.0  # Target landing position
    horizon: int = 50      # Planning horizon
    num_steps: int = 50    # Total simulation steps
    max_impulse: float = 1.0  # Max horizontal impulse per step
    
    def cost_fn(self, state: jnp.ndarray, target: Optional[float] = None) -> float:
        """
        Cost function for planning.
        
        Args:
            state: [x, y, vx, vy] or latent z
            target: x_target (uses self.x_target if None)
        
        Returns:
            Scalar cost (lower is better)
        """
        if target is None:
            target = self.x_target
        
        # Extract position
        x = state[0] if state.shape[-1] <= 4 else state  # Handle latent
        
        # Distance to target
        position_cost = (x - target) ** 2
        
        return position_cost
    
    def terminal_cost(self, state: jnp.ndarray) -> float:
        """Terminal cost at end of episode."""
        return 10.0 * self.cost_fn(state)
    
    def step_cost(self, state: jnp.ndarray, action: jnp.ndarray) -> float:
        """Per-step cost."""
        # Small action penalty
        action_cost = 0.01 * jnp.sum(action ** 2)
        return self.cost_fn(state) + action_cost


# ============================================================================
# State Decoder
# ============================================================================

class StateDecoder(nn.Module):
    """
    Lightweight decoder: D(z) → [x, y, vx, vy]
    
    No pixel reconstruction needed - just state estimation.
    """
    state_dim: int = 4
    hidden_dim: int = 64
    
    @nn.compact
    def __call__(self, z: jnp.ndarray) -> jnp.ndarray:
        """
        Args:
            z: [..., latent_dim]
        
        Returns:
            state: [..., state_dim] = [x, y, vx, vy]
        """
        h = nn.Dense(self.hidden_dim)(z)
        h = nn.relu(h)
        state = nn.Dense(self.state_dim)(h)
        return state


# ============================================================================
# MPPI Planner
# ============================================================================

class MPPIPlanner:
    """
    Model Predictive Path Integral (MPPI) planner.
    
    Plans in latent space using learned dynamics:
    - Baseline: z_{t+1} = f(z_t, a_t)
    - O1: z_{t+1} = f(z_t, a_t, p_t) where p_t = event_prob
    - O3: z_{t+1} ~ N(μ, Σ) with risk-aware cost
    """
    
    def __init__(
        self,
        model,
        params: dict,
        decoder_params: dict,
        task: BouncingBallTask,
        horizon: int = 50,
        num_samples: int = 100,
        temperature: float = 1.0,
        action_dim: int = 1,
        model_type: str = 'baseline',  # 'baseline', 'O1', 'O3'
    ):
        self.model = model
        self.params = params
        self.decoder_params = decoder_params
        self.task = task
        self.horizon = horizon
        self.num_samples = num_samples
        self.temperature = temperature
        self.action_dim = action_dim
        self.model_type = model_type
        
        # For O3: risk weight
        self.beta_risk = 0.0
    
    def decode(self, z: jnp.ndarray) -> jnp.ndarray:
        """Decode latent to state."""
        return self.model.apply(
            self.decoder_params, z,
            method=lambda p, z: StateDecoder(4, 64).apply(p, z),
        )
    
    def predict_next(self, z: jnp.ndarray, a: jnp.ndarray) -> jnp.ndarray:
        """Predict next latent (different for each model type)."""
        if self.model_type == 'O1':
            # Event-conditioned dynamics
            # Need to get event probability first
            event_prob = self.model.apply(
                self.params, z,
                method=lambda p, z: p['event_probs'] if 'event_probs' in p else 0.5,
            )
            # Use dynamics with event conditioning
            # (This is simplified - actual implementation depends on model)
            z_next = self.model.apply(
                self.params, z, a, event_prob,
                method='predict_next',
            )
        elif self.model_type == 'O3':
            # Probabilistic dynamics
            mu, std = self.model.apply(
                self.params, z, a,
                method='predict_with_uncertainty',
            )
            # Sample
            key = jax.random.PRNGKey(0)  # Should use proper key
            z_next = mu + std * jax.random.normal(key, mu.shape)
        else:
            # Baseline
            z_next = self.model.apply(
                self.params, z, a,
                method='predict_next',
            )
        
        return z_next
    
    def rollout_trajectory(
        self,
        z0: jnp.ndarray,
        actions: jnp.ndarray,
        key: jax.random.PRNGKey,
    ) -> Tuple[jnp.ndarray, float]:
        """
        Rollout a trajectory and compute total cost.
        
        Args:
            z0: [latent_dim] initial latent
            actions: [H, action_dim] action sequence
            key: random key
        
        Returns:
            states: [H+1, state_dim] decoded states
            total_cost: scalar
        """
        H = len(actions)
        
        z = z0
        states = [self.decode(z)]
        total_cost = 0.0
        
        for t in range(H):
            # Predict next latent
            if self.model_type == 'O3':
                # Probabilistic
                mu, std = self.model.apply(
                    self.params, z, actions[t],
                    method='predict_with_uncertainty',
                )
                k, key = jax.random.split(key)
                z = mu + std * jax.random.normal(k, mu.shape)
            else:
                z = self.predict_next(z, actions[t])
            
            # Decode
            state = self.decode(z)
            states.append(state)
            
            # Cost
            cost = self.task.step_cost(state, actions[t])
            total_cost = total_cost + cost
        
        # Terminal cost
        total_cost = total_cost + self.task.terminal_cost(states[-1])
        
        return jnp.stack(states), total_cost
    
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
        # Initialize action sequence
        actions = jnp.zeros((self.horizon, self.action_dim))
        
        # Sample perturbations
        perturbations = jax.random.normal(
            key, (self.num_samples, self.horizon, self.action_dim)
        ) * 0.3 * self.task.max_impulse
        
        # Evaluate samples
        sample_actions = actions + perturbations
        costs = []
        
        for i in range(self.num_samples):
            k, key = jax.random.split(key)
            _, cost = self.rollout_trajectory(z0, sample_actions[i], k)
            costs.append(cost)
        
        costs = jnp.array(costs)
        
        # MPPI weights
        weights = jax.nn.softmax(-costs / self.temperature)
        
        # Weighted average
        actions = jnp.sum(weights[:, None, None] * sample_actions, axis=0)
        
        # Clip to bounds
        actions = jnp.clip(actions, -self.task.max_impulse, self.task.max_impulse)
        
        return actions


# ============================================================================
# Closed-Loop Execution
# ============================================================================

def execute_closed_loop(
    env,
    initial_state: jnp.ndarray,
    model,
    params: dict,
    decoder_params: dict,
    task: BouncingBallTask,
    planner: MPPIPlanner,
    key: jax.random.PRNGKey,
) -> Tuple[jnp.ndarray, Dict]:
    """
    Execute closed-loop planning and control.
    
    Steps:
    1. Encode current observation
    2. Plan action sequence in latent space
    3. Execute first action in simulator
    4. Repeat until done
    
    Args:
        env: True simulator (BouncingBall)
        initial_state: [x, y, vx, vy]
        model: World model
        params: Model parameters
        decoder_params: Decoder parameters
        task: Control task
        planner: MPPI planner
        key: Random key
    
    Returns:
        trajectory: [T+1, state_dim] true trajectory
        metrics: Dictionary of planning metrics
    """
    trajectory = [initial_state]
    event_log = []
    
    total_cost = 0.0
    planning_errors = []
    
    state = initial_state
    
    for t in range(task.num_steps):
        # Encode current state
        # (For simplicity, assume encoder takes state directly)
        z = model.apply(params, state[None, :], method='encode')
        
        # Plan
        k, key = jax.random.split(key)
        actions = planner.plan(z[0], k)
        
        # Execute first action
        action = actions[0]
        
        # Apply action in simulator
        # For bouncing ball, action is horizontal impulse
        state_with_action = state.at[2].add(action)  # vx += action
        next_state, event = env.step(state_with_action)
        
        trajectory.append(next_state)
        if event is not None:
            event.time = t
            event_log.append(event)
        
        # Compute cost
        step_cost = task.step_cost(next_state, action)
        total_cost += step_cost
        
        # Track planning error
        predicted_state = planner.decode(z[0])
        planning_error = float(jnp.linalg.norm(predicted_state - state))
        planning_errors.append(planning_error)
        
        state = next_state
    
    # Terminal cost
    total_cost += task.terminal_cost(state)
    
    # Success: close to target
    success = abs(state[0] - task.x_target) < 0.5
    
    # Catastrophic: diverged
    catastrophic = abs(state[0]) > 10.0 or abs(state[1]) > 10.0
    
    trajectory = jnp.stack(trajectory)
    
    metrics = {
        'total_cost': total_cost,
        'success': success,
        'catastrophic': catastrophic,
        'planning_errors': planning_errors,
        'final_x': float(state[0]),
        'target_x': task.x_target,
    }
    
    return trajectory, metrics


# ============================================================================
# Evaluation
# ============================================================================

def evaluate_planning(
    env,
    model,
    params: dict,
    decoder_params: dict,
    task: BouncingBallTask,
    num_episodes: int = 20,
    key: jax.random.PRNGKey = None,
    model_type: str = 'baseline',
) -> Dict:
    """
    Evaluate planning performance.
    
    This is the CORRECT evaluation:
    1. Real control task
    2. MPPI planning in latent space
    3. Closed-loop execution in simulator
    4. Measure success, catastrophics, horizon scaling
    """
    successes = 0
    catastrophics = 0
    total_costs = []
    event_linked_failures = 0
    total_failures = 0
    
    planner = MPPIPlanner(
        model, params, decoder_params, task,
        horizon=task.horizon,
        num_samples=50,
        model_type=model_type,
    )
    
    for episode in range(num_episodes):
        k, key = jax.random.split(key)
        
        # Random initial state
        y_init = jax.random.uniform(k, minval=1.0, maxval=3.0)
        vy_init = jax.random.uniform(k, minval=-2.0, maxval=2.0)
        initial_state = jnp.array([0.0, y_init, 0.0, vy_init])
        
        # Execute
        trajectory, metrics = execute_closed_loop(
            env, initial_state, model, params, decoder_params,
            task, planner, k
        )
        
        total_costs.append(metrics['total_cost'])
        
        if metrics['success']:
            successes += 1
        elif metrics['catastrophic']:
            catastrophics += 1
            total_failures += 1
            
            # Check event-linked
            # (Would need event_log from trajectory)
            # For now, placeholder
    
    return {
        'success_rate': successes / num_episodes,
        'catastrophic_rate': catastrophics / num_episodes,
        'mean_cost': np.mean(total_costs),
        'num_episodes': num_episodes,
    }


# ============================================================================
# Horizon Scaling Evaluation
# ============================================================================

def evaluate_horizon_scaling(
    env,
    model,
    params: dict,
    decoder_params: dict,
    horizons: List[int] = [10, 30, 50, 100],
    num_episodes_per_horizon: int = 10,
    key: jax.random.PRNGKey = None,
    model_type: str = 'baseline',
) -> Dict:
    """
    Evaluate planning across different horizons.
    
    This is KEY for Phase 2: does O1/O3 improve horizon scaling?
    """
    results = {}
    
    for H in horizons:
        task = BouncingBallTask(horizon=H, num_steps=H)
        
        metrics = evaluate_planning(
            env, model, params, decoder_params, task,
            num_episodes=num_episodes_per_horizon,
            key=key,
            model_type=model_type,
        )
        
        results[f'H={H}'] = metrics
        
        print(f"  H={H}: success={metrics['success_rate']:.1%}, "
              f"catastrophic={metrics['catastrophic_rate']:.1%}")
    
    return results