"""
Tier 2 Environments: Hybrid/Discontinuous Physics

These environments have events (contacts, stick-slip, impacts) that
are the focus of our research.

Event logging is CRITICAL - we need to know when events happen
to test our hypotheses about event-linked failures.
"""

import jax
import jax.numpy as jnp
from typing import Dict, List, Tuple, Optional, NamedTuple
from dataclasses import dataclass
from enum import Enum


class EventType(Enum):
    """Types of physics events we track."""
    IMPACT = "impact"          # Collision/impact
    STICK = "stick"            # Friction transition: slip → stick
    SLIP = "slip"              # Friction transition: stick → slip
    SATURATION = "saturation"  # Actuator/joint limit reached
    CONTACT = "contact"        # New contact formed
    RELEASE = "release"        # Contact broken


@dataclass
class Event:
    """A single physics event."""
    type: EventType
    time: int              # Timestep when event occurred
    state_before: jnp.ndarray
    state_after: jnp.ndarray
    info: Dict             # Additional event info


class EventLog:
    """Log of events during a trajectory."""
    
    def __init__(self):
        self.events: List[Event] = []
    
    def add(self, event: Event):
        self.events.append(event)
    
    def get_events(self, event_type: EventType = None) -> List[Event]:
        """Get events, optionally filtered by type."""
        if event_type is None:
            return self.events
        return [e for e in self.events if e.type == event_type]
    
    def get_event_times(self, event_type: EventType = None) -> List[int]:
        """Get timesteps when events occurred."""
        events = self.get_events(event_type)
        return [e.time for e in events]
    
    def __len__(self):
        return len(self.events)


# ============================================================================
# Bouncing Ball: Impact Events
# ============================================================================

class BouncingBallParams(NamedTuple):
    """Parameters for bouncing ball."""
    mass: float = 1.0
    gravity: float = 9.81
    restitution: float = 0.8      # Coefficient of restitution
    radius: float = 0.1
    ground_height: float = 0.0


class BouncingBall:
    """
    Bouncing ball with impact events.
    
    State: [x, y, vx, vy] (position and velocity)
    
    Events:
    - IMPACT: Ball hits ground
    """
    
    def __init__(self, params: BouncingBallParams = None):
        self.params = params or BouncingBallParams()
    
    def step(
        self,
        state: jnp.ndarray,
        dt: float = 0.02,
    ) -> Tuple[jnp.ndarray, Optional[Event]]:
        """
        Step the simulation.
        
        Returns:
            next_state: New state
            event: Event if one occurred, else None
        """
        x, y, vx, vy = state
        p = self.params
        
        # Gravity
        vy_new = vy - p.gravity * dt
        y_new = y + vy_new * dt
        x_new = x + vx * dt
        
        # Check for impact (ball hits ground)
        ground_contact = y_new - p.radius < p.ground_height
        
        # Handle impact
        event = None
        if ground_contact:
            # Record event
            state_before = jnp.array([x, y, vx, vy])
            
            # Bounce
            y_new = p.ground_height + p.radius
            vy_new = -p.restitution * vy_new
            
            state_after = jnp.array([x_new, y_new, vx, vy_new])
            
            event = Event(
                type=EventType.IMPACT,
                time=0,  # Will be set by caller
                state_before=state_before,
                state_after=state_after,
                info={'restitution': p.restitution, 'impact_velocity': abs(vy)}
            )
        
        next_state = jnp.array([x_new, y_new, vx, vy_new])
        
        return next_state, event
    
    def simulate(
        self,
        initial_state: jnp.ndarray,
        num_steps: int,
        dt: float = 0.02,
    ) -> Tuple[jnp.ndarray, EventLog]:
        """
        Simulate trajectory and log events.
        
        Returns:
            trajectory: [num_steps+1, state_dim]
            event_log: Log of events
        """
        trajectory = [initial_state]
        event_log = EventLog()
        
        state = initial_state
        for t in range(num_steps):
            state, event = self.step(state, dt)
            
            if event is not None:
                event.time = t
                event_log.add(event)
            
            trajectory.append(state)
        
        return jnp.stack(trajectory), event_log


# ============================================================================
# Stick-Slip Block: Friction Events
# ============================================================================

class StickSlipParams(NamedTuple):
    """Parameters for stick-slip block."""
    mass: float = 1.0
    static_friction: float = 0.6    # μ_s
    kinetic_friction: float = 0.4   # μ_k
    spring_constant: float = 10.0   # Spring pulling block
    spring_velocity: float = 0.5    # Velocity of spring end


class StickSlipBlock:
    """
    Block on surface with stick-slip friction.
    
    State: [x, v, spring_end] (position, velocity, spring end position)
    
    Events:
    - STICK: Block transitions from slipping to stuck
    - SLIP: Block transitions from stuck to slipping
    """
    
    def __init__(self, params: StickSlipParams = None):
        self.params = params or StickSlipParams()
        self.is_stuck = False  # Track current state
    
    def step(
        self,
        state: jnp.ndarray,
        dt: float = 0.02,
    ) -> Tuple[jnp.ndarray, Optional[Event]]:
        """
        Step the simulation.
        
        Returns:
            next_state: New state
            event: Event if one occurred
        """
        x, v, spring_end = state
        p = self.params
        
        # Move spring end
        spring_end_new = spring_end + p.spring_velocity * dt
        
        # Spring force
        spring_extension = spring_end_new - x
        F_spring = p.spring_constant * spring_extension
        
        # Friction
        event = None
        
        if self.is_stuck:
            # Check if we break free
            # Maximum static friction
            F_static_max = p.static_friction * p.mass * 9.81
            
            if abs(F_spring) > F_static_max:
                # Break free - SLIP event
                self.is_stuck = False
                
                state_before = state
                state_after = jnp.array([x, F_spring / p.mass, spring_end_new])
                
                event = Event(
                    type=EventType.SLIP,
                    time=0,
                    state_before=state_before,
                    state_after=state_after,
                    info={'force': float(F_spring), 'threshold': float(F_static_max)}
                )
                
                v_new = 0.0  # Start with zero velocity
            else:
                # Stay stuck
                v_new = 0.0
                x_new = x
        else:
            # Slipping - apply kinetic friction
            F_friction = p.kinetic_friction * p.mass * 9.81
            if v > 0:
                F_friction = -F_friction
            elif v == 0:
                F_friction = -jnp.sign(F_spring) * F_friction
            
            # Acceleration
            a = (F_spring + F_friction) / p.mass
            
            # Update velocity
            v_new = v + a * dt
            x_new = x + v_new * dt
            
            # Check if we stick
            # (velocity crosses zero and force < static threshold)
            if v * v_new < 0:  # Velocity changed sign
                F_static_max = p.static_friction * p.mass * 9.81
                if abs(F_spring) < F_static_max:
                    # Stick!
                    self.is_stuck = True
                    v_new = 0.0
                    
                    state_before = jnp.array([x, v, spring_end])
                    state_after = jnp.array([x_new, v_new, spring_end_new])
                    
                    event = Event(
                        type=EventType.STICK,
                        time=0,
                        state_before=state_before,
                        state_after=state_after,
                        info={'position': float(x_new)}
                    )
        
        next_state = jnp.array([x_new, v_new, spring_end_new])
        
        return next_state, event
    
    def simulate(
        self,
        initial_state: jnp.ndarray,
        num_steps: int,
        dt: float = 0.02,
    ) -> Tuple[jnp.ndarray, EventLog]:
        """Simulate trajectory and log events."""
        trajectory = [initial_state]
        event_log = EventLog()
        
        state = initial_state
        for t in range(num_steps):
            state, event = self.step(state, dt)
            
            if event is not None:
                event.time = t
                event_log.add(event)
            
            trajectory.append(state)
        
        return jnp.stack(trajectory), event_log


# ============================================================================
# Event Metrics
# ============================================================================

class EventMetrics:
    """
    Metrics for evaluating event-related performance.
    
    These metrics are CRITICAL for testing our hypotheses.
    """
    
    @staticmethod
    def event_timing_error(
        predicted_events: List[int],
        true_events: List[int],
        tolerance: int = 2,
    ) -> Dict[str, float]:
        """
        Measure how accurately events are predicted.
        
        Args:
            predicted_events: Predicted event times
            true_events: True event times
            tolerance: Allowed timing error (timesteps)
        
        Returns:
            Dict with precision, recall, timing_error
        """
        if len(true_events) == 0:
            return {
                'precision': 1.0 if len(predicted_events) == 0 else 0.0,
                'recall': 1.0,
                'timing_error': 0.0,
            }
        
        if len(predicted_events) == 0:
            return {
                'precision': 0.0,
                'recall': 0.0,
                'timing_error': float('inf'),
            }
        
        # Match predicted to true events
        matched_true = set()
        timing_errors = []
        
        for pred_t in predicted_events:
            # Find closest true event
            distances = [abs(pred_t - true_t) for true_t in true_events]
            min_idx = jnp.argmin(jnp.array(distances))
            min_dist = distances[min_idx]
            
            if min_dist <= tolerance:
                matched_true.add(min_idx)
                timing_errors.append(min_dist)
        
        precision = len(matched_true) / len(predicted_events)
        recall = len(matched_true) / len(true_events)
        timing_error = jnp.mean(jnp.array(timing_errors)) if timing_errors else 0.0
        
        return {
            'precision': float(precision),
            'recall': float(recall),
            'timing_error': float(timing_error),
        }
    
    @staticmethod
    def event_linked_failure_fraction(
        divergence_times: List[int],
        event_times: List[int],
        window: int = 5,
    ) -> float:
        """
        Fraction of failures preceded by an event.
        
        This is the KEY metric for testing the "event dominance" hypothesis.
        
        Args:
            divergence_times: When planned trajectory diverged from true
            event_times: When events occurred
            window: How close event must be to be "linked"
        
        Returns:
            Fraction of divergences linked to events
        """
        if len(divergence_times) == 0:
            return 0.0
        
        linked_count = 0
        for div_t in divergence_times:
            # Check if any event happened within window before divergence
            for event_t in event_times:
                if 0 <= div_t - event_t <= window:
                    linked_count += 1
                    break
        
        return linked_count / len(divergence_times)
    
    @staticmethod
    def event_conditioned_calibration(
        predictions: jnp.ndarray,
        targets: jnp.ndarray,
        uncertainties: jnp.ndarray,
        event_mask: jnp.ndarray,
    ) -> Dict[str, float]:
        """
        Calibration separately for event and non-event segments.
        
        Args:
            predictions: [time, dim]
            targets: [time, dim]
            uncertainties: [time, 1] (predicted uncertainty)
            event_mask: [time] (1 if event nearby, 0 otherwise)
        
        Returns:
            Calibration metrics for event vs non-event
        """
        errors = jnp.mean((predictions - targets)**2, axis=-1)
        
        # Event segments
        event_errors = errors[event_mask == 1]
        event_uncerts = uncertainties[event_mask == 1]
        
        # Non-event segments
        nonevent_errors = errors[event_mask == 0]
        nonevent_uncerts = uncertainties[event_mask == 0]
        
        def calibration_ratio(errors, uncertainties):
            """Ratio of observed error to predicted uncertainty."""
            if len(errors) == 0:
                return 0.0
            return float(jnp.mean(errors) / (jnp.mean(uncertainties) + 1e-8))
        
        return {
            'event_calibration': calibration_ratio(event_errors, event_uncerts),
            'nonevent_calibration': calibration_ratio(nonevent_errors, nonevent_uncerts),
            'event_error_mean': float(jnp.mean(event_errors)) if len(event_errors) > 0 else 0.0,
            'nonevent_error_mean': float(jnp.mean(nonevent_errors)) if len(nonevent_errors) > 0 else 0.0,
        }


# ============================================================================
# Horizon Scaling Evaluation
# ============================================================================

class HorizonScalingEval:
    """
    Evaluate planning performance across horizons.
    
    This is the PRIMARY evaluation for our research.
    """
    
    def __init__(self, horizons: List[int] = None):
        self.horizons = horizons or [10, 30, 100, 300]
    
    def evaluate(
        self,
        model,
        params,
        env,
        initial_states: jnp.ndarray,
        planner_fn,
        cost_fn,
    ) -> Dict[str, List[float]]:
        """
        Evaluate model across horizons.
        
        Args:
            model: World model
            params: Model parameters
            env: Environment
            initial_states: [batch, state_dim]
            planner_fn: Function that plans given model
            cost_fn: Cost function
        
        Returns:
            Dict with success rates, returns, etc. for each horizon
        """
        results = {
            'horizon': self.horizons,
            'success_rate': [],
            'mean_return': [],
            'event_timing_error': [],
            'prediction_mse': [],
        }
        
        for H in self.horizons:
            successes = []
            returns = []
            timing_errors = []
            mses = []
            
            for initial_state in initial_states:
                # Plan
                planned_actions, planned_traj = planner_fn(
                    model, params, initial_state, H
                )
                
                # Execute in environment
                true_traj, event_log = env.simulate(initial_state, H)
                
                # Compute metrics
                return_val = -cost_fn(true_traj)
                success = return_val > -100  # Task-specific threshold
                
                # Event timing
                predicted_events = []  # From model's event detector
                true_events = event_log.get_event_times()
                timing = EventMetrics.event_timing_error(
                    predicted_events, true_events
                )
                
                # Prediction MSE
                mse = float(jnp.mean((planned_traj - true_traj)**2))
                
                successes.append(success)
                returns.append(return_val)
                timing_errors.append(timing['timing_error'])
                mses.append(mse)
            
            results['success_rate'].append(jnp.mean(jnp.array(successes)))
            results['mean_return'].append(jnp.mean(jnp.array(returns)))
            results['event_timing_error'].append(jnp.mean(jnp.array(timing_errors)))
            results['prediction_mse'].append(jnp.mean(jnp.array(mses)))
        
        return results


# ============================================================================
# Test / Demo
# ============================================================================

def test_bouncing_ball():
    """Test bouncing ball environment."""
    print("\n" + "=" * 60)
    print("TEST: Bouncing Ball Environment")
    print("=" * 60)
    
    env = BouncingBall(BouncingBallParams(restitution=0.8))
    
    # Initial state: ball above ground
    initial_state = jnp.array([0.0, 1.0, 0.0, 0.0])  # x, y, vx, vy
    
    # Simulate
    trajectory, event_log = env.simulate(initial_state, num_steps=100)
    
    print(f"Trajectory shape: {trajectory.shape}")
    print(f"Number of impacts: {len(event_log.get_events(EventType.IMPACT))}")
    print(f"Impact times: {event_log.get_event_times(EventType.IMPACT)}")
    
    return trajectory, event_log


def test_stick_slip():
    """Test stick-slip environment."""
    print("\n" + "=" * 60)
    print("TEST: Stick-Slip Block Environment")
    print("=" * 60)
    
    env = StickSlipBlock(StickSlipParams(
        static_friction=0.6,
        kinetic_friction=0.4,
        spring_constant=10.0,
        spring_velocity=0.5,
    ))
    
    # Initial state: block at origin, spring at origin
    initial_state = jnp.array([0.0, 0.0, 0.0])
    
    # Simulate
    trajectory, event_log = env.simulate(initial_state, num_steps=200)
    
    print(f"Trajectory shape: {trajectory.shape}")
    print(f"Number of STICK events: {len(event_log.get_events(EventType.STICK))}")
    print(f"Number of SLIP events: {len(event_log.get_events(EventType.SLIP))}")
    
    return trajectory, event_log


if __name__ == "__main__":
    test_bouncing_ball()
    test_stick_slip()