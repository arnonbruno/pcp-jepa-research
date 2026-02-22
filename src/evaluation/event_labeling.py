"""
Event Detection and Labeling Infrastructure

Global setup for Phase 2 objectives O1-O3.

Event types:
- Impact: contact impulse > threshold
- Stick-slip: friction regime change
- Saturation: action clipping

Event windows: ŷ_t = 1 if any event in [t-Δ, t+Δ]
"""

import jax
import jax.numpy as jnp
import numpy as np
from typing import Dict, List, Tuple, Optional, Callable
from dataclasses import dataclass
from enum import Enum, auto


class EventType(Enum):
    IMPACT = auto()
    STICK_SLIP = auto()
    SATURATION = auto()
    CONSTRAINT_SWITCH = auto()


@dataclass
class EventLabels:
    """Event labels for a trajectory."""
    y: jnp.ndarray  # [T] binary event occurrence
    y_window: jnp.ndarray  # [T] binary event window (Δ=5)
    types: Optional[jnp.ndarray] = None  # [T] event type indices
    severity: Optional[jnp.ndarray] = None  # [T] event severity (impulse magnitude, etc.)
    
    def get_event_times(self) -> List[int]:
        """Get timesteps where events occur."""
        return list(np.where(np.array(self.y) > 0.5)[0])
    
    def get_event_window_times(self) -> List[int]:
        """Get timesteps in event windows."""
        return list(np.where(np.array(self.y_window) > 0.5)[0])


class EventDetector:
    """
    Detect events from trajectory data.
    
    Different event types require different detection methods.
    """
    
    def __init__(self, delta: int = 5):
        self.delta = delta
    
    def detect_impacts(
        self,
        trajectory: jnp.ndarray,
        velocity_idx: int = 1,
        position_idx: int = 0,
        threshold: float = 0.1,
    ) -> jnp.ndarray:
        """
        Detect impact events from velocity discontinuities.
        
        Args:
            trajectory: [T, state_dim] state trajectory
            velocity_idx: index of velocity component
            position_idx: index of position (for ground contact)
            threshold: velocity change threshold
        
        Returns:
            y: [T] binary event labels
        """
        # Velocity changes
        velocity = trajectory[:, velocity_idx]
        vel_change = jnp.abs(jnp.diff(velocity))
        
        # Pad to match trajectory length
        vel_change = jnp.concatenate([jnp.zeros(1), vel_change])
        
        # Ground contact (position near zero)
        position = trajectory[:, position_idx]
        near_ground = position < threshold
        
        # Impact = large velocity change near ground
        impacts = (vel_change > threshold) & near_ground
        
        return impacts.astype(jnp.float32)
    
    def detect_stick_slip(
        self,
        trajectory: jnp.ndarray,
        velocity_idx: int = 0,
        threshold: float = 0.01,
    ) -> jnp.ndarray:
        """
        Detect stick-slip events from velocity sign changes.
        
        Args:
            trajectory: [T, state_dim] state trajectory
            velocity_idx: index of velocity component
            threshold: velocity threshold for "stuck"
        
        Returns:
            y: [T] binary event labels
        """
        velocity = trajectory[:, velocity_idx]
        
        # Sign changes
        sign = jnp.sign(velocity)
        sign_changes = jnp.abs(jnp.diff(sign))
        sign_changes = jnp.concatenate([jnp.zeros(1), sign_changes])
        
        # Crossing zero threshold
        near_zero = jnp.abs(velocity) < threshold
        
        # Stick-slip = sign change near zero velocity
        events = (sign_changes > 0) & near_zero
        
        return events.astype(jnp.float32)
    
    def detect_from_simulator(
        self,
        event_log,  # EventLog from environment
        num_steps: int,
    ) -> jnp.ndarray:
        """
        Create event labels from simulator event log.
        
        Args:
            event_log: EventLog object from environment
            num_steps: total trajectory length
        
        Returns:
            y: [T] binary event labels
        """
        y = jnp.zeros(num_steps)
        
        for event in event_log.events:
            y = y.at[event.timestep].set(1.0)
        
        return y
    
    def create_window_labels(
        self,
        y: jnp.ndarray,
        delta: Optional[int] = None,
    ) -> jnp.ndarray:
        """
        Create event window labels.
        
        ŷ_t = 1 if any y_{t-Δ:t+Δ} = 1
        
        Args:
            y: [T] binary event labels
            delta: window radius (default: self.delta)
        
        Returns:
            y_window: [T] binary window labels
        """
        if delta is None:
            delta = self.delta
        
        T = len(y)
        y_window = jnp.zeros_like(y)
        
        # Convolve with window kernel
        kernel = jnp.ones(2 * delta + 1)
        
        # Pad y for convolution
        y_padded = jnp.concatenate([
            jnp.zeros(delta),
            y,
            jnp.zeros(delta),
        ])
        
        # Sliding window
        for t in range(T):
            window = y_padded[t:t + 2 * delta + 1]
            if jnp.any(window > 0.5):
                y_window = y_window.at[t].set(1.0)
        
        return y_window
    
    def label_trajectory(
        self,
        trajectory: jnp.ndarray,
        event_log=None,
        event_types: Optional[List[EventType]] = None,
    ) -> EventLabels:
        """
        Create full event labels for a trajectory.
        
        Args:
            trajectory: [T, state_dim] state trajectory
            event_log: EventLog from simulator (if available)
            event_types: which event types to detect
        
        Returns:
            EventLabels object
        """
        if event_types is None:
            event_types = [EventType.IMPACT]
        
        T = len(trajectory)
        
        # Collect events
        y = jnp.zeros(T)
        
        for event_type in event_types:
            if event_type == EventType.IMPACT:
                y_impact = self.detect_impacts(trajectory)
                y = jnp.maximum(y, y_impact)
            elif event_type == EventType.STICK_SLIP:
                y_slip = self.detect_stick_slip(trajectory)
                y = jnp.maximum(y, y_slip)
        
        # Use simulator events if available
        if event_log is not None:
            y_sim = self.detect_from_simulator(event_log, T)
            y = jnp.maximum(y, y_sim)
        
        # Create window labels
        y_window = self.create_window_labels(y)
        
        return EventLabels(y=y, y_window=y_window)


def compute_event_metrics(
    predicted_events: jnp.ndarray,
    true_events: jnp.ndarray,
    delta: int = 5,
) -> Dict[str, float]:
    """
    Compute event detection metrics.
    
    Args:
        predicted_events: [T] predicted event probabilities
        true_events: [T] true binary event labels
        delta: tolerance for timing
    
    Returns:
        Dictionary of metrics
    """
    # Threshold predictions
    pred_binary = (predicted_events > 0.5).astype(jnp.float32)
    true_binary = (true_events > 0.5).astype(jnp.float32)
    
    # Event timing error
    pred_times = jnp.where(pred_binary)[0]
    true_times = jnp.where(true_binary)[0]
    
    if len(true_times) == 0:
        timing_error = 0.0 if len(pred_times) == 0 else float('inf')
    elif len(pred_times) == 0:
        timing_error = float('inf')
    else:
        # Match each true event to nearest predicted
        errors = []
        for t_true in true_times:
            distances = jnp.abs(pred_times - t_true)
            min_dist = jnp.min(distances)
            errors.append(float(min_dist))
        timing_error = jnp.mean(jnp.array(errors))
    
    # Event recall @ Δ
    true_times_set = set(int(t) for t in true_times)
    detected = 0
    for t_pred in pred_times:
        for t_true in true_times:
            if abs(int(t_pred) - int(t_true)) <= delta:
                detected += 1
                break
    
    recall = detected / len(true_times) if len(true_times) > 0 else 1.0
    
    # Precision
    precision = detected / len(pred_times) if len(pred_times) > 0 else 1.0
    
    return {
        'timing_error': timing_error,
        'recall_at_delta': recall,
        'precision': precision,
        'f1': 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0,
        'num_predicted': len(pred_times),
        'num_true': len(true_times),
    }


def event_linked_failure_fraction(
    failure_times: List[int],
    event_times: List[int],
    window: int = 5,
) -> float:
    """
    Compute fraction of failures linked to events.
    
    This is the KEY metric from Phase 1 (97.1% result).
    """
    if len(failure_times) == 0:
        return 0.0
    
    linked = 0
    for fail_t in failure_times:
        for event_t in event_times:
            if 0 <= fail_t - event_t <= window:
                linked += 1
                break
    
    return linked / len(failure_times)


# ============================================================================
# Batch Processing
# ============================================================================

def label_batch(
    trajectories: List[jnp.ndarray],
    event_logs: Optional[List] = None,
    event_types: Optional[List[EventType]] = None,
) -> List[EventLabels]:
    """
    Label a batch of trajectories.
    
    Args:
        trajectories: List of [T, state_dim] trajectories
        event_logs: Optional list of EventLog objects
        event_types: Which event types to detect
    
    Returns:
        List of EventLabels
    """
    detector = EventDetector()
    
    labels = []
    for i, traj in enumerate(trajectories):
        event_log = event_logs[i] if event_logs else None
        label = detector.label_trajectory(traj, event_log, event_types)
        labels.append(label)
    
    return labels


def compute_class_weights(
    labels: List[EventLabels],
) -> Dict[str, float]:
    """
    Compute class weights for balanced training.
    
    Prevents event head collapse to always-on/off.
    """
    total_pos = 0
    total_neg = 0
    
    for label in labels:
        total_pos += int(jnp.sum(label.y_window))
        total_neg += int(jnp.sum(1 - label.y_window))
    
    # Inverse frequency weighting
    pos_weight = total_neg / (total_pos + 1e-8)
    neg_weight = 1.0
    
    return {
        'pos_weight': pos_weight,
        'neg_weight': neg_weight,
        'total_pos': total_pos,
        'total_neg': total_neg,
    }