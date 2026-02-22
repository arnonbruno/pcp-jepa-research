"""
Evaluation Metrics for PCP-JEPA Research

Key metrics:
- Horizon scaling curves (H=10, 30, 100, 300)
- Event timing error
- Event-conditioned calibration
- Parameter OOD curves
- Planning robustness
"""

import jax.numpy as jnp
import numpy as np
from typing import Dict, List, Tuple
from dataclasses import dataclass


@dataclass
class HorizonScalingResults:
    """Results from horizon scaling evaluation."""
    horizon: int
    success_rate: float
    mean_return: float
    event_timing_error: float
    prediction_mse: float


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
        initial_states,
        planner_fn,
        cost_fn,
    ) -> List[HorizonScalingResults]:
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
            List of HorizonScalingResults for each horizon
        """
        results = []
        
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
                predicted_events = []
                true_events = event_log.get_event_times()
                timing = self._event_timing_error(predicted_events, true_events)
                
                # Prediction MSE
                mse = float(jnp.mean((planned_traj - true_traj)**2))
                
                successes.append(float(success))
                returns.append(float(return_val))
                timing_errors.append(timing['timing_error'])
                mses.append(mse)
            
            results.append(HorizonScalingResults(
                horizon=H,
                success_rate=np.mean(successes),
                mean_return=np.mean(returns),
                event_timing_error=np.mean(timing_errors),
                prediction_mse=np.mean(mses),
            ))
        
        return results
    
    def _event_timing_error(
        self,
        predicted_events: List[int],
        true_events: List[int],
        tolerance: int = 2,
    ) -> Dict[str, float]:
        """Measure how accurately events are predicted."""
        if len(true_events) == 0:
            return {'precision': 1.0, 'recall': 1.0, 'timing_error': 0.0}
        
        if len(predicted_events) == 0:
            return {'precision': 0.0, 'recall': 0.0, 'timing_error': float('inf')}
        
        # Match predicted to true events
        matched_true = set()
        timing_errors = []
        
        for pred_t in predicted_events:
            distances = [abs(pred_t - true_t) for true_t in true_events]
            min_idx = np.argmin(distances)
            min_dist = distances[min_idx]
            
            if min_dist <= tolerance:
                matched_true.add(min_idx)
                timing_errors.append(min_dist)
        
        precision = len(matched_true) / len(predicted_events)
        recall = len(matched_true) / len(true_events)
        timing_error = np.mean(timing_errors) if timing_errors else 0.0
        
        return {
            'precision': precision,
            'recall': recall,
            'timing_error': timing_error,
        }


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
        for event_t in event_times:
            if 0 <= div_t - event_t <= window:
                linked_count += 1
                break
    
    return linked_count / len(divergence_times)


def plot_horizon_scaling(results: List[HorizonScalingResults], save_path: str = None):
    """Plot horizon scaling curves."""
    import matplotlib.pyplot as plt
    
    horizons = [r.horizon for r in results]
    success_rates = [r.success_rate for r in results]
    timing_errors = [r.event_timing_error for r in results]
    mses = [r.prediction_mse for r in results]
    
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    
    # Success rate vs horizon
    axes[0].plot(horizons, success_rates, 'o-', linewidth=2, markersize=8)
    axes[0].set_xlabel('Planning Horizon')
    axes[0].set_ylabel('Success Rate')
    axes[0].set_title('Success Rate vs Horizon')
    axes[0].set_xscale('log')
    axes[0].grid(True, alpha=0.3)
    
    # Event timing error vs horizon
    axes[1].plot(horizons, timing_errors, 's-', linewidth=2, markersize=8, color='orange')
    axes[1].set_xlabel('Planning Horizon')
    axes[1].set_ylabel('Event Timing Error')
    axes[1].set_title('Event Timing Error vs Horizon')
    axes[1].set_xscale('log')
    axes[1].grid(True, alpha=0.3)
    
    # Prediction MSE vs horizon
    axes[2].plot(horizons, mses, '^-', linewidth=2, markersize=8, color='green')
    axes[2].set_xlabel('Planning Horizon')
    axes[2].set_ylabel('Prediction MSE')
    axes[2].set_title('Prediction MSE vs Horizon')
    axes[2].set_xscale('log')
    axes[2].grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    
    return fig