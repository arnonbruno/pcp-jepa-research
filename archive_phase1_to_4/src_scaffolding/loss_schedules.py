"""
Loss Weight Schedules for Phase 2 Objectives

Based on practical schedules for BouncingBall with sparse impacts (Δ=5 window).

Key principles:
- 3-stage curriculum to avoid collapse and instability
- Adaptive rules based on observed metrics
- Event-window oversampling for sparse events
"""

import jax.numpy as jnp
from typing import Dict, Tuple
from dataclasses import dataclass


# ============================================================================
# O1 Schedules
# ============================================================================

@dataclass
class O1Weights:
    """O1 loss weights."""
    lambda_evt: float
    lambda_timing: float
    lambda_seq: float


def get_o1_weights_stage(stage: int) -> O1Weights:
    """
    Get O1 weights for a training stage.
    
    Stage 0: Warm-start event detection (first 10-20%)
    Stage 1: Add timing alignment (next 30-40%)
    Stage 2: Add event-consistent rollouts (final 40-60%)
    """
    if stage == 0:
        return O1Weights(lambda_evt=1.0, lambda_timing=0.0, lambda_seq=0.0)
    elif stage == 1:
        return O1Weights(lambda_evt=0.5, lambda_timing=0.5, lambda_seq=0.0)
    else:  # stage == 2
        return O1Weights(lambda_evt=0.2, lambda_timing=0.3, lambda_seq=0.5)


def get_o1_weights_continuous(progress: float) -> O1Weights:
    """
    Get O1 weights based on training progress p ∈ [0, 1].
    
    Uses smooth formulas instead of discrete stages.
    """
    import math
    
    # λ_evt(p) = 1.0 * exp(-3p) + 0.15
    lambda_evt = 1.0 * math.exp(-3 * progress) + 0.15
    
    # λ_timing(p) = 0.6 * sigmoid(10*(p-0.2)) * (1 - 0.4*p)
    sigmoid = 1.0 / (1.0 + math.exp(-10 * (progress - 0.2)))
    lambda_timing = 0.6 * sigmoid * (1 - 0.4 * progress)
    
    # λ_seq(p) = 0.6 * sigmoid(10*(p-0.5))
    sigmoid_seq = 1.0 / (1.0 + math.exp(-10 * (progress - 0.5)))
    lambda_seq = 0.6 * sigmoid_seq
    
    # Clamp to sane ranges
    lambda_evt = max(0.15, min(1.0, lambda_evt))
    lambda_timing = max(0.0, min(0.8, lambda_timing))
    lambda_seq = max(0.0, min(0.6, lambda_seq))
    
    return O1Weights(lambda_evt, lambda_timing, lambda_seq)


def adapt_o1_weights(
    current: O1Weights,
    metrics: Dict[str, float],
) -> Tuple[O1Weights, str]:
    """
    Adapt O1 weights based on observed metrics.
    
    Triggers:
    - p_t collapse (mean < 0.02 or > 0.98)
    - Recall@Δ fine but timing error high
    - Training unstable when λ_seq starts
    
    Returns:
        (new_weights, reason_for_change)
    """
    event_mean = metrics.get('event_prob_mean', 0.5)
    recall = metrics.get('event_recall', 0.0)
    timing_error = metrics.get('event_timing_error', float('inf'))
    
    # Check for collapse
    if event_mean < 0.02 or event_mean > 0.98:
        return (
            O1Weights(lambda_evt=1.0, lambda_timing=0.0, lambda_seq=0.0),
            f"Event head collapse detected (mean={event_mean:.3f})"
        )
    
    # Check timing error
    if recall >= 0.80 and timing_error > 2.0:
        new_timing = min(0.8, current.lambda_timing + 0.1)
        new_evt = max(0.2, current.lambda_evt - 0.1)
        return (
            O1Weights(lambda_evt=new_evt, lambda_timing=new_timing, lambda_seq=current.lambda_seq),
            f"High timing error ({timing_error:.1f}), increased λ_timing"
        )
    
    # Check stability with λ_seq
    if current.lambda_seq > 0.3 and metrics.get('loss_variance', 0) > 1.0:
        return (
            O1Weights(lambda_evt=current.lambda_evt, lambda_timing=current.lambda_timing, lambda_seq=0.3),
            "Unstable with λ_seq, reducing"
        )
    
    return (current, "No adaptation needed")


# ============================================================================
# O3 Schedules
# ============================================================================

@dataclass
class O3Weights:
    """O3 loss weights and risk parameter."""
    lambda_varshape: float
    beta_risk: float


def get_o3_weights_stage(stage: int) -> O3Weights:
    """
    Get O3 weights for a training stage.
    
    Stage 0: First 20% (λ_varshape small, β_risk = 0)
    Stage 1: Next 30% (λ_varshape medium, β_risk = 0.25)
    Stage 2: Final 50% (λ_varshape high, β_risk = 0.5)
    """
    if stage == 0:
        return O3Weights(lambda_varshape=0.1, beta_risk=0.0)
    elif stage == 1:
        return O3Weights(lambda_varshape=0.3, beta_risk=0.25)
    else:  # stage == 2
        return O3Weights(lambda_varshape=0.5, beta_risk=0.5)


def adapt_o3_weights(
    current: O3Weights,
    metrics: Dict[str, float],
) -> Tuple[O3Weights, str]:
    """
    Adapt O3 weights based on observed metrics.
    
    Monitors:
    - R = mean(σ_event) / mean(σ_nonevent)
    - Catastrophic rate improvement with risk
    """
    R = metrics.get('std_ratio', 1.0)  # σ_event / σ_nonevent
    nll = metrics.get('nll_loss', float('inf'))
    
    # Target ratio progression: 1.5 → 2.0 → 2.5-4.0
    if R < 1.3:
        new_varshape = min(0.7, current.lambda_varshape + 0.1)
        return (
            O3Weights(lambda_varshape=new_varshape, beta_risk=current.beta_risk),
            f"Low std_ratio ({R:.2f}), increasing λ_varshape"
        )
    
    if R > 6.0 or nll > 2.0:
        new_varshape = max(0.1, current.lambda_varshape - 0.1)
        return (
            O3Weights(lambda_varshape=new_varshape, beta_risk=current.beta_risk),
            f"High std_ratio ({R:.2f}) or NLL ({nll:.2f}), decreasing λ_varshape"
        )
    
    # Check catastrophic improvement with risk
    if current.beta_risk > 0:
        catastrophics_neutral = metrics.get('catastrophics_neutral', 100)
        catastrophics_risk = metrics.get('catastrophics_risk', 100)
        improvement = (catastrophics_neutral - catastrophics_risk) / max(catastrophics_neutral, 1)
        
        if improvement < 0.10 and current.beta_risk < 1.0:
            return (
                O3Weights(lambda_varshape=current.lambda_varshape, beta_risk=current.beta_risk + 0.1),
                "Insufficient catastrophic improvement, increasing β_risk"
            )
        
        success_drop = metrics.get('success_drop', 0)
        if success_drop > 0.1:
            return (
                O3Weights(lambda_varshape=current.lambda_varshape, beta_risk=max(0.0, current.beta_risk - 0.1)),
                "Success rate dropping away from events, decreasing β_risk"
            )
    
    return (current, "No adaptation needed")


# ============================================================================
# O2 Schedules
# ============================================================================

@dataclass
class O2Weights:
    """O2 loss weights."""
    event_weight: float  # w_evt
    use_gradient_alignment: bool


def get_o2_weights_stage(stage: int) -> O2Weights:
    """
    Get O2 weights for a training stage.
    
    Stage 0: First 30% (w_evt = 2)
    Stage 1: Next 30% (w_evt = 5)
    Stage 2: Final 40% (w_evt = 10)
    
    Gradient alignment OFF until stable improvements.
    """
    if stage == 0:
        return O2Weights(event_weight=2, use_gradient_alignment=False)
    elif stage == 1:
        return O2Weights(event_weight=5, use_gradient_alignment=False)
    else:  # stage == 2
        return O2Weights(event_weight=10, use_gradient_alignment=False)


def adapt_o2_weights(
    current: O2Weights,
    metrics: Dict[str, float],
) -> Tuple[O2Weights, str]:
    """
    Adapt O2 weights based on observed metrics.
    
    Monitors:
    - Fraction of catastrophics that are event-linked
    - Event timing error
    """
    event_linked_fraction = metrics.get('event_linked_fraction', 1.0)
    
    # If event-linked catastrophics remain high
    if event_linked_fraction > 0.80:
        new_weight = min(15, current.event_weight + 2)
        return (
            O2Weights(event_weight=new_weight, use_gradient_alignment=current.use_gradient_alignment),
            f"Event-linked catastrophics still high ({event_linked_fraction:.1%}), increasing w_evt"
        )
    
    # If non-event prediction degrades
    non_event_mse = metrics.get('non_event_mse', 0)
    event_mse = metrics.get('event_mse', 0)
    
    if non_event_mse > 2 * event_mse and current.event_weight > 2:
        new_weight = max(2, current.event_weight - 2)
        return (
            O2Weights(event_weight=new_weight, use_gradient_alignment=current.use_gradient_alignment),
            "Non-event prediction degrading, decreasing w_evt"
        )
    
    return (current, "No adaptation needed")


# ============================================================================
# Stage Determination
# ============================================================================

def determine_stage(
    progress: float,
    num_stages: int = 3,
) -> int:
    """Determine current stage based on training progress."""
    if num_stages == 3:
        if progress < 0.2:
            return 0
        elif progress < 0.6:
            return 1
        else:
            return 2
    else:
        # General case
        stage_size = 1.0 / num_stages
        return min(int(progress / stage_size), num_stages - 1)


def compute_progress(epoch: int, total_epochs: int) -> float:
    """Compute training progress p ∈ [0, 1]."""
    return min(1.0, epoch / max(total_epochs, 1))


# ============================================================================
# Event-Window Oversampling
# ============================================================================

def create_event_weighted_sampler(
    event_labels,
    target_event_fraction: float = 0.4,
):
    """
    Create sampler that ensures target fraction of samples include events.
    
    For BouncingBall with sparse events (100 events vs 3430 failures),
    this prevents training from being dominated by non-event steps.
    
    Args:
        event_labels: List of EventLabels objects
        target_event_fraction: Desired fraction of samples with events
    
    Returns:
        indices: Sample indices with oversampled event windows
    """
    import numpy as np
    
    # Find trajectories with events
    has_event = []
    for i, label in enumerate(event_labels):
        if hasattr(label, 'y_window'):
            has_event.append(float(jnp.any(label.y_window > 0.5)))
        else:
            has_event.append(0.0)
    
    has_event = np.array(has_event)
    
    # Indices with and without events
    event_indices = np.where(has_event > 0.5)[0]
    non_event_indices = np.where(has_event <= 0.5)[0]
    
    if len(event_indices) == 0:
        # No events, return all indices
        return list(range(len(event_labels)))
    
    # Oversample event indices
    n_total = len(event_labels)
    n_event_target = int(n_total * target_event_fraction)
    
    # Sample with replacement from event indices
    sampled_event = np.random.choice(
        event_indices,
        size=min(n_event_target, len(event_indices) * 3),
        replace=True,
    )
    
    # Sample from non-event indices
    n_non_event = n_total - len(sampled_event)
    if len(non_event_indices) > 0:
        sampled_non_event = np.random.choice(
            non_event_indices,
            size=min(n_non_event, len(non_event_indices)),
            replace=False,
        )
    else:
        sampled_non_event = np.array([], dtype=int)
    
    # Combine and shuffle
    all_indices = np.concatenate([sampled_event, sampled_non_event])
    np.random.shuffle(all_indices)
    
    return list(all_indices)


# ============================================================================
# Summary
# ============================================================================

def print_schedule_summary():
    """Print summary of recommended schedules."""
    print("\n" + "=" * 70)
    print("RECOMMENDED LOSS WEIGHT SCHEDULES")
    print("=" * 70)
    
    print("\nO1 (Event-Consistency):")
    print("  Stage 0 (0-20%):   λ_evt=1.0, λ_timing=0.0, λ_seq=0.0")
    print("  Stage 1 (20-60%):  λ_evt=0.5, λ_timing=0.5, λ_seq=0.0")
    print("  Stage 2 (60-100%): λ_evt=0.2, λ_timing=0.3, λ_seq=0.5")
    
    print("\nO3 (Event-Localized Uncertainty):")
    print("  Stage 0 (0-20%):   λ_varshape=0.1, β_risk=0.0")
    print("  Stage 1 (20-50%):  λ_varshape=0.3, β_risk=0.25")
    print("  Stage 2 (50-100%): λ_varshape=0.5, β_risk=0.5")
    
    print("\nO2 (Horizon-Consistency):")
    print("  Stage 0 (0-30%):   w_evt=2")
    print("  Stage 1 (30-60%):  w_evt=5")
    print("  Stage 2 (60-100%): w_evt=10")
    print("  (Gradient alignment OFF until stable)")
    
    print("\nEvent-Window Oversampling:")
    print("  Target: 30-50% of samples include event windows")
    print("  Prevents non-event domination with sparse events")
    
    print("\n" + "=" * 70)


if __name__ == "__main__":
    print_schedule_summary()
    
    # Test continuous schedule
    print("\nO1 Continuous Schedule Test:")
    for p in [0.0, 0.2, 0.4, 0.6, 0.8, 1.0]:
        w = get_o1_weights_continuous(p)
        print(f"  p={p:.1f}: λ_evt={w.lambda_evt:.3f}, λ_timing={w.lambda_timing:.3f}, λ_seq={w.lambda_seq:.3f}")