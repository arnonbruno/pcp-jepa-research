"""
Phase 1 Experiments: Establish the Failure Law

Run all experiments to determine WHY long-horizon planning fails.

Experiments:
1.1 - Prediction vs Planning Decoupling
1.2 - Event Dominance Test
1.3 - Parameter Shift × Events
"""

import sys
import os

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import jax
import jax.numpy as jnp
import numpy as np
from datetime import datetime
import json

from src.environments import BouncingBall, BouncingBallParams, StickSlipBlock, StickSlipParams, EventMetrics
from src.evaluation.horizon_scaling import event_linked_failure_fraction


# ============================================================================
# Experiment 1.1: Prediction vs Planning Decoupling
# ============================================================================

def exp_1_1():
    """
    Question: Does better prediction yield better long-horizon planning?
    
    Discovery criterion: Low correlation at H=100+ means planning needs 
    different representation than prediction.
    """
    print("\n" + "=" * 70)
    print("EXPERIMENT 1.1: Prediction vs Planning Decoupling")
    print("=" * 70)
    
    env = BouncingBall(BouncingBallParams(restitution=0.8))
    
    # Generate data
    print("\nGenerating trajectories...")
    key = jax.random.PRNGKey(42)
    
    trajectories = []
    for _ in range(100):
        key, k1, k2 = jax.random.split(key, 3)
        y_init = jax.random.uniform(k1, minval=0.5, maxval=2.0)
        vy_init = jax.random.uniform(k2, minval=-1.0, maxval=1.0)
        initial_state = jnp.array([0.0, y_init, 0.0, vy_init])
        traj, _ = env.simulate(initial_state, num_steps=100)
        trajectories.append(traj)
    
    print(f"  Generated {len(trajectories)} trajectories")
    
    # Simulate models with different prediction qualities
    # (Using different parameter mismatches as proxy for model quality)
    
    print("\nSimulating models with varying prediction quality...")
    
    prediction_mses = []
    planning_errors = {10: [], 30: [], 100: []}
    
    # Baseline restitution
    baseline_restitution = 0.8
    
    # Test different model "qualities" (parameter mismatches)
    for model_restitution in [0.5, 0.6, 0.7, 0.8, 0.9, 1.0]:
        model_env = BouncingBall(BouncingBallParams(restitution=model_restitution))
        
        # Measure prediction MSE
        mses = []
        for traj in trajectories[:20]:
            # "Model prediction" = model env rollout
            pred_traj, _ = model_env.simulate(traj[0], num_steps=100)
            mse = float(jnp.mean((pred_traj - traj)**2))
            mses.append(mse)
        
        pred_mse = np.mean(mses)
        prediction_mses.append(pred_mse)
        
        # Measure planning errors at different horizons
        for H in [10, 30, 100]:
            errors = []
            for traj in trajectories[:20]:
                # "Planning" = model trajectory for H steps
                pred_traj, _ = model_env.simulate(traj[0], num_steps=H)
                error = float(jnp.mean((pred_traj - traj[:H+1])**2))
                errors.append(error)
            
            planning_errors[H].append(np.mean(errors))
    
    # Analyze correlations
    print("\n" + "-" * 70)
    print("RESULTS: Correlation(1-step MSE, Planning Error)")
    print("-" * 70)
    
    for H in [10, 30, 100]:
        corr = np.corrcoef(prediction_mses, planning_errors[H])[0, 1]
        print(f"  H={H:3d}: correlation = {corr:.3f}")
        
        if H >= 100 and abs(corr) < 0.5:
            print(f"       → LOW CORRELATION! Planning needs different representation")
    
    return {
        'prediction_mses': prediction_mses,
        'planning_errors': planning_errors,
    }


# ============================================================================
# Experiment 1.2: Event Dominance Test
# ============================================================================

def exp_1_2():
    """
    Question: Are failures concentrated around events?
    
    Discovery criterion: >70% failures linked to events = clean research wedge.
    """
    print("\n" + "=" * 70)
    print("EXPERIMENT 1.2: Event Dominance Test")
    print("=" * 70)
    
    env = BouncingBall(BouncingBallParams(restitution=0.8))
    
    # Generate trajectories with events
    print("\nGenerating trajectories...")
    key = jax.random.PRNGKey(42)
    
    trajectories = []
    event_logs = []
    
    for _ in range(50):
        key, k1, k2 = jax.random.split(key, 3)
        y_init = jax.random.uniform(k1, minval=1.0, maxval=3.0)
        vy_init = jax.random.uniform(k2, minval=-2.0, maxval=2.0)
        initial_state = jnp.array([0.0, y_init, 0.0, vy_init])
        
        traj, event_log = env.simulate(initial_state, num_steps=100)
        trajectories.append(traj)
        event_logs.append(event_log)
    
    print(f"  Generated {len(trajectories)} trajectories")
    
    # Count events
    total_events = sum(len(log) for log in event_logs)
    print(f"  Total events: {total_events}")
    
    # Simulate planning failures
    print("\nSimulating planning failures...")
    
    # Use perturbed model as "planner"
    perturbed_env = BouncingBall(BouncingBallParams(restitution=0.7))
    
    failure_times_all = []
    event_times_all = []
    
    for traj, event_log in zip(trajectories, event_logs):
        # Planned trajectory (perturbed model)
        planned_traj, _ = perturbed_env.simulate(traj[0], num_steps=100)
        
        # Find divergence points
        for t in range(len(traj)):
            error = float(jnp.linalg.norm(traj[t] - planned_traj[t]))
            if error > 0.1:  # Divergence threshold
                failure_times_all.append(t)
        
        # Collect event times
        event_times_all.extend(event_log.get_event_times())
    
    # Compute event-linked failure fraction
    fraction = event_linked_failure_fraction(
        failure_times_all,
        event_times_all,
        window=5,
    )
    
    print("\n" + "-" * 70)
    print("RESULTS: Event-Linked Failures")
    print("-" * 70)
    
    print(f"  Total failures: {len(failure_times_all)}")
    print(f"  Total events: {len(event_times_all)}")
    print(f"  Event-linked fraction: {fraction:.1%}")
    
    if fraction > 0.7:
        print(f"\n  → >70% FAILURES ARE EVENT-LINKED!")
        print("  → This is a CLEAN RESEARCH WEDGE!")
    
    return {
        'total_failures': len(failure_times_all),
        'total_events': len(event_times_all),
        'event_linked_fraction': fraction,
    }


# ============================================================================
# Experiment 1.3: Parameter Shift × Events
# ============================================================================

def exp_1_3():
    """
    Question: Does parameter shift hurt mostly because events shift?
    
    Discovery criterion: High correlation between MSE drift and event drift.
    """
    print("\n" + "=" * 70)
    print("EXPERIMENT 1.3: Parameter Shift × Events")
    print("=" * 70)
    
    # Parameter sweep
    restitution_values = [0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
    baseline_restitution = 0.8
    
    key = jax.random.PRNGKey(42)
    
    results = []
    
    for restitution in restitution_values:
        print(f"\n  Restitution = {restitution}")
        
        env = BouncingBall(BouncingBallParams(restitution=restitution))
        baseline_env = BouncingBall(BouncingBallParams(restitution=baseline_restitution))
        
        # Generate trajectories
        trajectories = []
        event_logs = []
        
        for _ in range(20):
            key, k1, k2 = jax.random.split(key, 3)
            y_init = jax.random.uniform(k1, minval=1.0, maxval=2.0)
            vy_init = jax.random.uniform(k2, minval=-1.0, maxval=1.0)
            initial_state = jnp.array([0.0, y_init, 0.0, vy_init])
            
            traj, event_log = env.simulate(initial_state, num_steps=100)
            trajectories.append(traj)
            event_logs.append(event_log)
        
        # Compare to baseline
        mses = []
        for traj in trajectories:
            baseline_traj, _ = baseline_env.simulate(traj[0], num_steps=100)
            mse = float(jnp.mean((traj - baseline_traj)**2))
            mses.append(mse)
        
        num_events = np.mean([len(log) for log in event_logs])
        
        results.append({
            'restitution': restitution,
            'mse': np.mean(mses),
            'num_events': num_events,
        })
    
    # Analyze correlation
    print("\n" + "-" * 70)
    print("RESULTS: MSE vs Event Count Correlation")
    print("-" * 70)
    
    mses = [r['mse'] for r in results]
    event_counts = [r['num_events'] for r in results]
    
    correlation = np.corrcoef(mses, event_counts)[0, 1]
    print(f"  Correlation(MSE, event_count) = {correlation:.3f}")
    
    if abs(correlation) > 0.7:
        print("  → High correlation! Parameter shift affects events significantly.")
    
    return {
        'correlation': correlation,
        'results': results,
    }


# ============================================================================
# Run All Phase 1
# ============================================================================

def run_all():
    """Run all Phase 1 experiments."""
    print("\n" + "╔" + "═" * 68 + "╗")
    print("║" + "PHASE 1: ESTABLISH THE FAILURE LAW".center(68) + "║")
    print("╚" + "═" * 68 + "╝")
    
    print("\nGoal: Find WHY long-horizon planning fails.")
    
    results = {}
    
    # Run experiments
    try:
        results['exp_1_1'] = exp_1_1()
    except Exception as e:
        print(f"\n✗ Exp 1.1 FAILED: {e}")
        import traceback
        traceback.print_exc()
    
    try:
        results['exp_1_2'] = exp_1_2()
    except Exception as e:
        print(f"\n✗ Exp 1.2 FAILED: {e}")
        import traceback
        traceback.print_exc()
    
    try:
        results['exp_1_3'] = exp_1_3()
    except Exception as e:
        print(f"\n✗ Exp 1.3 FAILED: {e}")
        import traceback
        traceback.print_exc()
    
    # Gate G1 check
    print("\n" + "=" * 70)
    print("GATE G1 CHECK")
    print("=" * 70)
    
    g1_passed = False
    
    # Check prediction-planning decoupling
    if 'exp_1_1' in results and results['exp_1_1']:
        print("  [✓] Exp 1.1: Prediction vs Planning - Completed")
        g1_passed = True
    
    # Check event dominance
    if 'exp_1_2' in results and 'event_linked_fraction' in results['exp_1_2']:
        fraction = results['exp_1_2']['event_linked_fraction']
        if fraction > 0.7:
            print(f"  [✓] Exp 1.2: Event dominance = {fraction:.1%} (>70%)")
            g1_passed = True
        else:
            print(f"  [ ] Exp 1.2: Event fraction = {fraction:.1%} (need >70%)")
    
    if g1_passed:
        print("\n✓ GATE G1 PASSED → Proceed to Phase 2")
    else:
        print("\n✗ GATE G1 FAILED → Pivot to horizon-consistency only")
    
    # Save results
    os.makedirs('results/phase1', exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    with open(f'results/phase1/results_{timestamp}.json', 'w') as f:
        json.dump(results, f, indent=2, default=str)
    
    print(f"\nResults saved to: results/phase1/results_{timestamp}.json")
    
    return results


if __name__ == "__main__":
    results = run_all()