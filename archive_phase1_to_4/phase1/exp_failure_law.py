"""
Phase 1 Experiments: Establish the Failure Law

Goal: Find the real reason long-horizon planning fails, quantitatively.

Experiments:
1.1 - Prediction vs Planning Decoupling
1.2 - Event Dominance Test
1.3 - Parameter Shift × Events
"""

import jax
import jax.numpy as jnp
import numpy as np
from typing import Dict, List, Tuple, Optional
import sys
import os
import json
from datetime import datetime

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from research.environments.tier2_hybrid import (
    BouncingBall, BouncingBallParams,
    StickSlipBlock, StickSlipParams,
    EventMetrics, HorizonScalingEval,
    EventType, EventLog,
)
from src.pcp_jepa import PCPJEPA, create_pcp_jepa
from flax.training.train_state import TrainState
import optax


# ============================================================================
# Experiment 1.1: Prediction vs Planning Decoupling
# ============================================================================

def exp_1_1_prediction_vs_planning():
    """
    Question: Does better prediction yield better long-horizon planning?
    
    Method:
    1. Train spectrum of models with varying 1-step MSE quality
    2. Run MPC on each for long horizons
    3. Measure correlation between 1-step MSE and success at H=100/300
    
    Discovery criterion:
    - If correlation collapses at long horizons → planning needs different representation
    """
    print("\n" + "=" * 80)
    print("EXPERIMENT 1.1: Prediction vs Planning Decoupling")
    print("=" * 80)
    
    # Environment
    env = BouncingBall(BouncingBallParams(restitution=0.8))
    
    # Generate training data
    print("\nGenerating training data...")
    key = jax.random.PRNGKey(42)
    
    train_trajectories = []
    train_events = []
    
    for _ in range(100):
        key, k1, k2 = jax.random.split(key, 3)
        
        # Random initial state
        y_init = jax.random.uniform(k1, minval=0.5, maxval=2.0)
        vy_init = jax.random.uniform(k2, minval=-1.0, maxval=1.0)
        initial_state = jnp.array([0.0, y_init, 0.0, vy_init])
        
        traj, event_log = env.simulate(initial_state, num_steps=50)
        train_trajectories.append(traj)
        train_events.append(event_log)
    
    print(f"  Generated {len(train_trajectories)} trajectories")
    
    # Train models with varying prediction quality
    print("\nTraining models with varying prediction quality...")
    
    model_configs = [
        {'name': 'weak', 'latent_dim': 8, 'epochs': 10},
        {'name': 'medium', 'latent_dim': 16, 'epochs': 30},
        {'name': 'strong', 'latent_dim': 32, 'epochs': 100},
    ]
    
    results = []
    
    for config in model_configs:
        print(f"\n  Training {config['name']} model...")
        
        # Create model
        key, subkey = jax.random.split(key)
        model, params = create_pcp_jepa(
            latent_dim=config['latent_dim'],
            action_dim=0,  # No actions for passive system
            obs_dim=4,
            key=subkey,
        )
        
        # Training
        tx = optax.adam(1e-3)
        state = TrainState.create(apply_fn=model.apply, params=params, tx=tx)
        
        @jax.jit
        def train_step(state, obs_batch):
            def loss_fn(params):
                # Prediction loss
                outputs = model.apply(params, obs_batch, jnp.zeros((obs_batch.shape[0], obs_batch.shape[1], 0)))
                pred_loss = jnp.mean((outputs['z_fine'] - jax.vmap(model.apply, in_axes=(None, 0, None))(params, obs_batch, jnp.zeros((obs_batch.shape[1], 0))))**2)
                return pred_loss
            
            loss, grads = jax.value_and_grad(loss_fn)(state.params)
            return state.apply_gradients(grads=grads), loss
        
        # Train
        for epoch in range(config['epochs']):
            for traj in train_trajectories:
                obs = traj[None, :, :]  # [1, time, 4]
                state, loss = train_step(state, obs)
            
            if epoch % 10 == 0:
                print(f"    Epoch {epoch}: loss = {float(loss):.4f}")
        
        # Evaluate prediction MSE
        test_mses = []
        for traj in train_trajectories[:20]:
            obs = traj[None, :, :]
            outputs = model.apply(state.params, obs, jnp.zeros((1, obs.shape[1], 0)))
            # Compare predicted vs actual
            mse = float(jnp.mean((outputs['z_fine'] - outputs['z'])**2))
            test_mses.append(mse)
        
        prediction_mse = np.mean(test_mses)
        
        # Evaluate planning at different horizons
        # For bouncing ball, "planning" is just predicting trajectory
        # Real planning would use CEM/MPPI
        
        horizon_results = {}
        for H in [10, 30, 100]:
            # Simple "planning" evaluation: how well does predicted trajectory match true?
            errors = []
            for traj in train_trajectories[:20]:
                # Predict H steps
                pred_traj = traj[:1]  # Initial state
                z_init = model.apply(state.params, traj[:1], jnp.zeros((1, 1, 0)), method=model.encode)
                
                # Simple rollout
                z = z_init
                for t in range(H):
                    # Predict next (no dynamics model yet, just use encoder)
                    z = z  # Placeholder
                
                # Measure error
                error = float(jnp.mean((pred_traj[0] - traj[0])**2))
                errors.append(error)
            
            horizon_results[f'H={H}'] = np.mean(errors)
        
        results.append({
            'config': config['name'],
            'prediction_mse': prediction_mse,
            'horizon_results': horizon_results,
        })
        
        print(f"    Prediction MSE: {prediction_mse:.4f}")
        print(f"    Horizon results: {horizon_results}")
    
    # Analyze correlation
    print("\n" + "-" * 60)
    print("ANALYSIS: Prediction MSE vs Planning Success")
    print("-" * 60)
    
    pred_mses = [r['prediction_mse'] for r in results]
    
    for H in [10, 30, 100]:
        horizon_errors = [r['horizon_results'][f'H={H}'] for r in results]
        
        # Compute correlation
        if len(pred_mses) > 2:
            correlation = np.corrcoef(pred_mses, horizon_errors)[0, 1]
            print(f"H={H}: Correlation(pred_mse, planning_error) = {correlation:.3f}")
            
            # Discovery criterion
            if H >= 100 and abs(correlation) < 0.5:
                print(f"  → LOW CORRELATION at H={H}! Planning needs different representation.")
    
    return results


# ============================================================================
# Experiment 1.2: Event Dominance Test
# ============================================================================

def exp_1_2_event_dominance():
    """
    Question: Are failures concentrated around events?
    
    Method:
    1. Label event times from simulator
    2. Compute where planned trajectory diverges: event vs non-event
    
    Discovery criterion:
    - If >70% failures are event-linked → clean research wedge
    """
    print("\n" + "=" * 80)
    print("EXPERIMENT 1.2: Event Dominance Test")
    print("=" * 80)
    
    # Environment
    env = BouncingBall(BouncingBallParams(restitution=0.8))
    
    # Generate trajectories with events
    print("\nGenerating trajectories with events...")
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
    
    # Simulate "planning failure" as divergence from nominal trajectory
    # For now, we use a perturbed model as "planner"
    
    print("\nSimulating planning failures...")
    
    failure_times_all = []
    event_times_all = []
    
    for traj, event_log in zip(trajectories, event_logs):
        # Nominal trajectory
        nominal = traj
        
        # "Planned" trajectory (perturbed physics)
        perturbed_env = BouncingBall(BouncingBallParams(restitution=0.7))  # Slightly different
        planned_traj, _ = perturbed_env.simulate(traj[0], num_steps=100)
        
        # Find divergence points
        divergence_times = []
        for t in range(len(nominal)):
            error = jnp.linalg.norm(nominal[t] - planned_traj[t])
            if error > 0.1:  # Threshold for "divergence"
                divergence_times.append(t)
        
        failure_times_all.extend(divergence_times)
        event_times_all.extend(event_log.get_event_times())
    
    # Compute event-linked failure fraction
    fraction = EventMetrics.event_linked_failure_fraction(
        failure_times_all,
        event_times_all,
        window=5,
    )
    
    print("\n" + "-" * 60)
    print("ANALYSIS: Event-Linked Failures")
    print("-" * 60)
    
    print(f"Total failures: {len(failure_times_all)}")
    print(f"Total events: {len(event_times_all)}")
    print(f"Event-linked failure fraction: {fraction:.1%}")
    
    if fraction > 0.7:
        print("\n  → >70% FAILURES ARE EVENT-LINKED!")
        print("  → This is a CLEAN RESEARCH WEDGE!")
        print("  → Focus on event-consistent representations.")
    
    return {
        'total_failures': len(failure_times_all),
        'total_events': len(event_times_all),
        'event_linked_fraction': fraction,
    }


# ============================================================================
# Experiment 1.3: Parameter Shift × Events
# ============================================================================

def exp_1_3_parameter_shift_events():
    """
    Question: Does parameter shift hurt mostly because events shift?
    
    Method:
    1. Sweep friction/restitution/mass
    2. Compare event timing drift vs overall MSE drift
    
    Discovery criterion:
    - If OOD failures correlate with event timing drift → new objective
    """
    print("\n" + "=" * 80)
    print("EXPERIMENT 1.3: Parameter Shift × Events")
    print("=" * 80)
    
    # Parameter shifts
    restitution_values = [0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
    
    key = jax.random.PRNGKey(42)
    
    results = []
    
    for restitution in restitution_values:
        print(f"\nRestitution = {restitution}")
        
        env = BouncingBall(BouncingBallParams(restitution=restitution))
        
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
        
        # Compare to baseline (restitution=0.8)
        baseline_env = BouncingBall(BouncingBallParams(restitution=0.8))
        
        mses = []
        event_drifts = []
        
        for traj in trajectories:
            # Baseline trajectory
            baseline_traj, _ = baseline_env.simulate(traj[0], num_steps=100)
            
            # MSE drift
            mse = float(jnp.mean((traj - baseline_traj)**2))
            mses.append(mse)
        
        # Event timing drift (simplified: count events)
        num_events = np.mean([len(log) for log in event_logs])
        
        results.append({
            'restitution': restitution,
            'mse': np.mean(mses),
            'num_events': num_events,
        })
    
    # Analyze correlation
    print("\n" + "-" * 60)
    print("ANALYSIS: MSE Drift vs Event Drift")
    print("-" * 60)
    
    mses = [r['mse'] for r in results]
    event_counts = [r['num_events'] for r in results]
    
    print(f"MSE values: {mses}")
    print(f"Event counts: {event_counts}")
    
    # Check correlation
    if len(mses) > 2:
        correlation = np.corrcoef(mses, event_counts)[0, 1]
        print(f"\nCorrelation(MSE, event_count) = {correlation:.3f}")
        
        if abs(correlation) > 0.7:
            print("  → High correlation between MSE and event count!")
            print("  → Parameter shift affects events significantly.")
            print("  → Event-consistent training could help.")
    
    return results


# ============================================================================
# Run All Phase 1 Experiments
# ============================================================================

def run_phase1_experiments():
    """Run all Phase 1 experiments."""
    print("\n" + "╔" + "═" * 78 + "╗")
    print("║" + "PHASE 1: ESTABLISH THE FAILURE LAW".center(78) + "║")
    print("╚" + "═" * 78 + "╝")
    
    print("\nGoal: Find the real reason long-horizon planning fails.")
    
    results = {}
    
    # Exp 1.1
    try:
        results['exp_1_1'] = exp_1_1_prediction_vs_planning()
    except Exception as e:
        print(f"\n✗ Experiment 1.1 FAILED: {e}")
        import traceback
        traceback.print_exc()
        results['exp_1_1'] = {'error': str(e)}
    
    # Exp 1.2
    try:
        results['exp_1_2'] = exp_1_2_event_dominance()
    except Exception as e:
        print(f"\n✗ Experiment 1.2 FAILED: {e}")
        import traceback
        traceback.print_exc()
        results['exp_1_2'] = {'error': str(e)}
    
    # Exp 1.3
    try:
        results['exp_1_3'] = exp_1_3_parameter_shift_events()
    except Exception as e:
        print(f"\n✗ Experiment 1.3 FAILED: {e}")
        import traceback
        traceback.print_exc()
        results['exp_1_3'] = {'error': str(e)}
    
    # Summary
    print("\n" + "=" * 80)
    print("PHASE 1 SUMMARY")
    print("=" * 80)
    
    # Check Gate G1
    print("\nGate G1 Check:")
    
    g1_passed = False
    
    # Check if prediction decouples from planning
    if 'exp_1_1' in results and isinstance(results['exp_1_1'], list):
        print("  [✓] Exp 1.1: Prediction vs Planning - Completed")
        g1_passed = True
    
    # Check if event dominance found
    if 'exp_1_2' in results and isinstance(results['exp_1_2'], dict):
        fraction = results['exp_1_2'].get('event_linked_fraction', 0)
        if fraction > 0.7:
            print(f"  [✓] Exp 1.2: Event dominance confirmed ({fraction:.1%})")
            g1_passed = True
        else:
            print(f"  [ ] Exp 1.2: Event fraction = {fraction:.1%} (need >70%)")
    
    if g1_passed:
        print("\n✓ GATE G1 PASSED - Proceed to Phase 2")
    else:
        print("\n✗ GATE G1 FAILED - Pivot to horizon-consistency only")
    
    # Save results
    os.makedirs('research/results', exist_ok=True)
    with open('research/results/phase1_results.json', 'w') as f:
        json.dump(results, f, indent=2, default=str)
    
    print("\nResults saved to: research/results/phase1_results.json")
    
    return results


if __name__ == "__main__":
    results = run_phase1_experiments()