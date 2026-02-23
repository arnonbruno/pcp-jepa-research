#!/usr/bin/env python3
"""
Debug: Why does MPPI fail even with perfect model?
"""

import jax
import jax.numpy as jnp
import pickle
import numpy as np
import sys
sys.path.insert(0, '/home/ulluboz/pcp-jepa-research')

from experiments.phase2.train_action_controllable import physics_step
from experiments.phase2.hybrid_simple import ResidualModel, pure_physics_step

def main():
    print("="*60)
    print("DEBUG: MPPI BEHAVIOR")
    print("="*60)
    
    # Load or train model
    model = ResidualModel(hidden_dim=64)
    key = jax.random.PRNGKey(42)
    params = model.init(key, jnp.zeros(2), jnp.zeros(1))
    
    # Quick train
    import optax
    optimizer = optax.adam(1e-3)
    opt_state = optimizer.init(params)
    
    # Generate data
    obs_list, action_list, next_obs_list = [], [], []
    for _ in range(5000):
        key, subkey = jax.random.split(key)
        x0 = float(jax.random.uniform(subkey, (), 0.5, 3.5))
        key, subkey = jax.random.split(key)
        v0 = float(jax.random.uniform(subkey, (), -0.5, 0.5))
        key, subkey = jax.random.split(key)
        a = float(jax.random.uniform(subkey, (), -1.0, 1.0))
        x1, v1 = physics_step(x0, v0, a)
        obs_list.append([x0, v0])
        action_list.append([a])
        next_obs_list.append([x1, v1])
    
    obs = jnp.array(obs_list)
    actions = jnp.array(action_list)
    next_obs = jnp.array(next_obs_list)
    
    @jax.jit
    def loss_fn(params):
        pred = jax.vmap(lambda o, a: model.apply(params, o, a))(obs, actions)
        return jnp.mean((pred - next_obs) ** 2)
    
    for _ in range(50):
        loss, grads = jax.value_and_grad(loss_fn)(params)
        updates, opt_state = optimizer.update(grads, opt_state)
        params = optax.apply_updates(params, updates)
    
    print(f"Model trained (loss={float(loss):.6f})")
    
    # Test 1: Does model agree with physics?
    print("\n" + "="*60)
    print("TEST 1: Model vs Physics Agreement")
    print("="*60)
    
    test_cases = [
        (1.0, 0.0, 0.5),   # Center, stationary, push right
        (0.5, 0.0, -0.3),  # Near wall, push toward wall
        (3.5, 0.0, 0.3),   # Near right wall, push toward wall
    ]
    
    for x, v, a in test_cases:
        obs_in = jnp.array([x, v])
        action = jnp.array([a])
        
        pred = model.apply(params, obs_in, action)
        x_real, v_real = physics_step(x, v, a)
        
        print(f"\n  Input: x={x:.2f}, v={v:.2f}, a={a:.2f}")
        print(f"  Model: x={float(pred[0]):.3f}, v={float(pred[1]):.3f}")
        print(f"  Real:  x={x_real:.3f}, v={v_real:.3f}")
        print(f"  Gap:   Δx={abs(float(pred[0]) - x_real):.4f}")
    
    # Test 2: What actions does MPPI choose?
    print("\n" + "="*60)
    print("TEST 2: MPPI Action Selection")
    print("="*60)
    
    def mppi_plan(obs, n_samples=64, horizon=10):
        key = jax.random.PRNGKey(0)
        key, subkey = jax.random.split(key)
        actions = jax.random.uniform(subkey, (n_samples, horizon, 1), minval=-1.0, maxval=1.0)
        
        def rollout(action_seq):
            current = obs
            for a in action_seq:
                current = model.apply(params, current, a[None])
            return current[0]
        
        final_x = jax.vmap(rollout)(actions)
        costs = jnp.abs(final_x - 2.0)
        weights = jax.nn.softmax(-costs)
        
        best_action = jnp.sum(weights[:, None] * actions[:, 0, :], axis=0)
        return float(best_action[0]), weights, final_x
    
    test_states = [
        (1.0, 0.0),  # Left of target
        (3.0, 0.0),  # Right of target
        (2.0, 0.5),  # At target, moving right
    ]
    
    for x, v in test_states:
        obs_in = jnp.array([x, v])
        best_a, weights, final_x = mppi_plan(obs_in)
        
        print(f"\n  State: x={x:.2f}, v={v:.2f}")
        print(f"  MPPI chooses: a={best_a:.3f}")
        print(f"  Expected: a={'positive' if x < 2.0 else 'negative' if x > 2.0 else 'zero'}")
        print(f"  Predicted final_x range: [{float(jnp.min(final_x)):.2f}, {float(jnp.max(final_x)):.2f}]")
    
    # Test 3: Trajectory following
    print("\n" + "="*60)
    print("TEST 3: Full Episode Rollout")
    print("="*60)
    
    # Starting state
    x, v = 1.0, 0.0
    target = 2.0
    
    print(f"\nStarting at x={x:.2f}, target={target:.2f}")
    print(f"{'Step':<6} {'Action':<10} {'Model Pred':<15} {'Real Physics':<15} {'Gap'}")
    print("-" * 60)
    
    for step in range(10):
        obs_in = jnp.array([x, v])
        action = jnp.array([0.5])  # Constant push right
        
        # Model prediction
        pred = model.apply(params, obs_in, action)
        
        # Real physics
        x_real, v_real = physics_step(x, v, 0.5)
        
        gap = abs(float(pred[0]) - x_real)
        print(f"{step:<6} {0.5:<10.2f} x={float(pred[0]):.3f}, v={float(pred[1]):.3f}  x={x_real:.3f}, v={v_real:.3f}  {gap:.4f}")
        
        x, v = x_real, v_real
    
    # Test 4: Why does random work better?
    print("\n" + "="*60)
    print("TEST 4: Random vs MPPI Comparison")
    print("="*60)
    
    n_trials = 100
    random_successes = 0
    mppi_successes = 0
    
    for trial in range(n_trials):
        key, subkey = jax.random.split(key)
        x0 = float(jax.random.uniform(subkey, (), 0.5, 3.5))
        key, subkey = jax.random.split(key)
        v0 = float(jax.random.uniform(subkey, (), -0.5, maxval=0.5))
        
        # Random
        x, v = x0, v0
        for _ in range(30):
            key, subkey = jax.random.split(key)
            a = float(jax.random.uniform(subkey, (), -1.0, 1.0))
            x, v = physics_step(x, v, a)
        if abs(x - 2.0) < 0.3:
            random_successes += 1
        
        # MPPI
        x, v = x0, v0
        for _ in range(30):
            obs_in = jnp.array([x, v])
            a, _, _ = mppi_plan(obs_in)
            x, v = physics_step(x, v, a)
        if abs(x - 2.0) < 0.3:
            mppi_successes += 1
    
    print(f"\nRandom success rate: {random_successes/n_trials:.1%}")
    print(f"MPPI success rate:   {mppi_successes/n_trials:.1%}")
    
    print("\n" + "="*60)
    print("HYPOTHESIS")
    print("="*60)
    print("""
If model is perfect but MPPI still fails, possible causes:
1. Task doesn't benefit from planning (random exploration is sufficient)
2. MPPI horizon is too short (H=10, need longer for this task)
3. MPPI is greedy - doesn't explore enough
4. The cost function (|x - target|) doesn't capture the right objective

Next: Test with longer horizon and different cost functions.
""")


if __name__ == '__main__':
    main()