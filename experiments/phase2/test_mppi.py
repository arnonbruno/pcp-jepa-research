"""
Quick MPPI Test - Minimal Version

Just check if MPPI beats random with simple dynamics.
"""

import jax
import jax.numpy as jnp
import numpy as np

from src.environments import BouncingBall, BouncingBallParams


def run_mppi(env, state0, config, key):
    """Simple MPPI planner."""
    H = 20
    N = 100
    x_target = 2.0
    max_impulse = 1.0
    
    best_cost = float('inf')
    best_actions = None
    
    for _ in range(N):
        k, key = jax.random.split(key)
        actions = jax.random.uniform(k, (H,), minval=-max_impulse, maxval=max_impulse)
        
        # Simulate
        state = state0
        cost = 0.0
        for t in range(H):
            state = state.at[2].add(actions[t] * 0.1)
            state, _ = env.step(state)
            cost = cost + (state[0] - x_target) ** 2
        
        cost = cost + 10.0 * (state[0] - x_target) ** 2  # Terminal
        
        if cost < best_cost:
            best_cost = cost
            best_actions = actions
    
    return best_actions


def main():
    print("\nQUICK MPPI TEST")
    print("=" * 50)
    
    env = BouncingBall(BouncingBallParams(restitution=0.8))
    key = jax.random.PRNGKey(42)
    
    # Random baseline
    print("\nRandom actions...")
    random_success = 0
    for _ in range(20):
        key, k1, k2 = jax.random.split(key, 3)
        state = jnp.array([0.0, 
                          jax.random.uniform(k1, minval=1.0, maxval=3.0),
                          0.0,
                          jax.random.uniform(k2, minval=-2.0, maxval=2.0)])
        
        for _ in range(50):
            k, key = jax.random.split(key)
            a = jax.random.uniform(k, (), minval=-1, maxval=1)
            state = state.at[2].add(a)
            state, _ = env.step(state)
        
        if abs(state[0] - 2.0) < 0.5:
            random_success += 1
    
    print(f"  Success: {random_success/20:.1%}")
    
    # MPPI
    print("\nMPPI planning...")
    mppi_success = 0
    for _ in range(20):
        key, k1, k2 = jax.random.split(key, 3)
        state = jnp.array([0.0,
                          jax.random.uniform(k1, minval=1.0, maxval=3.0),
                          0.0,
                          jax.random.uniform(k2, minval=-2.0, maxval=2.0)])
        
        for _ in range(50):
            k, key = jax.random.split(key)
            actions = run_mppi(env, state, {}, k)
            state = state.at[2].add(actions[0])
            state, _ = env.step(state)
        
        if abs(state[0] - 2.0) < 0.5:
            mppi_success += 1
    
    print(f"  Success: {mppi_success/20:.1%}")
    
    print(f"\nRandom: {random_success/20:.1%}")
    print(f"MPPI:   {mppi_success/20:.1%}")
    
    if mppi_success > random_success:
        print("\n✓ MPPI beats random!")


if __name__ == "__main__":
    main()