#!/usr/bin/env python3
"""
MPPI Debug: Check what's happening inside the planner
"""

import jax
import jax.numpy as jnp
import numpy as np

HORIZON = 10
N_SAMPLES = 64
X_TARGET = 2.0

@jax.jit
def physics_step(obs, a):
    x, v = obs[0], obs[1]
    v_new = v + a * 0.05
    x_new = x + v_new * 0.05
    x_new = jnp.where(x_new < 0, -x_new, x_new)
    v_new = jnp.where(x_new < 0, -v_new * 0.8, v_new)
    x_new = jnp.where(x_new > 4, 8 - x_new, x_new)
    v_new = jnp.where(x_new > 4, -v_new * 0.8, v_new)
    return jnp.array([x_new, v_new])

@jax.jit
def rollout(obs, actions):
    current = obs
    for t in range(HORIZON):
        current = physics_step(current, actions[t])
    return current[0]

def main():
    print("="*60)
    print("MPPI INTERNALS DEBUG")
    print("="*60)
    
    key = jax.random.PRNGKey(42)
    
    # Test from different starting positions
    test_positions = [0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5]
    
    for x0 in test_positions:
        obs = jnp.array([x0, 0.0])
        
        # Sample trajectories
        key, subkey = jax.random.split(key)
        actions = jax.random.uniform(subkey, (N_SAMPLES, HORIZON, 1), minval=-1.0, maxval=1.0)
        
        # Compute final positions
        final_xs = jax.vmap(lambda a: rollout(obs, a))(actions)
        
        # Compute costs
        costs = jnp.abs(final_xs - X_TARGET)
        
        # Compute weights
        weights = jax.nn.softmax(-costs)
        
        # Best action
        best_action = jnp.sum(weights * actions[:, 0, 0])
        
        # Stats
        print(f"\nStart x={x0:.1f}:")
        print(f"  Final x range: [{float(jnp.min(final_xs)):.2f}, {float(jnp.max(final_xs)):.2f}]")
        print(f"  Cost range: [{float(jnp.min(costs)):.2f}, {float(jnp.max(costs)):.2f}]")
        print(f"  Weight distribution: max={float(jnp.max(weights)):.3f}, min={float(jnp.min(weights)):.3f}")
        print(f"  Best action: {float(best_action):.3f}")
        print(f"  Expected: {'positive (push right)' if x0 < X_TARGET else 'negative (push left)' if x0 > X_TARGET else 'zero'}")
        
        # Check if best action is correct direction
        correct = (x0 < X_TARGET and best_action > 0) or (x0 > X_TARGET and best_action < 0) or (abs(x0 - X_TARGET) < 0.1)
        print(f"  Direction: {'✅' if correct else '❌ WRONG!'}")
        
        # Show weight distribution
        sorted_weights = np.sort(weights)[::-1]
        print(f"  Top 3 weights: {sorted_weights[:3]}")
        print(f"  Sum of top 3: {sum(sorted_weights[:3]):.3f}")
    
    print("\n" + "="*60)
    print("DIAGNOSIS")
    print("="*60)
    print("""
If weights are collapsed (one weight ~1.0):
  → Temperature too low, increase lambda
  
If best action is wrong direction:
  → Horizon too short, can't reach target
  → Need to consider intermediate states

If all directions give similar costs:
  → Task doesn't have gradient toward target
  → Need different cost function
""")


if __name__ == '__main__':
    main()