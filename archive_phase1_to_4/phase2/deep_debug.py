#!/usr/bin/env python3
"""
Debug why H=30 MPPI still fails
"""

import jax
import jax.numpy as jnp
import numpy as np

HORIZON = 30
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
def batch_rollout(obs, actions):
    def single(action_seq):
        current = obs
        for t in range(HORIZON):
            current = physics_step(current, action_seq[t])
        return current[0]
    return jax.vmap(single)(actions)

def main():
    print("="*60)
    print("DEEP DEBUG: Why H=30 MPPI fails")
    print("="*60)
    
    key = jax.random.PRNGKey(42)
    
    # Test from x=1.0 (should be easier than x=0.5)
    for x0 in [0.5, 1.0, 1.5, 2.0]:
        obs = jnp.array([x0, 0.0])
        
        key, subkey = jax.random.split(key)
        actions = jax.random.uniform(subkey, (N_SAMPLES, HORIZON, 1), minval=-1.0, maxval=1.0)
        
        final_x = batch_rollout(obs, actions)
        costs = jnp.abs(final_x - X_TARGET)
        weights = jax.nn.softmax(-costs)
        best_action = jnp.sum(weights * actions[:, 0, 0])
        
        print(f"\nStart x={x0:.1f}:")
        print(f"  Final x range: [{float(jnp.min(final_x)):.2f}, {float(jnp.max(final_x)):.2f}]")
        print(f"  Target x={X_TARGET} is in range: {float(jnp.min(final_x)) <= X_TARGET <= float(jnp.max(final_x))}")
        print(f"  Cost range: [{float(jnp.min(costs)):.3f}, {float(jnp.max(costs)):.3f}]")
        print(f"  Weight max: {float(jnp.max(weights)):.4f}")
        print(f"  Best action: {float(best_action):.3f}")
        
        # Check if best action makes sense
        if x0 < X_TARGET:
            expected = "positive"
            correct = best_action > 0
        elif x0 > X_TARGET:
            expected = "negative"
            correct = best_action < 0
        else:
            expected = "any"
            correct = True
        
        print(f"  Expected: {expected}, Got: {'positive' if best_action > 0 else 'negative' if best_action < 0 else 'zero'}")
        print(f"  Direction: {'✅' if correct else '❌'}")
        
        # Show cost distribution
        print(f"  Cost distribution:")
        print(f"    Best trajectory: cost={float(jnp.min(costs)):.3f}, final_x={float(final_x[jnp.argmin(costs)]):.2f}")
        print(f"    Worst trajectory: cost={float(jnp.max(costs)):.3f}, final_x={float(final_x[jnp.argmax(costs)]):.2f}")

if __name__ == '__main__':
    main()