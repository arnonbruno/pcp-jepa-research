#!/usr/bin/env python3
"""
Test different horizons to find minimum needed for task.
"""

import jax
import jax.numpy as jnp
import numpy as np

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

def max_distance(horizon):
    """Calculate max distance reachable with H steps."""
    # From rest, with max acceleration
    x, v = 1.0, 0.0  # Start at center
    for _ in range(horizon):
        x, v = physics_step(jnp.array([x, v]), 1.0)
        x, v = float(x), float(v)
    return abs(x - 1.0)

def main():
    print("="*60)
    print("HORIZON ANALYSIS")
    print("="*60)
    print("\nTask: Reach x=2.0 from x=0.5 (distance = 1.5)")
    print("\nHorizon | Max Reachable Distance | Can Reach Target?")
    print("-" * 55)
    
    for h in [5, 10, 15, 20, 30, 40, 50, 60, 80, 100]:
        max_dist = max_distance(h)
        can_reach = "✅" if max_dist >= 1.5 else "❌"
        print(f"  {h:3d}    |        {max_dist:.2f}             | {can_reach}")
    
    print("\n" + "="*60)
    print("CONCLUSION")
    print("="*60)
    print("""
To reach target from farthest positions (x=0.5 or x=3.5),
you need H >= 30-40 steps minimum.

MPPI with H=10 cannot plan successfully because it can't
even REACH the target in its planning horizon!

FIX: Use H >= 40 for this task.
""")


if __name__ == '__main__':
    main()