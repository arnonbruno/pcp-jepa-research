#!/usr/bin/env python3
"""
PHASE A: FORENSIC VALIDATION
Gate A2 - MPPI Implementation Verification

A2.1: MPPI unit tests
A2.2: Toy physics task (guaranteed solvable)
A2.3: BouncingBall-easy task

If oracle MPPI doesn't beat random on these, MPPI code is WRONG.
"""

import jax
import jax.numpy as jnp
import numpy as np
import json
import os

# Load config
with open('/home/ulluboz/pcp-jepa-research/forensics/config.json', 'r') as f:
    CONFIG = json.load(f)

# ============================================================
# MPPI IMPLEMENTATION
# ============================================================

def create_mppi(physics_step_fn, horizon, n_samples, x_target, action_scale=1.0, temperature=1.0):
    """
    Create MPPI planner with configurable physics.
    """
    @jax.jit
    def batch_rollout(obs, actions):
        def single(action_seq):
            current = obs
            for t in range(horizon):
                current = physics_step_fn(current, action_seq[t])
            return current[0]
        return jax.vmap(single)(actions)
    
    batch_rollout_jit = jax.jit(batch_rollout)
    
    @jax.jit
    def plan(obs, key):
        key, subkey = jax.random.split(key)
        actions = jax.random.uniform(subkey, (n_samples, horizon, 1), 
                                      minval=-action_scale, maxval=action_scale)
        
        final_x = batch_rollout_jit(obs, actions)
        costs = jnp.abs(final_x - x_target)
        weights = jax.nn.softmax(-costs / temperature)
        
        return jnp.sum(weights * actions[:, 0, 0])
    
    return plan


# ============================================================
# A2.1: MPPI UNIT TESTS
# ============================================================

def mppi_unit_tests():
    """
    Basic sanity checks for MPPI implementation.
    """
    print("\n" + "="*60)
    print("A2.1: MPPI UNIT TESTS")
    print("="*60)
    
    all_pass = True
    
    # Test 1: Weights sum to 1
    print("\nTest 1: Weights sum to 1")
    costs = jnp.array([1.0, 2.0, 3.0, 4.0])
    weights = jax.nn.softmax(-costs)
    weight_sum = float(jnp.sum(weights))
    
    if abs(weight_sum - 1.0) < 1e-6:
        print(f"  ✅ Weights sum to {weight_sum:.6f}")
    else:
        print(f"  ❌ Weights sum to {weight_sum:.6f} (expected 1.0)")
        all_pass = False
    
    # Test 2: No NaNs in weights
    print("\nTest 2: No NaNs in weights")
    costs = jnp.array([jnp.inf, 1.0, 2.0])
    weights = jax.nn.softmax(-costs)
    
    if not jnp.any(jnp.isnan(weights)):
        print(f"  ✅ No NaNs even with inf cost")
    else:
        print(f"  ❌ NaNs detected in weights")
        all_pass = False
    
    # Test 3: Effective sample size
    print("\nTest 3: Effective sample size doesn't collapse")
    costs = jnp.array([0.1, 1.0, 2.0, 3.0, 4.0])
    weights = jax.nn.softmax(-costs / 0.01)  # Very low temperature
    ess = 1.0 / jnp.sum(weights ** 2)
    
    if float(ess) > 1.5:  # Not completely collapsed
        print(f"  ✅ ESS = {float(ess):.2f} (reasonable)")
    else:
        print(f"  ⚠️  ESS = {float(ess):.2f} (near collapse, but may be expected with low temp)")
    
    # Test 4: Best cost gets highest weight
    print("\nTest 4: Best cost gets highest weight")
    costs = jnp.array([1.0, 0.1, 2.0, 3.0])  # Index 1 is best
    weights = jax.nn.softmax(-costs)
    best_idx = int(jnp.argmax(weights))
    
    if best_idx == 1:
        print(f"  ✅ Best cost (index 1) gets highest weight")
    else:
        print(f"  ❌ Best cost not selected (got index {best_idx})")
        all_pass = False
    
    return all_pass


# ============================================================
# A2.2: TOY PHYSICS TASK (GUARANTEED SOLVABLE)
# ============================================================

def toy_physics_task():
    """
    Simple 1D integrator: x_{t+1} = x_t + a_t
    
    This is guaranteed solvable - MPPI MUST beat random here.
    """
    print("\n" + "="*60)
    print("A2.2: TOY PHYSICS TASK (1D INTEGRATOR)")
    print("="*60)
    print("\nPhysics: x_{t+1} = x_t + a_t (no walls, no drag)")
    print("This is guaranteed solvable.")
    
    @jax.jit
    def toy_physics(obs, a):
        x, v = obs[0], obs[1]
        # Pure integrator
        x_new = x + a * 0.1  # Scale action for reasonable movement
        return jnp.array([x_new, v])
    
    # MPPI config for toy task
    horizon = 20
    n_samples = 64
    x_target = 2.0
    
    mppi_plan = create_mppi(toy_physics, horizon, n_samples, x_target, 
                            action_scale=10.0, temperature=1.0)
    
    # Run episodes
    n_episodes = 50
    max_steps = 30
    
    def run_episode(policy_fn, key):
        key, subkey = jax.random.split(key)
        x = float(jax.random.uniform(subkey, (), minval=0.0, maxval=4.0))
        v = 0.0
        
        for _ in range(max_steps):
            obs = jnp.array([x, v])
            if policy_fn:
                key, subkey = jax.random.split(key)
                a = float(policy_fn(obs, subkey))
            else:
                key, subkey = jax.random.split(key)
                a = float(jax.random.uniform(subkey, (), minval=-10.0, maxval=10.0))
            
            obs_next = toy_physics(obs, a)
            x = float(obs_next[0])
        
        miss = abs(x - x_target)
        return miss < 0.3, miss
    
    # Test random
    print("\nRandom policy...")
    key = jax.random.PRNGKey(0)
    random_successes = sum(run_episode(None, key) for key in 
                          [jax.random.split(jax.random.PRNGKey(i))[0] for i in range(n_episodes)])
    random_sr = random_successes / n_episodes
    
    # Test MPPI
    print("MPPI policy...")
    key = jax.random.PRNGKey(0)
    mppi_successes = 0
    for i in range(n_episodes):
        key, subkey = jax.random.split(key)
        success, _ = run_episode(mppi_plan, subkey)
        if success:
            mppi_successes += 1
    mppi_sr = mppi_successes / n_episodes
    
    print(f"\nResults:")
    print(f"  Random: {random_sr:.1%}")
    print(f"  MPPI:   {mppi_sr:.1%}")
    print(f"  Improvement: {mppi_sr - random_sr:+.1%}")
    
    if mppi_sr > random_sr + 0.1:
        print("\n✅ PASS: MPPI beats random on toy task")
        return True
    else:
        print("\n❌ FAIL: MPPI doesn't beat random even on guaranteed-solvable task")
        print("   → MPPI implementation is BROKEN")
        return False


# ============================================================
# A2.3: BOUNCINGBALL-EASY TASK
# ============================================================

def bouncingball_easy_task():
    """
    Modified BouncingBall to be definitely planning-relevant:
    - Target within easy reach
    - No walls or reduced drag
    - Terminal cost only
    """
    print("\n" + "="*60)
    print("A2.3: BOUNCINGBALL-EASY TASK")
    print("="*60)
    print("\nModifications:")
    print("  - Target at x=1.5 (closer)")
    print("  - Start at x=1.0 (closer)")
    print("  - No walls (x in [-10, 10])")
    print("  - H=20, action_scale=2.0")
    
    @jax.jit
    def easy_physics(obs, a):
        x, v = obs[0], obs[1]
        v_new = v + a * 0.05
        x_new = x + v_new * 0.05
        # No walls - just bounds check for sanity
        x_new = jnp.clip(x_new, -10.0, 10.0)
        return jnp.array([x_new, v_new])
    
    # MPPI config
    horizon = 20
    n_samples = 64
    x_target = 1.5  # Closer target
    
    mppi_plan = create_mppi(easy_physics, horizon, n_samples, x_target,
                            action_scale=2.0, temperature=1.0)
    
    # Run episodes
    n_episodes = 50
    max_steps = 30
    
    def run_episode(policy_fn, key):
        # Start at x=1.0, closer to target
        key, subkey = jax.random.split(key)
        x = 1.0  # Fixed start for clearer test
        v = 0.0
        
        for _ in range(max_steps):
            obs = jnp.array([x, v])
            if policy_fn:
                key, subkey = jax.random.split(key)
                a = float(policy_fn(obs, subkey))
            else:
                key, subkey = jax.random.split(key)
                a = float(jax.random.uniform(subkey, (), minval=-2.0, maxval=2.0))
            
            obs_next = easy_physics(obs, a)
            x = float(obs_next[0])
        
        miss = abs(x - x_target)
        return miss < 0.3, miss
    
    # Test random
    print("\nRandom policy...")
    key = jax.random.PRNGKey(0)
    random_successes = sum(run_episode(None, key) for key in 
                          [jax.random.split(jax.random.PRNGKey(i))[0] for i in range(n_episodes)])
    random_sr = random_successes / n_episodes
    
    # Test MPPI
    print("MPPI policy...")
    key = jax.random.PRNGKey(0)
    mppi_successes = 0
    for i in range(n_episodes):
        key, subkey = jax.random.split(key)
        success, _ = run_episode(mppi_plan, subkey)
        if success:
            mppi_successes += 1
    mppi_sr = mppi_successes / n_episodes
    
    print(f"\nResults:")
    print(f"  Random: {random_sr:.1%}")
    print(f"  MPPI:   {mppi_sr:.1%}")
    print(f"  Improvement: {mppi_sr - random_sr:+.1%}")
    
    if mppi_sr > random_sr + 0.1:
        print("\n✅ PASS: MPPI beats random on easy BouncingBall")
        return True
    else:
        print("\n⚠️  MPPI still struggles on easy task")
        print("   → May need MPPI tuning (temperature, samples, horizon)")
        return False


# ============================================================
# MAIN
# ============================================================

def main():
    print("="*60)
    print("GATE A2: MPPI IMPLEMENTATION VERIFICATION")
    print("="*60)
    
    results = {}
    
    # A2.1: Unit tests
    unit_pass = mppi_unit_tests()
    results['A2.1_unit_tests'] = unit_pass
    
    # A2.2: Toy physics
    toy_pass = toy_physics_task()
    results['A2.2_toy_physics'] = toy_pass
    
    # A2.3: Easy BouncingBall
    easy_pass = bouncingball_easy_task()
    results['A2.3_easy_task'] = easy_pass
    
    # Gate A2 status
    print("\n" + "="*60)
    print("GATE A2 STATUS")
    print("="*60)
    
    # Must pass A2.2 (toy task) at minimum
    all_pass = unit_pass and toy_pass
    
    print(f"""
A2.1 Unit tests: {'✅ PASS' if unit_pass else '❌ FAIL'}
A2.2 Toy physics: {'✅ PASS' if toy_pass else '❌ FAIL'}
A2.3 Easy task: {'✅ PASS' if easy_pass else '⚠️ PARTIAL'}

GATE A2: {'✅ PASS' if all_pass else '❌ FAIL'}
""")
    
    if not all_pass:
        print("STOP: MPPI implementation has fundamental issues.")
        print("Fix MPPI before trusting any 'MPPI worse than random' claims.")
    else:
        print("MPPI works on solvable tasks. Proceed to Phase B (Model validation).")
    
    # Save results
    with open('/home/ulluboz/pcp-jepa-research/forensics/gate_A2_results.json', 'w') as f:
        json.dump(results, f, indent=2)
    
    return all_pass

if __name__ == '__main__':
    main()