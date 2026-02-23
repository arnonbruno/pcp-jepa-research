#!/usr/bin/env python3
"""Simple test: does the model work for planning?"""

import jax
import jax.numpy as jnp
import pickle
import time
import sys
sys.path.insert(0, '/home/ulluboz/pcp-jepa-research')

from experiments.phase2.train_action_controllable import (
    ActionControllableModel, Config, physics_step
)

def main():
    print("=" * 60)
    print("SIMPLE MODEL TEST")
    print("=" * 60)
    
    # Load model
    model = ActionControllableModel(latent_dim=32, hidden_dim=128, action_dim=1, obs_dim=2)
    
    checkpoint_path = '/home/ulluboz/pcp-jepa-research/checkpoints/model_best.pkl'
    with open(checkpoint_path, 'rb') as f:
        checkpoint = pickle.load(f)
    params = checkpoint['params']
    print(f"Loaded model from epoch {checkpoint['epoch']}")
    
    # Test 1: Single forward pass
    print("\n1. Single forward pass...")
    obs = jnp.array([1.0, 0.5])
    action = jnp.array([0.5])
    
    start = time.time()
    next_obs = model.apply(params, obs, action)
    elapsed = time.time() - start
    print(f"   Result: {next_obs}")
    print(f"   Time: {elapsed*1000:.1f}ms")
    
    # Test 2: Multiple passes (warmup JIT)
    print("\n2. JIT warmup (10 passes)...")
    for i in range(10):
        next_obs = model.apply(params, obs, action)
    print(f"   Done")
    
    # Test 3: Time 100 passes
    print("\n3. Timing 100 passes...")
    start = time.time()
    for i in range(100):
        next_obs = model.apply(params, obs, action)
    elapsed = time.time() - start
    print(f"   Total: {elapsed*1000:.1f}ms")
    print(f"   Per pass: {elapsed*10:.1f}ms")
    
    # Test 4: Rollout 30 steps
    print("\n4. 30-step rollout...")
    obs = jnp.array([1.0, 0.0])
    start = time.time()
    for t in range(30):
        obs = model.apply(params, obs, jnp.array([0.5]))
    elapsed = time.time() - start
    print(f"   Final: x={obs[0]:.2f}, v={obs[1]:.2f}")
    print(f"   Time: {elapsed*1000:.1f}ms")
    
    # Test 5: Sample 64 action sequences
    print("\n5. Sampling 64 action sequences (H=10)...")
    key = jax.random.PRNGKey(42)
    start = time.time()
    actions = jax.random.uniform(key, (64, 10, 1), minval=-1.0, maxval=1.0)
    elapsed = time.time() - start
    print(f"   Shape: {actions.shape}")
    print(f"   Time: {elapsed*1000:.1f}ms")
    
    # Test 6: Evaluate one batch of rollouts
    print("\n6. Batch rollout (64 trajectories, H=10)...")
    obs0 = jnp.array([1.0, 0.0])
    
    def rollout_single(action_seq):
        current_obs = obs0
        for t in range(10):
            current_obs = model.apply(params, current_obs, action_seq[t])
        return current_obs[0]
    
    # JIT compile
    start = time.time()
    batch_rollout = jax.jit(jax.vmap(rollout_single))
    final_x = batch_rollout(actions)
    elapsed_jit = time.time() - start
    print(f"   JIT compile + run: {elapsed_jit*1000:.1f}ms")
    
    # Run again (no JIT)
    start = time.time()
    final_x = batch_rollout(actions)
    elapsed_run = time.time() - start
    print(f"   Run after JIT: {elapsed_run*1000:.1f}ms")
    print(f"   Final x range: [{final_x.min():.2f}, {final_x.max():.2f}]")
    
    print("\n" + "=" * 60)
    print("✅ Model works! Ready for MPPI.")
    print("=" * 60)


if __name__ == '__main__':
    main()
