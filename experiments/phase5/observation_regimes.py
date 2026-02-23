#!/usr/bin/env python3
"""
PHASE 5: Observation Regimes + Dataset Generation

Implements:
- O0: Full state [x, v]
- O1: Partial state - position only + noise/dropout
- O2: Pixels - 64x64 grayscale render

With event labels:
- contact flag
- time-to-impact
- restitution value
"""

import numpy as np
import os


class BouncingBallGravity:
    """Bouncing ball with configurable restitution."""
    
    def __init__(self, tau=0.3, restitution=0.8):
        self.g = 9.81
        self.e = restitution
        self.dt = 0.05
        self.x_target = 2.0
        self.tau = tau
        self.x_bounds = (0, 3)
        
    def reset(self, seed=None):
        if seed is not None:
            np.random.seed(seed)
        start_positions = [0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5]
        self.x = start_positions[seed % len(start_positions)]
        self.v = 0.0
        self.bounce_count = 0
        self.steps_since_bounce = 0
        return np.array([self.x, self.v])
    
    def step(self, a):
        a = np.clip(a, -2.0, 2.0)
        self.v += (-self.g + a) * self.dt
        self.x += self.v * self.dt
        
        bounced = False
        if self.x < self.x_bounds[0]:
            self.x = -self.x * self.e
            self.v = -self.v * self.e
            bounced = True
            self.bounce_count += 1
            self.steps_since_bounce = 0
        elif self.x > self.x_bounds[1]:
            self.x = self.x_bounds[1] - (self.x - self.x_bounds[1]) * self.e
            self.v = -self.v * self.e
            bounced = True
            self.bounce_count += 1
            self.steps_since_bounce = 0
        else:
            self.steps_since_bounce += 1
            
        return np.array([self.x, self.v]), bounced


class ObservationModel:
    """Observation regimes for the ball."""
    
    def __init__(self, regime='full', noise_std=0.0, dropout_prob=0.0):
        """
        regime: 'full', 'partial', 'pixels'
        """
        self.regime = regime
        self.noise_std = noise_std
        self.dropout_prob = dropout_prob
    
    def observe(self, state):
        """Convert true state to observation."""
        x, v = state
        
        if self.regime == 'full':
            # Full state: [x, v]
            obs = np.array([x, v])
            
        elif self.regime == 'partial':
            # Partial: position only + noise + dropout
            obs = np.array([x])
            
        elif self.regime == 'pixels':
            # 64x64 grayscale render
            obs = self._render(x, v)
        
        # Add noise
        if self.noise_std > 0:
            obs = obs + np.random.randn(*obs.shape) * self.noise_std
        
        # Apply dropout
        if self.dropout_prob > 0 and np.random.rand() < self.dropout_prob:
            obs = np.zeros_like(obs)
        
        return obs
    
    def _render(self, x, v):
        """Render 64x64 grayscale image."""
        # Simple render: ball position mapped to image
        img = np.zeros((64, 64), dtype=np.float32)
        
        # Map x ∈ [0, 3] to pixels
        px = int(x / 3.0 * 63)
        py = 32  # centered vertically
        
        # Draw ball (circle)
        for dy in range(-4, 5):
            for dx in range(-4, 5):
                if dx*dx + dy*dy <= 16:
                    ny, nx = py + dy, px + dx
                    if 0 <= ny < 64 and 0 <= nx < 64:
                        img[ny, nx] = 1.0
        
        # Add velocity indicator (small dot)
        pv = int((v + 5) / 10.0 * 63)  # v ∈ [-5, 5]
        pv = np.clip(pv, 0, 63)
        img[10, pv] = 0.5
        
        return img.flatten()


def generate_trajectories(
    n_episodes=1000,
    horizon=30,
    restitution=0.8,
    observation='full',
    policy='random',
    seed=0
):
    """
    Generate trajectory dataset with labels.
    
    Returns:
        dict with keys: observations, states, actions, contacts, times_to_impact, restitions
    """
    np.random.seed(seed)
    
    obs_model = ObservationModel(regime=observation)
    env = BouncingBallGravity(restitution=restitution)
    
    data = {
        'observations': [],
        'states': [],
        'actions': [],
        'contacts': [],  # 1 if contact just occurred
        'times_to_impact': [],  # steps until next impact (0 if currently impacting)
        'restitutions': [],
    }
    
    for ep in range(n_episodes):
        obs = env.reset(seed=ep + seed)
        x, v = obs[0], obs[1]
        
        for step in range(horizon):
            # Get observation
            observation = obs_model.observe(np.array([x, v]))
            
            # Determine action
            if policy == 'random':
                a = np.random.uniform(-2.0, 2.0)
            elif policy == 'pd':
                a = 1.5 * (2.0 - x) + (-2.0) * (-v)
                a = np.clip(a, -2.0, 2.0)
            else:
                a = 0.0
            
            # Step
            next_obs, bounced = env.step(a)
            x_next, v_next = next_obs[0], next_obs[1]
            
            # Labels
            # Time to impact: count steps until next bounce
            time_to_impact = 0
            if bounced:
                time_to_impact = 0
                contact = 1
            else:
                # Look ahead to find next bounce
                x_look, v_look = x_next, v_next
                tti = 0
                for look_step in range(1, 20):
                    v_look += (-9.81 + a) * env.dt
                    x_look += v_look * env.dt
                    if x_look < 0 or x_look > 3:
                        tti = look_step
                        break
                time_to_impact = tti
                contact = 0
            
            # Store
            data['observations'].append(observation)
            data['states'].append([x, v])
            data['actions'].append(a)
            data['contacts'].append(contact)
            data['times_to_impact'].append(time_to_impact)
            data['restitutions'].append(restitution)
            
            x, v = x_next, v_next
    
    # Convert to arrays
    for key in data:
        data[key] = np.array(data[key])
    
    return data


def generate_dataset_split(
    n_train=500,
    n_val=250,
    n_test=250,
    observation='full',
    policy='random',
    restitution_values=[0.3, 0.5, 0.7, 0.85],
    seed=0
):
    """Generate train/val/test split across restitution values."""
    
    all_data = {'train': [], 'val': [], 'test': []}
    
    for rest in restitution_values:
        # Split seeds
        train_data = generate_trajectories(
            n_episodes=n_train,
            restitution=rest,
            observation=observation,
            policy=policy,
            seed=seed + rest * 1000
        )
        val_data = generate_trajectories(
            n_episodes=n_val,
            restitution=rest,
            observation=observation,
            policy=policy,
            seed=seed + rest * 2000
        )
        test_data = generate_trajectories(
            n_episodes=n_test,
            restitution=rest,
            observation=observation,
            policy=policy,
            seed=seed + rest * 3000
        )
        
        all_data['train'].append(train_data)
        all_data['val'].append(val_data)
        all_data['test'].append(test_data)
    
    return all_data


def test_observation_regimes():
    """Test that observation regimes work correctly."""
    print("="*70)
    print("Testing Observation Regimes")
    print("="*70)
    
    for regime in ['full', 'partial', 'pixels']:
        obs_model = ObservationModel(regime=regime)
        env = BouncingBallGravity(restitution=0.8)
        obs = env.reset(seed=0)
        
        observation = obs_model.observe(obs)
        
        print(f"\n{regime}:")
        print(f"  True state: {obs}")
        print(f"  Observation shape: {observation.shape}")
        print(f"  Observation sample: {observation[:5] if len(observation) > 5 else observation}")
    
    # Test trajectory generation
    print("\n" + "="*70)
    print("Testing Trajectory Generation")
    print("="*70)
    
    for obs in ['full', 'partial']:
        data = generate_trajectories(
            n_episodes=10,
            horizon=30,
            restitution=0.8,
            observation=obs,
            policy='pd',
            seed=0
        )
        
        n_contacts = sum(data['contacts'])
        print(f"\n{obs}:")
        print(f"  Episodes: {len(data['states']) // 30}")
        print(f"  Contacts: {n_contacts}")
        print(f"  States shape: {data['states'].shape}")
        print(f"  Obs shape: {data['observations'].shape}")


if __name__ == '__main__':
    test_observation_regimes()
