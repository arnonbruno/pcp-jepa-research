import numpy as np
import pytest
import gymnasium as gym
from src.envs.contact_dropout import ContactDropoutEnv

def test_contact_dropout_env():
    # Make sure Hopper-v4 or similar is installed. We can mock it if needed, or rely on gym.
    # If the system doesn't have mujoco, we might need a fallback.
    # Let's try to mock the inner environment to make it bulletproof without Mujoco.
    class MockEnv(gym.Env):
        def __init__(self):
            self.observation_space = gym.spaces.Box(low=-1, high=1, shape=(11,))
            self.action_space = gym.spaces.Box(low=-1, high=1, shape=(3,))
            self.state = np.zeros(11)
        def reset(self, seed=None):
            self.state = np.zeros(11)
            return self.state, {}
        def step(self, action):
            # Move state slightly
            self.state = self.state + np.ones(11) * 0.05
            return self.state, 0.0, False, False, {}

    # Patch the gym.make to return our mock
    original_make = gym.make
    def mock_make(env_id):
        return MockEnv()
    
    gym.make = mock_make
    try:
        env = ContactDropoutEnv(env_id="MockEnv", dropout_duration=2, velocity_threshold=0.1)
        obs, info = env.reset()
        assert obs.shape == (11,)
        
        # Take 11 steps to exceed step_count > 10
        for _ in range(11):
            obs, reward, term, trunc, info = env.step(np.zeros(3))
            
        assert not info['dropout_active']
        
        # Force a large state change
        env.env.state += np.ones(11) * 0.5
        obs, reward, term, trunc, info = env.step(np.zeros(3))
        
        assert info['dropout_active']
        assert info['dropout_step'] == 0
        # Obs should be frozen
        assert np.array_equal(obs, info['frozen_obs'])
        
        # Next step, still dropout
        obs2, reward, term, trunc, info2 = env.step(np.zeros(3))
        assert info2['dropout_active']
        assert info2['dropout_step'] == 1
        assert np.array_equal(obs2, obs)
        
        # Next step, dropout ends (prevent re-trigger by resetting state close to obs_prev)
        env.env.state = env.obs_prev.copy()
        obs3, reward, term, trunc, info3 = env.step(np.zeros(3))
        assert not info3['dropout_active']
        assert not np.array_equal(obs3, obs)
    finally:
        gym.make = original_make
