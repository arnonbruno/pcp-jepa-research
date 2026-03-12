import numpy as np
import pytest
import gymnasium as gym
from src.envs.contact_dropout import ContactDropoutEnv

def test_contact_dropout_env():
    # Mock the MuJoCo contact-force interface so the wrapper can be tested
    # without requiring a real MuJoCo rollout.
    class MockEnv(gym.Env):
        def __init__(self):
            self.observation_space = gym.spaces.Box(low=-1, high=1, shape=(11,))
            self.action_space = gym.spaces.Box(low=-1, high=1, shape=(3,))
            self.state = np.zeros(11)
            self.step_idx = 0

            class Data:
                def __init__(self):
                    self.cfrc_ext = np.zeros((3, 6))
                    self.ncon = 0

            self.data = Data()

        def reset(self, seed=None):
            self.state = np.zeros(11)
            self.step_idx = 0
            self.data.cfrc_ext = np.zeros((3, 6))
            self.data.ncon = 0
            return self.state, {}

        def step(self, action):
            self.step_idx += 1
            self.state = self.state + np.ones(11) * 0.05
            self.data.cfrc_ext = np.zeros((3, 6))
            self.data.ncon = 0
            if self.step_idx == 12:
                self.data.cfrc_ext[1, 3] = 80.0
                self.data.ncon = 1
            return self.state.copy(), 0.0, False, False, {}

    original_make = gym.make

    def mock_make(env_id):
        return MockEnv()

    gym.make = mock_make
    try:
        env = ContactDropoutEnv(env_id="MockEnv", dropout_duration=2, velocity_threshold=0.1)
        obs, info = env.reset()
        assert obs.shape == (11,)

        for _ in range(11):
            obs, reward, term, trunc, info = env.step(np.zeros(3))

        assert not info['dropout_active']

        obs, reward, term, trunc, info = env.step(np.zeros(3))

        assert info['dropout_active']
        assert info['dropout_step'] == 0
        assert np.array_equal(obs, info['frozen_obs'])
        assert info['contact_detected']
        assert info['contact_force_max'] >= info['contact_force_threshold']
        assert info['contact_source'] == 'data.cfrc_ext'

        obs2, reward, term, trunc, info2 = env.step(np.zeros(3))
        assert info2['dropout_active']
        assert info2['dropout_step'] == 1
        assert np.array_equal(obs2, obs)

        obs3, reward, term, trunc, info3 = env.step(np.zeros(3))
        assert not info3['dropout_active']
        assert not np.array_equal(obs3, obs)
    finally:
        gym.make = original_make
