import numpy as np
import pytest
import gymnasium as gym
import src.envs.contact_dropout as contact_dropout_module
from src.envs.contact_dropout import ContactDropoutEnv

def test_contact_dropout_env(monkeypatch):
    # Mock the MuJoCo contact API so the wrapper can be tested without
    # requiring a real MuJoCo rollout.
    class MockContact:
        def __init__(self, geom1=-1, geom2=-1, dist=1.0):
            self.geom1 = geom1
            self.geom2 = geom2
            self.dist = dist

    class MockEnv(gym.Env):
        def __init__(self):
            self.observation_space = gym.spaces.Box(low=-1, high=1, shape=(11,))
            self.action_space = gym.spaces.Box(low=-1, high=1, shape=(3,))
            self.state = np.zeros(11)
            self.step_idx = 0
            self.model = type(
                "Model",
                (),
                {
                    "geom_bodyid": np.array([0, 1, 2], dtype=int),
                    "geom_id2name": lambda self, idx: f"geom_{idx}",
                },
            )()

            class Data:
                def __init__(self):
                    self.cfrc_ext = np.zeros((3, 6))
                    self.ncon = 0
                    self.contact = [MockContact()]

            self.data = Data()

        def reset(self, seed=None):
            self.state = np.zeros(11)
            self.step_idx = 0
            self.data.cfrc_ext = np.zeros((3, 6))
            self.data.ncon = 0
            self.data.contact = [MockContact()]
            return self.state, {}

        def step(self, action):
            self.step_idx += 1
            self.state = self.state + np.ones(11) * 0.05
            self.data.cfrc_ext = np.zeros((3, 6))
            self.data.ncon = 0
            self.data.contact = [MockContact()]
            if self.step_idx == 12:
                self.data.cfrc_ext[1, 3] = 80.0
                self.data.ncon = 1
                self.data.contact = [MockContact(0, 1, -0.002)]
            return self.state.copy(), 0.0, False, False, {}

    original_make = gym.make

    def mock_make(env_id):
        return MockEnv()

    class MockMujoco:
        @staticmethod
        def mj_contactForce(model, data, idx, wrench):
            contact = data.contact[idx]
            if contact.dist <= 0:
                wrench[:] = np.array([80.0, 5.0, 0.0, 0.0, 0.0, 0.0])
            else:
                wrench[:] = 0.0

    monkeypatch.setattr(contact_dropout_module, "mujoco", MockMujoco)
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
        assert info['contact_source'] == 'mujoco.mj_contactForce'
        assert info['contact_pair_count'] == 1
        assert info['contact_distance_min'] < 0.0
        assert info['contact_normal_force_max'] > 0.0
        assert info['contact_pairs'][0]['geom1_name'] == 'geom_0'
        assert info['contact_pairs'][0]['geom2_name'] == 'geom_1'

        obs2, reward, term, trunc, info2 = env.step(np.zeros(3))
        assert info2['dropout_active']
        assert info2['dropout_step'] == 1
        assert np.array_equal(obs2, obs)

        obs3, reward, term, trunc, info3 = env.step(np.zeros(3))
        assert not info3['dropout_active']
        assert not np.array_equal(obs3, obs)
    finally:
        gym.make = original_make
