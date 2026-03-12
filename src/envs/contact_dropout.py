import numpy as np
import gymnasium as gym

class ContactDropoutEnv:
    """
    Wrapper that triggers sensor dropout when large state changes are detected
    (proxy for contact events). During dropout, observations are frozen.
    """
    def __init__(self, env_id='Hopper-v4', dropout_duration=5, velocity_threshold=0.1):
        self.env = gym.make(env_id)
        self.dropout_duration = dropout_duration
        self.velocity_threshold = velocity_threshold
        self.obs_prev = None
        self.frozen_obs = None
        self.dropout_countdown = 0
        self.step_count = 0

    def reset(self, seed=None):
        obs, info = self.env.reset(seed=seed)
        self.obs_prev = obs.copy()
        self.frozen_obs = obs.copy()
        self.dropout_countdown = 0
        self.step_count = 0
        return obs, info

    def step(self, action):
        obs, reward, term, trunc, info = self.env.step(action)
        self.step_count += 1

        # Trigger dropout on large state changes (contact proxy)
        if self.obs_prev is not None and self.dropout_countdown == 0:
            delta = np.abs(obs - self.obs_prev).max()
            if delta > self.velocity_threshold and self.step_count > 10:
                self.dropout_countdown = self.dropout_duration
                self.frozen_obs = obs.copy()

        info['true_obs'] = obs.copy()
        info['dropout_active'] = self.dropout_countdown > 0
        info['dropout_step'] = self.dropout_duration - self.dropout_countdown
        info['frozen_obs'] = self.frozen_obs.copy()

        if self.dropout_countdown > 0:
            obs_return = self.frozen_obs.copy()
            self.dropout_countdown -= 1
        else:
            obs_return = obs.copy()
            self.obs_prev = obs.copy()

        return obs_return, reward, term, trunc, info

    @property
    def observation_space(self):
        return self.env.observation_space

    @property
    def action_space(self):
        return self.env.action_space

    def close(self):
        self.env.close()

# Alias for compatibility with older code
CriticalDropoutEnv = ContactDropoutEnv
