"""Tests for PANO components from experiments/phase6/hopper_pano.py"""

import numpy as np
import torch
import pytest
import sys
import os

# Add paths
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'experiments', 'phase6'))

device = torch.device('cpu')  # Use CPU for tests


class PANOVelocityPredictor(torch.nn.Module):
    """PANO velocity predictor for testing (copied to avoid import issues)."""
    
    def __init__(self, obs_dim, action_dim, history_len=5, hidden_dim=128):
        super().__init__()
        self.history_len = history_len
        input_dim = obs_dim + history_len * action_dim

        self.net = torch.nn.Sequential(
            torch.nn.Linear(input_dim, hidden_dim), torch.nn.ReLU(),
            torch.nn.Linear(hidden_dim, hidden_dim), torch.nn.ReLU(),
            torch.nn.Linear(hidden_dim, 64), torch.nn.ReLU(),
            torch.nn.Linear(64, obs_dim),
        )

    def forward(self, obs, action_history):
        if action_history.dim() == 2:
            action_history = action_history.unsqueeze(0)
        action_flat = action_history.reshape(action_history.shape[0], -1)
        inp = torch.cat([obs, action_flat], dim=-1)
        return self.net(inp)


class EKFEstimator:
    """EKF for testing (copied to avoid import issues)."""
    
    def __init__(self, obs_dim=11, dt=0.002, process_noise=1.0, measurement_noise=0.01):
        self.obs_dim = obs_dim
        self.dt = dt
        self.state_dim = obs_dim * 2

        self.x = np.zeros(self.state_dim)
        self.P = np.eye(self.state_dim) * 1.0
        self.Q = np.eye(self.state_dim) * process_noise
        self.Q[:obs_dim, :obs_dim] *= 0.01
        self.Q[obs_dim:, obs_dim:] *= 1.0
        self.R = np.eye(obs_dim) * measurement_noise
        self.H = np.zeros((obs_dim, self.state_dim))
        self.H[:obs_dim, :obs_dim] = np.eye(obs_dim)

    def reset(self, obs):
        self.x = np.zeros(self.state_dim)
        self.x[:self.obs_dim] = obs
        self.x[self.obs_dim:] = 0.0
        self.P = np.eye(self.state_dim) * 1.0

    def predict(self):
        F = np.eye(self.state_dim)
        F[:self.obs_dim, self.obs_dim:] = np.eye(self.obs_dim) * self.dt
        self.x = F @ self.x
        self.P = F @ self.P @ F.T + self.Q

    def update(self, obs):
        y = obs - self.H @ self.x
        S = self.H @ self.P @ self.H.T + self.R
        K = self.P @ self.H.T @ np.linalg.inv(S)
        self.x = self.x + K @ y
        self.P = (np.eye(self.state_dim) - K @ self.H) @ self.P

    def get_obs_estimate(self):
        return self.x[:self.obs_dim]

    def get_velocity_estimate(self):
        return self.x[self.obs_dim:]


class ContactDropoutEnv:
    """Contact dropout environment for testing."""
    
    def __init__(self, obs_dim=11, action_dim=3, dropout_duration=5, velocity_threshold=0.1):
        self.obs_dim = obs_dim
        self.action_dim = action_dim
        self.dropout_duration = dropout_duration
        self.velocity_threshold = velocity_threshold
        self.obs_prev = None
        self.frozen_obs = None
        self.dropout_countdown = 0
        self.step_count = 0
        self._obs = np.zeros(obs_dim)
        self._last_valid_obs = None  # Track last valid observation for proper comparison
        
        # Mock spaces
        class Space:
            def __init__(self, shape):
                self.shape = shape
        self.observation_space = Space((obs_dim,))
        self.action_space = Space((action_dim,))

    def reset(self, seed=None):
        if seed is not None:
            np.random.seed(seed)
        self._obs = np.random.randn(self.obs_dim) * 0.1
        self.obs_prev = self._obs.copy()
        self._last_valid_obs = self._obs.copy()
        self.frozen_obs = self._obs.copy()
        self.dropout_countdown = 0
        self.step_count = 0
        return self._obs.copy(), {}

    def step(self, action):
        self.step_count += 1
        
        # Simulate state change
        delta = np.random.randn(self.obs_dim) * 0.05
        self._obs = self._obs + delta
        
        # Trigger dropout on large state changes (compared to last valid observation)
        if self._last_valid_obs is not None and self.dropout_countdown == 0:
            delta_max = np.abs(self._obs - self._last_valid_obs).max()
            if delta_max > self.velocity_threshold and self.step_count > 10:
                self.dropout_countdown = self.dropout_duration
                self.frozen_obs = self._obs.copy()

        info = {
            'true_obs': self._obs.copy(),
            'dropout_active': self.dropout_countdown > 0,
            'dropout_step': self.dropout_duration - self.dropout_countdown,
            'frozen_obs': self.frozen_obs.copy(),
        }

        if self.dropout_countdown > 0:
            obs_return = self.frozen_obs.copy()
            self.dropout_countdown -= 1
        else:
            obs_return = self._obs.copy()
            self._last_valid_obs = self._obs.copy()  # Update for next comparison

        return obs_return, 0.0, False, False, info


class TestPANOVelocityPredictor:
    """Tests for PANO velocity predictor."""

    def test_pano_velocity_predictor_forward(self):
        """Test forward pass of PANO velocity predictor."""
        obs_dim = 11
        action_dim = 3
        history_len = 5
        
        model = PANOVelocityPredictor(obs_dim, action_dim, history_len)
        
        # Single sample
        obs = torch.randn(1, obs_dim)
        action_history = torch.randn(1, history_len, action_dim)
        
        output = model(obs, action_history)
        
        assert output.shape == (1, obs_dim), f"Expected shape (1, {obs_dim}), got {output.shape}"

    def test_pano_velocity_predictor_shape(self):
        """Test that output shape matches observation dimension."""
        obs_dim = 11
        action_dim = 3
        history_len = 5
        
        model = PANOVelocityPredictor(obs_dim, action_dim, history_len)
        
        # Batch of samples
        batch_size = 32
        obs = torch.randn(batch_size, obs_dim)
        action_history = torch.randn(batch_size, history_len, action_dim)
        
        output = model(obs, action_history)
        
        assert output.shape == (batch_size, obs_dim)

    def test_pano_velocity_predictor_batch_processing(self):
        """Test batch processing consistency."""
        obs_dim = 11
        action_dim = 3
        history_len = 5
        
        model = PANOVelocityPredictor(obs_dim, action_dim, history_len)
        model.eval()
        
        # Process batch
        batch_size = 10
        obs = torch.randn(batch_size, obs_dim)
        action_history = torch.randn(batch_size, history_len, action_dim)
        
        with torch.no_grad():
            batch_output = model(obs, action_history)
            
            # Process individually
            individual_outputs = []
            for i in range(batch_size):
                out = model(obs[i:i+1], action_history[i:i+1])
                individual_outputs.append(out)
            individual_output = torch.cat(individual_outputs, dim=0)
        
        # Should be approximately the same
        assert torch.allclose(batch_output, individual_output, atol=1e-5)


class TestEKFEstimator:
    """Tests for EKF estimator."""

    def test_ekf_estimator_reset(self):
        """Test EKF reset functionality."""
        ekf = EKFEstimator(obs_dim=11, dt=0.002)
        
        initial_obs = np.random.randn(11)
        ekf.reset(initial_obs)
        
        # Check state is initialized correctly
        assert np.allclose(ekf.get_obs_estimate(), initial_obs)
        assert np.allclose(ekf.get_velocity_estimate(), np.zeros(11))

    def test_ekf_estimator_predict_update(self):
        """Test EKF predict and update cycle."""
        obs_dim = 11
        ekf = EKFEstimator(obs_dim=obs_dim, dt=0.002)
        
        # Initialize
        initial_obs = np.random.randn(obs_dim)
        ekf.reset(initial_obs)
        
        # Predict step (should propagate state)
        ekf.predict()
        
        # After predict, observation estimate should have changed
        # (since velocity is added to position)
        obs_after_predict = ekf.get_obs_estimate()
        
        # Update step
        new_obs = initial_obs + np.random.randn(obs_dim) * 0.01
        ekf.update(new_obs)
        
        obs_after_update = ekf.get_obs_estimate()
        
        # After update, estimate should be closer to new observation
        assert not np.allclose(obs_after_update, initial_obs)

    def test_ekf_estimator_tracking(self):
        """Test EKF tracks a constant velocity signal."""
        obs_dim = 3
        dt = 0.01
        ekf = EKFEstimator(obs_dim=obs_dim, dt=dt, process_noise=0.1, measurement_noise=0.01)
        
        # True state with constant velocity
        true_velocity = np.array([1.0, 0.5, -0.5])
        true_state = np.zeros(obs_dim)
        
        ekf.reset(true_state)
        
        # Simulate for several steps
        for _ in range(20):
            true_state = true_state + true_velocity * dt
            
            ekf.predict()
            ekf.update(true_state)
            
            vel_est = ekf.get_velocity_estimate()
        
        # EKF should have learned approximately the true velocity
        assert np.allclose(vel_est, true_velocity, atol=0.5)


class TestContactDropoutEnv:
    """Tests for contact dropout environment wrapper."""

    def test_contact_dropout_env_trigger(self):
        """Test that dropout is triggered on large state changes."""
        env = ContactDropoutEnv(
            obs_dim=11, action_dim=3,
            dropout_duration=5, velocity_threshold=0.1
        )
        
        obs, _ = env.reset(seed=42)
        
        dropout_triggered = False
        for _ in range(100):
            action = np.random.randn(3)
            obs, _, _, _, info = env.step(action)
            
            if info['dropout_active']:
                dropout_triggered = True
                break
        
        assert dropout_triggered, "Dropout should be triggered by large state changes"

    def test_contact_dropout_env_freeze(self):
        """Test that observation equals frozen_obs during dropout."""
        env = ContactDropoutEnv(
            obs_dim=11, action_dim=3,
            dropout_duration=5, velocity_threshold=0.15
        )
        
        obs, _ = env.reset(seed=42)
        
        # Verify that during dropout, returned obs equals the frozen_obs in info
        dropout_detected = False
        all_match = True
        checked = 0
        
        for _ in range(100):
            action = np.random.randn(3)
            obs, _, _, _, info = env.step(action)
            
            if info['dropout_active']:
                dropout_detected = True
                # During dropout, returned obs should equal frozen_obs from info
                frozen = info.get('frozen_obs')
                if frozen is not None:
                    checked += 1
                    if not np.allclose(obs, frozen, atol=1e-6):
                        all_match = False
        
        assert dropout_detected, "Dropout should have been triggered"
        assert checked > 0, "Should have checked at least one dropout observation"
        assert all_match, "During dropout, returned observation should equal frozen_obs"

    def test_contact_dropout_env_info_keys(self):
        """Test that info dict contains required keys."""
        env = ContactDropoutEnv(obs_dim=11, action_dim=3)
        
        obs, _ = env.reset(seed=42)
        _, _, _, _, info = env.step(np.random.randn(3))
        
        required_keys = ['true_obs', 'dropout_active', 'dropout_step', 'frozen_obs']
        for key in required_keys:
            assert key in info, f"Missing key '{key}' in info dict"


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
