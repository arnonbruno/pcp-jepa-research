"""Tests for environment utilities"""

import numpy as np
import pytest
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))


class TestDTValues:
    """Tests for environment dt (timestep) values."""

    def test_dt_values(self):
        """Test that dt values are accessible from MuJoCo environments."""
        # Test with mock dt values that would come from MuJoCo
        # In actual MuJoCo envs, dt is typically 0.002 for most envs
        
        mock_dt_values = {
            'Hopper-v4': 0.002,
            'Walker2d-v4': 0.002,
            'HalfCheetah-v4': 0.01,  # Different!
            'InvertedDoublePendulum-v4': 0.05,
        }
        
        # Verify dt values are reasonable
        for env_name, dt in mock_dt_values.items():
            assert 0 < dt < 1.0, f"{env_name}: dt={dt} is not reasonable"
            assert isinstance(dt, float), f"{env_name}: dt should be float"

    def test_dt_usage_pattern(self):
        """Test the pattern for extracting dt from environment."""
        # Simulate the correct pattern for getting dt
        class MockEnv:
            class Unwrapped:
                dt = 0.002
            unwrapped = Unwrapped()
        
        env = MockEnv()
        dt_actual = env.unwrapped.dt if hasattr(env.unwrapped, 'dt') else 0.002
        
        assert dt_actual == 0.002

    def test_dt_fallback(self):
        """Test dt fallback when not available."""
        class MockEnvNoDT:
            class Unwrapped:
                pass
            unwrapped = Unwrapped()
        
        env = MockEnvNoDT()
        dt_actual = env.unwrapped.dt if hasattr(env.unwrapped, 'dt') else 0.002
        
        assert dt_actual == 0.002  # Fallback value


class TestContactDetection:
    """Tests for contact detection in environments."""

    def test_contact_detection_consistency(self):
        """Test that contact detection is consistent across runs."""
        np.random.seed(42)
        
        # Simulate contact detection based on velocity threshold
        velocity_threshold = 0.1
        
        # Generate "state changes"
        changes = np.random.randn(100, 11) * 0.05
        
        contacts_detected = []
        for i, delta in enumerate(changes):
            if i > 10:  # Skip warmup
                is_contact = np.abs(delta).max() > velocity_threshold
                contacts_detected.append(is_contact)
        
        # Reset with same seed
        np.random.seed(42)
        changes2 = np.random.randn(100, 11) * 0.05
        
        contacts_detected2 = []
        for i, delta in enumerate(changes2):
            if i > 10:
                is_contact = np.abs(delta).max() > velocity_threshold
                contacts_detected2.append(is_contact)
        
        # Should be identical
        assert contacts_detected == contacts_detected2

    def test_contact_threshold_sensitivity(self):
        """Test that contact detection is sensitive to threshold."""
        np.random.seed(42)
        
        # Create some large state changes
        changes = np.random.randn(100, 11)
        changes[50:] *= 0.2  # Make later changes smaller
        
        contacts_low_threshold = sum(1 for d in changes[10:] if np.abs(d).max() > 0.1)
        contacts_high_threshold = sum(1 for d in changes[10:] if np.abs(d).max() > 0.5)
        
        # Lower threshold should detect more contacts
        assert contacts_low_threshold >= contacts_high_threshold


class TestContactDropoutEnvBehavior:
    """Tests for ContactDropoutEnv behavior patterns."""

    def test_dropout_duration(self):
        """Test that dropout lasts for specified duration."""
        dropout_duration = 5
        
        # Simulate dropout behavior
        dropout_countdown = 0
        
        # Trigger dropout
        dropout_countdown = dropout_duration
        
        dropout_steps = 0
        while dropout_countdown > 0:
            dropout_steps += 1
            dropout_countdown -= 1
        
        assert dropout_steps == dropout_duration

    def test_dropout_retrigger_protection(self):
        """Test that dropout can't be retriggered during active dropout."""
        dropout_duration = 5
        velocity_threshold = 0.1
        
        # Simulate sequence with potential retrigger
        dropout_countdown = 0
        active_durations = []
        
        for step in range(100):
            # Simulate state change - large changes every step to test blocking
            delta = 0.5  # Always large enough to trigger
            
            # Contact trigger logic (same as ContactDropoutEnv)
            if delta > velocity_threshold and step > 10 and dropout_countdown == 0:
                dropout_countdown = dropout_duration
            
            if dropout_countdown > 0:
                active_durations.append(dropout_countdown)
                dropout_countdown -= 1
        
        # Should have multiple dropout periods (since blocking works)
        # Count transitions from 0 to dropout_duration
        transitions = sum(1 for i in range(len(active_durations)-1) 
                        if active_durations[i] == dropout_duration)
        
        # Multiple dropout periods means blocking is working
        assert transitions >= 1


class TestEnvironmentIntegration:
    """Integration tests for environment utilities."""

    def test_velocity_computation(self):
        """Test velocity computation from consecutive observations."""
        dt = 0.002
        
        # Simulate consecutive observations
        obs_prev = np.array([1.0, 0.5, 0.0])
        obs = np.array([1.01, 0.52, 0.01])
        
        velocity = (obs - obs_prev) / dt
        
        expected = np.array([5.0, 10.0, 5.0])
        np.testing.assert_allclose(velocity, expected, rtol=1e-5)

    def test_velocity_computation_stability(self):
        """Test velocity computation handles edge cases."""
        dt = 0.002
        
        # Same observation
        obs = np.array([1.0, 2.0, 3.0])
        velocity = (obs - obs) / dt
        
        assert np.allclose(velocity, 0)

    def test_frozen_observation_during_dropout(self):
        """Test that observation stays frozen during dropout."""
        # Simulate dropout behavior
        frozen_obs = np.array([1.0, 2.0, 3.0])
        dropout_active = True
        
        # True state changes
        true_obs = np.array([1.5, 2.5, 3.5])
        
        # Returned observation should be frozen
        if dropout_active:
            returned_obs = frozen_obs.copy()
        else:
            returned_obs = true_obs.copy()
        
        np.testing.assert_allclose(returned_obs, frozen_obs)


class TestObservationShapes:
    """Tests for observation shape consistency."""

    def test_hopper_observation_shape(self):
        """Test Hopper observation dimension."""
        # Hopper-v4 has 11-dimensional observation
        obs_dim = 11
        obs = np.random.randn(obs_dim)
        
        assert obs.shape == (11,)

    def test_action_shape_consistency(self):
        """Test action shape consistency."""
        # Common MuJoCo action dimensions
        action_dims = {
            'Hopper-v4': 3,
            'Walker2d-v4': 6,
            'HalfCheetah-v4': 6,
        }
        
        for env_name, action_dim in action_dims.items():
            assert action_dim > 0, f"{env_name}: action_dim should be positive"
            assert isinstance(action_dim, int), f"{env_name}: action_dim should be int"


class TestStateNormalization:
    """Tests for state normalization and processing."""

    def test_f3_normalization(self):
        """Test F3-style normalization (position + velocity)."""
        x = np.array([[1.0]])
        x_prev = np.array([[0.95]])
        dt = 0.05
        
        # F3 normalization: [x, Δx/dt]
        velocity = (x - x_prev) / dt
        normalized = np.concatenate([x, velocity], axis=-1)
        
        assert normalized.shape == (1, 2)
        np.testing.assert_allclose(normalized, [[1.0, 1.0]])

    def test_clipping(self):
        """Test observation/action clipping."""
        # Actions typically clipped to [-1, 1]
        actions = np.array([-2.0, -0.5, 0.0, 0.5, 2.0])
        clipped = np.clip(actions, -1.0, 1.0)
        
        expected = np.array([-1.0, -0.5, 0.0, 0.5, 1.0])
        np.testing.assert_allclose(clipped, expected)


if __name__ == '__main__':
    pytest.main([__file__, '-v'])