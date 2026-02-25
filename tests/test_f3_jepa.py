"""Tests for F3-JEPA architecture from experiments/phase5/f3_jepa.py"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import pytest
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

device = torch.device('cpu')


# =============================================================================
# F3-JEPA Components (copied for testing)
# =============================================================================

class F3Encoder(nn.Module):
    """Encoder with F3-normalized input: (x, Δx/dt)"""
    def __init__(self, latent_dim=32):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(2, 64), nn.ReLU(),
            nn.Linear(64, 64), nn.ReLU(),
            nn.Linear(64, latent_dim)
        )
    
    def forward(self, x, x_prev, dt=0.05):
        delta_v = (x - x_prev) / dt
        inp = torch.cat([x, delta_v], dim=-1)
        return self.net(inp)


class TargetEncoder(nn.Module):
    """EMA of context encoder for stable targets"""
    def __init__(self, latent_dim=32):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(2, 64), nn.ReLU(),
            nn.Linear(64, 64), nn.ReLU(),
            nn.Linear(64, latent_dim)
        )
    
    def forward(self, x, x_prev, dt=0.05):
        delta_v = (x - x_prev) / dt
        inp = torch.cat([x, delta_v], dim=-1)
        return self.net(inp)
    
    @torch.no_grad()
    def update_ema(self, encoder, tau=0.996):
        for tp, ep in zip(self.parameters(), encoder.parameters()):
            tp.data.mul_(tau).add_(ep.data, alpha=1 - tau)


class VelocityDecoder(nn.Module):
    """Decode velocity from latent state"""
    def __init__(self, latent_dim=32):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(latent_dim, 32), nn.ReLU(),
            nn.Linear(32, 1)
        )
    
    def forward(self, z):
        return self.net(z)


class LatentPredictor(nn.Module):
    """Predict z_next from z and action"""
    def __init__(self, latent_dim=32):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(latent_dim + 1, 64), nn.ReLU(),
            nn.Linear(64, 64), nn.ReLU(),
            nn.Linear(64, latent_dim)
        )
    
    def forward(self, z, a):
        return self.net(torch.cat([z, a], dim=-1))


class EventHead(nn.Module):
    """Predict impact probability from latent state"""
    def __init__(self, latent_dim=32):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(latent_dim, 32), nn.ReLU(),
            nn.Linear(32, 1), nn.Sigmoid()
        )
    
    def forward(self, z):
        return self.net(z)


class F3JEPA(nn.Module):
    """F3-JEPA: Unified architecture"""
    def __init__(self, latent_dim=32):
        super().__init__()
        self.encoder = F3Encoder(latent_dim)
        self.target_encoder = TargetEncoder(latent_dim)
        self.velocity_decoder = VelocityDecoder(latent_dim)
        self.predictor = LatentPredictor(latent_dim)
        self.event_head = EventHead(latent_dim)
        
        self.target_encoder.load_state_dict(self.encoder.state_dict())
    
    def encode(self, x, x_prev, dt=0.05):
        return self.encoder(x, x_prev, dt)
    
    def encode_target(self, x, x_prev, dt=0.05):
        with torch.no_grad():
            return self.target_encoder(x, x_prev, dt)
    
    def decode_velocity(self, z):
        return self.velocity_decoder(z)
    
    def predict(self, z, a):
        return self.predictor(z, a)
    
    def predict_impact(self, z):
        return self.event_head(z)
    
    def update_target(self, tau=0.996):
        self.target_encoder.update_ema(self.encoder, tau)


class F3JEPAController:
    """Controller using F3-JEPA for velocity estimation"""
    
    def __init__(self, model, dt=0.05):
        self.model = model
        self.dt = dt
        self.z = None
        self.last_x = None
        self.last_a = None
    
    def reset(self, x0):
        self.z = None
        self.last_x = x0
        self.last_a = 0.0
    
    def get_velocity(self, x, observation_available=True):
        if observation_available:
            x_t = torch.tensor([[x]], dtype=torch.float32, device=device)
            xp_t = torch.tensor([[self.last_x]], dtype=torch.float32, device=device)
            
            with torch.no_grad():
                self.z = self.model.encode(x_t, xp_t, self.dt)
                v = self.model.decode_velocity(self.z).item()
            
            self.last_x = x
        else:
            if self.z is not None and self.last_a is not None:
                a_t = torch.tensor([[self.last_a]], dtype=torch.float32, device=device)
                with torch.no_grad():
                    self.z = self.model.predict(self.z, a_t)
                    v = self.model.decode_velocity(self.z).item()
            else:
                v = 0.0
        
        return v


class Ball:
    """Simple bouncing ball environment for testing."""
    def __init__(self, dt=0.05, restitution=0.8):
        self.dt = dt
        self.restitution = restitution
        self.g = 9.81
        self.x = 0
        self.v = 0
        self.impact = False
    
    def reset(self, x0):
        self.x = x0
        self.v = 0
        self.impact = False
        return self.x, self.v
    
    def step(self, a):
        a = np.clip(a, -2, 2)
        self.v += (-self.g + a) * self.dt
        self.x += self.v * self.dt
        self.impact = False
        if self.x < 0:
            self.x = -self.x * self.restitution
            self.v = -self.v * self.restitution
            self.impact = True
        if self.x > 3:
            self.x = 3 - (self.x - 3) * self.restitution
            self.v = -self.v * self.restitution
            self.impact = True
        return self.x, self.v


class TestF3Encoder:
    """Tests for F3 Encoder."""

    def test_f3_encoder(self):
        """Test F3 encoder forward pass."""
        latent_dim = 32
        encoder = F3Encoder(latent_dim)
        
        x = torch.randn(16, 1)
        x_prev = torch.randn(16, 1)
        dt = 0.05
        
        z = encoder(x, x_prev, dt)
        
        assert z.shape == (16, latent_dim), f"Expected (16, {latent_dim}), got {z.shape}"

    def test_f3_encoder_velocity_encoding(self):
        """Test that F3 encoder uses velocity information."""
        latent_dim = 16
        encoder = F3Encoder(latent_dim)
        
        # Same position, different velocities
        x = torch.tensor([[1.0]])
        x_prev_slow = torch.tensor([[0.9]])  # v = 2
        x_prev_fast = torch.tensor([[0.0]])  # v = 20
        
        dt = 0.05
        
        z_slow = encoder(x, x_prev_slow, dt)
        z_fast = encoder(x, x_prev_fast, dt)
        
        # Should produce different latents for different velocities
        assert not torch.allclose(z_slow, z_fast, atol=1e-5)


class TestVelocityDecoder:
    """Tests for Velocity Decoder."""

    def test_velocity_decoder(self):
        """Test velocity decoder output shape."""
        latent_dim = 32
        decoder = VelocityDecoder(latent_dim)
        
        z = torch.randn(8, latent_dim)
        v = decoder(z)
        
        assert v.shape == (8, 1), f"Expected (8, 1), got {v.shape}"

    def test_velocity_decoder_range(self):
        """Test that velocity decoder can output positive and negative values."""
        latent_dim = 32
        decoder = VelocityDecoder(latent_dim)
        
        # Different latent states should produce different velocities
        z1 = torch.randn(1, latent_dim)
        z2 = -z1  # Opposite latent
        
        v1 = decoder(z1).item()
        v2 = decoder(z2).item()
        
        # Should be different (may not be exact opposites due to non-linearity)
        assert v1 != v2


class TestLatentPredictor:
    """Tests for Latent Predictor."""

    def test_latent_predictor(self):
        """Test latent predictor output shape."""
        latent_dim = 32
        predictor = LatentPredictor(latent_dim)
        
        z = torch.randn(8, latent_dim)
        a = torch.randn(8, 1)
        
        z_pred = predictor(z, a)
        
        assert z_pred.shape == (8, latent_dim), f"Expected (8, {latent_dim}), got {z_pred.shape}"

    def test_latent_predictor_action_conditioned(self):
        """Test that predictor uses action information."""
        latent_dim = 16
        predictor = LatentPredictor(latent_dim)
        
        z = torch.randn(1, latent_dim)
        a_pos = torch.tensor([[1.0]])
        a_neg = torch.tensor([[-1.0]])
        
        z_pred_pos = predictor(z, a_pos)
        z_pred_neg = predictor(z, a_neg)
        
        # Different actions should produce different predictions
        assert not torch.allclose(z_pred_pos, z_pred_neg, atol=1e-5)


class TestF3JEPAController:
    """Tests for F3-JEPA Controller."""

    def test_f3jepa_controller(self):
        """Test controller initialization and reset."""
        model = F3JEPA(latent_dim=32)
        controller = F3JEPAController(model, dt=0.05)
        
        controller.reset(1.0)
        
        assert controller.last_x == 1.0
        assert controller.last_a == 0.0
        assert controller.z is None

    def test_f3jepa_controller_velocity_estimation(self):
        """Test that controller can estimate velocity."""
        model = F3JEPA(latent_dim=32)
        controller = F3JEPAController(model, dt=0.05)
        controller.reset(1.0)
        
        # Get velocity estimate with observation
        v = controller.get_velocity(1.1, observation_available=True)
        
        assert isinstance(v, float)

    def test_f3jepa_controller_dropout_handling(self):
        """Test controller behavior during dropout (no observation)."""
        model = F3JEPA(latent_dim=32)
        controller = F3JEPAController(model, dt=0.05)
        controller.reset(1.0)
        
        # First, get observation to initialize latent
        controller.get_velocity(1.0, observation_available=True)
        controller.last_a = 0.5  # Set last action
        
        # During dropout, use predicted latent
        v_dropout = controller.get_velocity(1.0, observation_available=False)
        
        assert isinstance(v_dropout, float)

    def test_f3jepa_controller_no_dropout_state(self):
        """Test controller when dropout starts without prior observation."""
        model = F3JEPA(latent_dim=32)
        controller = F3JEPAController(model, dt=0.05)
        controller.reset(1.0)
        
        # Immediately request dropout prediction (no prior latent state)
        v = controller.get_velocity(1.0, observation_available=False)
        
        # Should return 0.0 as default when no latent state
        assert v == 0.0


class TestF3JEPAModel:
    """Tests for full F3-JEPA model."""

    def test_f3jepa_encode(self):
        """Test F3-JEPA encoding."""
        model = F3JEPA(latent_dim=32)
        
        x = torch.randn(8, 1)
        x_prev = torch.randn(8, 1)
        
        z = model.encode(x, x_prev, dt=0.05)
        
        assert z.shape == (8, 32)

    def test_f3jepa_target_encode(self):
        """Test F3-JEPA target encoding."""
        model = F3JEPA(latent_dim=32)
        
        x = torch.randn(8, 1)
        x_prev = torch.randn(8, 1)
        
        z_target = model.encode_target(x, x_prev, dt=0.05)
        
        assert z_target.shape == (8, 32)
        # Target encoder should not have gradients
        assert not z_target.requires_grad

    def test_f3jepa_predict_impact(self):
        """Test F3-JEPA impact prediction."""
        model = F3JEPA(latent_dim=32)
        
        z = torch.randn(8, 32)
        impact_prob = model.predict_impact(z)
        
        assert impact_prob.shape == (8, 1)
        # Should be probabilities between 0 and 1
        assert (impact_prob >= 0).all() and (impact_prob <= 1).all()

    def test_f3jepa_update_target(self):
        """Test F3-JEPA target encoder EMA update."""
        model = F3JEPA(latent_dim=32)
        
        # First, modify encoder weights to create a difference
        with torch.no_grad():
            model.encoder.net[0].weight.add_(torch.randn_like(model.encoder.net[0].weight) * 0.1)
        
        # Store initial target encoder weights
        initial_weight = model.target_encoder.net[0].weight.clone()
        
        # Update target encoder
        model.update_target(tau=0.996)
        
        # Weights should have changed (moved toward encoder)
        updated_weight = model.target_encoder.net[0].weight
        # The update should move toward encoder (which is now different)
        assert not torch.allclose(initial_weight, updated_weight, atol=1e-8)


class TestBallEnvironment:
    """Tests for Ball environment."""

    def test_ball_reset(self):
        """Test ball reset."""
        env = Ball(dt=0.05)
        x, v = env.reset(1.5)
        
        assert x == 1.5
        assert v == 0
        assert not env.impact

    def test_ball_dynamics(self):
        """Test ball dynamics without impact."""
        env = Ball(dt=0.05, restitution=0.8)
        env.reset(3.0)  # Start at higher position
        
        # Apply upward force to counteract gravity
        for _ in range(10):
            x, v = env.step(a=9.81)  # Counteract gravity
        
        # Should stay roughly at same height (no ground impact)
        # Ball at x=3.0 won't hit ground with gravity counteracted
        assert not env.impact or x > 2.5  # Either no impact or stayed high

    def test_ball_impact(self):
        """Test ball impact detection."""
        env = Ball(dt=0.05)
        env.reset(0.1)  # Start near ground
        
        # Let it fall (no action)
        impact_detected = False
        for _ in range(20):
            _, _ = env.step(a=0)
            if env.impact:
                impact_detected = True
                break
        
        assert impact_detected, "Ball should impact ground"


if __name__ == '__main__':
    pytest.main([__file__, '-v'])