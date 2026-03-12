import torch
import pytest
from src.models.event_jepa import EventConsistentJEPA

def test_event_consistent_jepa_initialization():
    model = EventConsistentJEPA(obs_dim=10, action_dim=2, latent_dim=32)
    assert model.obs_dim == 10
    assert model.action_dim == 2
    assert model.latent_dim == 32
    assert hasattr(model, 'contact_detector')
    assert hasattr(model, 'constraint_proj')

def test_detect_event():
    model = EventConsistentJEPA(obs_dim=10, action_dim=2, latent_dim=32)
    latent = torch.randn(5, 32)
    event_prob = model.detect_event(latent)
    assert event_prob.shape == (5, 1)
    assert torch.all(event_prob >= 0) and torch.all(event_prob <= 1)

def test_constrain_prediction():
    model = EventConsistentJEPA(obs_dim=10, action_dim=2, latent_dim=32)
    latent = torch.randn(5, 32)
    event_prob = torch.rand(5, 1)
    constrained_latent = model.constrain_prediction(latent, event_prob)
    assert constrained_latent.shape == (5, 32)

def test_forward_latent():
    model = EventConsistentJEPA(obs_dim=10, action_dim=2, latent_dim=32)
    z = torch.randn(5, 32)
    action = torch.randn(5, 2)
    z_next = model.forward_latent(z, action)
    assert z_next.shape == (5, 32)

def test_contrastive_event_loss():
    model = EventConsistentJEPA(obs_dim=10, action_dim=2, latent_dim=32)
    pred = torch.randn(5, 32)
    target = torch.randn(5, 32)
    events = torch.tensor([[1.0], [0.0], [1.0], [0.0], [0.5]])
    loss = model.contrastive_event_loss(pred, target, events)
    assert loss.dim() == 0
    assert loss.item() >= 0
