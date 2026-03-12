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

def test_detect_event_logits():
    model = EventConsistentJEPA(obs_dim=10, action_dim=2, latent_dim=32)
    latent = torch.randn(5, 32)
    event_logits = model.detect_event_logits(latent)
    assert event_logits.shape == (5, 1)

def test_predict_contact_impulse():
    model = EventConsistentJEPA(obs_dim=10, action_dim=2, latent_dim=32)
    latent = torch.randn(5, 32)
    contact_impulse = model.predict_contact_impulse(latent)
    assert contact_impulse.shape == (5, 1)
    assert torch.all(contact_impulse >= 0)

def test_constrain_prediction():
    model = EventConsistentJEPA(obs_dim=10, action_dim=2, latent_dim=32)
    latent = torch.randn(5, 32)
    event_prob = torch.rand(5, 1)
    contact_impulse = torch.rand(5, 1)
    constrained_latent = model.constrain_prediction(latent, event_prob, contact_impulse)
    assert constrained_latent.shape == (5, 32)

def test_forward_latent():
    model = EventConsistentJEPA(obs_dim=10, action_dim=2, latent_dim=32)
    z = torch.randn(5, 32)
    action = torch.randn(5, 2)
    z_next, aux = model.forward_latent(z, action, return_aux=True)
    assert z_next.shape == (5, 32)
    assert aux['event_logits'].shape == (5, 1)
    assert aux['event_prob'].shape == (5, 1)
    assert aux['contact_impulse'].shape == (5, 1)

def test_physics_constraint_losses():
    model = EventConsistentJEPA(obs_dim=10, action_dim=2, latent_dim=32)
    impulse = torch.rand(5, 1)
    contact_distance = torch.tensor([[0.1], [0.0], [-0.01], [0.2], [-0.02]])
    events = torch.tensor([[0.0], [1.0], [1.0], [0.0], [1.0]])

    loss_event = model.event_supervision_loss(torch.randn(5, 1), events)
    loss_impulse = model.contact_impulse_loss(impulse, torch.rand(5, 1))
    loss_comp = model.complementarity_loss(impulse, contact_distance, events)
    loss_contact = model.contact_constraint_loss(impulse, contact_distance, events)

    assert loss_event.dim() == 0
    assert loss_impulse.dim() == 0
    assert loss_comp.dim() == 0
    assert loss_contact.dim() == 0
    assert loss_event.item() >= 0
    assert loss_impulse.item() >= 0
    assert loss_comp.item() >= 0
    assert loss_contact.item() >= 0

def test_contrastive_event_loss():
    model = EventConsistentJEPA(obs_dim=10, action_dim=2, latent_dim=32)
    pred = torch.randn(5, 32)
    target = torch.randn(5, 32)
    events = torch.tensor([[1.0], [0.0], [1.0], [0.0], [0.5]])
    loss = model.contrastive_event_loss(pred, target, events)
    assert loss.dim() == 0
    assert loss.item() >= 0
