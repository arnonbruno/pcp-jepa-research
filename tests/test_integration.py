import torch
from src.models.jepa import StandardLatentJEPA
from src.models.event_jepa import EventConsistentJEPA
from src.models.pano import PANOVelocityPredictor
from src.utils.training import train_standard_jepa, train_pano, train_event_consistent_jepa

def test_train_standard_jepa():
    obs_dim = 11
    action_dim = 3
    batch_size = 8
    
    model = StandardLatentJEPA(obs_dim=obs_dim, action_dim=action_dim)
    
    data = {
        'obs': torch.randn(batch_size, obs_dim),
        'obs_prev': torch.randn(batch_size, obs_dim),
        'obs_next': torch.randn(batch_size, obs_dim),
        'obs_prev_next': torch.randn(batch_size, obs_dim),
        'action': torch.randn(batch_size, action_dim)
    }
    
    # Train for 1 epoch to get initial loss
    initial_loss_vel, initial_loss_pred = train_standard_jepa(
        model, data, dt=0.002, n_epochs=1, device='cpu', return_model=False
    )
    
    # Train for 10 more epochs
    final_loss_vel, final_loss_pred = train_standard_jepa(
        model, data, dt=0.002, n_epochs=10, device='cpu', return_model=False
    )
    
    # Loss should decrease
    assert final_loss_vel < initial_loss_vel or final_loss_pred < initial_loss_pred

def test_train_pano():
    obs_dim = 11
    action_dim = 3
    history_len = 5
    batch_size = 8
    
    model = PANOVelocityPredictor(obs_dim=obs_dim, action_dim=action_dim, history_len=history_len)
    
    data = {
        'obs': torch.randn(batch_size, obs_dim),
        'action_history': torch.randn(batch_size, history_len, action_dim),
        'velocity': torch.randn(batch_size, obs_dim)
    }
    
    import torch.nn.functional as F
    
    # Initial loss
    with torch.no_grad():
        initial_pred = model(data['obs'], data['action_history'])
        initial_loss = F.mse_loss(initial_pred, data['velocity']).item()

    trained_model = train_pano(
        model, data, n_epochs=10, batch_size=4, device='cpu'
    )
    
    # Final loss
    with torch.no_grad():
        final_pred = trained_model(data['obs'], data['action_history'])
        final_loss = F.mse_loss(final_pred, data['velocity']).item()
        
    assert final_loss < initial_loss
    assert trained_model is model


def test_train_event_consistent_jepa():
    obs_dim = 11
    action_dim = 3
    batch_size = 8

    model = EventConsistentJEPA(obs_dim=obs_dim, action_dim=action_dim)

    data = {
        'obs': torch.randn(batch_size, obs_dim),
        'obs_prev': torch.randn(batch_size, obs_dim),
        'obs_next': torch.randn(batch_size, obs_dim),
        'obs_prev_next': torch.randn(batch_size, obs_dim),
        'action': torch.randn(batch_size, action_dim),
        'contact': torch.randint(0, 2, (batch_size, 1)).float(),
        'contact_force': torch.rand(batch_size, 1),
        'contact_impulse': torch.rand(batch_size, 1),
        'contact_distance': torch.randn(batch_size, 1),
    }

    # Train for 1 epoch
    initial_losses = train_event_consistent_jepa(
        model, data, dt=0.002, n_epochs=1, device='cpu', return_model=False
    )
    
    # Train for 10 more epochs
    final_losses = train_event_consistent_jepa(
        model, data, dt=0.002, n_epochs=10, device='cpu', return_model=False
    )
    
    assert final_losses['total_loss'] < initial_losses['total_loss']
