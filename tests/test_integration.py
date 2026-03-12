import torch
from src.models.jepa import StandardLatentJEPA
from src.models.pano import PANOVelocityPredictor
from src.utils.training import train_standard_jepa, train_pano

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
    
    # Train for 2 epochs
    trained_model = train_standard_jepa(
        model, data, dt=0.002, n_epochs=2, device='cpu', return_model=True
    )
    assert trained_model is model

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
    
    trained_model = train_pano(
        model, data, n_epochs=2, batch_size=4, device='cpu'
    )
    assert trained_model is model
