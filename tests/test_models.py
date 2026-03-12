import torch
import numpy as np
from src.models.jepa import StandardLatentJEPA
from src.models.pano import PANOVelocityPredictor
from src.models.ekf import EKFEstimator

def test_standard_latent_jepa():
    obs_dim = 11
    action_dim = 3
    latent_dim = 64
    model = StandardLatentJEPA(obs_dim=obs_dim, action_dim=action_dim, latent_dim=latent_dim)
    
    # Check parameter count > 0
    param_count = sum(p.numel() for p in model.parameters())
    assert param_count > 0

    batch_size = 4
    obs = torch.randn(batch_size, obs_dim)
    obs_prev = torch.randn(batch_size, obs_dim)
    action = torch.randn(batch_size, action_dim)
    
    # Forward pass shapes
    z = model.encode(obs, obs_prev)
    assert z.shape == (batch_size, latent_dim)
    
    z_target = model.encode_target(obs, obs_prev)
    assert z_target.shape == (batch_size, latent_dim)
    
    delta_z = model.predict_residual(z, action)
    assert delta_z.shape == (batch_size, latent_dim)
    
    v_pred = model.decode_velocity(z)
    assert v_pred.shape == (batch_size, obs_dim)
    
        # Gradient flow
    loss = v_pred.sum() + delta_z.sum()
    loss.backward()
    
    # Check parameters (excluding target_encoder which doesn't get gradients)
    for name, param in model.named_parameters():
        if 'target_encoder' not in name:
            assert param.grad is not None

def test_pano_velocity_predictor():
    obs_dim = 11
    action_dim = 3
    history_len = 5
    model = PANOVelocityPredictor(obs_dim=obs_dim, action_dim=action_dim, history_len=history_len)
    
    param_count = sum(p.numel() for p in model.parameters())
    assert param_count > 0

    batch_size = 4
    obs = torch.randn(batch_size, obs_dim)
    action_history = torch.randn(batch_size, history_len, action_dim)
    
    # Forward pass shape
    v_pred = model(obs, action_history)
    assert v_pred.shape == (batch_size, obs_dim)
    
    # Gradient flow
    loss = v_pred.sum()
    loss.backward()
    for param in model.parameters():
        assert param.grad is not None

def test_ekf_estimator():
    obs_dim = 11
    ekf = EKFEstimator(obs_dim=obs_dim)
    
    obs = np.random.randn(obs_dim)
    ekf.reset(obs)
    
    est_obs = ekf.get_obs_estimate()
    assert est_obs.shape == (obs_dim,)
    assert np.allclose(est_obs, obs)
    
    est_vel = ekf.get_velocity_estimate()
    assert est_vel.shape == (obs_dim,)
    assert np.allclose(est_vel, np.zeros(obs_dim))
    
    ekf.predict()
    ekf.update(obs + 0.1)
    
    est_obs_new = ekf.get_obs_estimate()
    assert est_obs_new.shape == (obs_dim,)
