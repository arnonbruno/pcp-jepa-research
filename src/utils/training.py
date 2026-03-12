import torch
import torch.nn.functional as F
import numpy as np

def train_standard_jepa(model, data, dt, n_epochs=100, lambda_vel=10.0, lambda_pred=0.1, device='cuda', return_model=False):
    """Train Standard Latent JEPA. Returns either (vel_loss, pred_loss) or the trained model."""
    if isinstance(device, str):
        device = torch.device(device if torch.cuda.is_available() else 'cpu')
        
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    batch_size = 256
    n_samples = len(data['obs'])

    final_vel_losses, final_pred_losses = [], []

    for epoch in range(n_epochs):
        idx = torch.randperm(n_samples, device=device)
        vel_losses, pred_losses = [], []
        
        total_loss = 0
        total_vel_loss = 0
        total_pred_loss = 0

        for i in range(0, n_samples, batch_size):
            b = idx[i:i+batch_size]
            z_t = model.encode(data['obs'][b], data['obs_prev'][b], dt)
            z_target = model.encode_target(data['obs_next'][b], data['obs_prev_next'][b], dt)

            delta_z = model.predict_residual(z_t, data['action'][b])
            z_pred = z_t + delta_z

            v_pred = model.decode_velocity(z_t)
            v_true = (data['obs'][b] - data['obs_prev'][b]) / dt

            loss_vel = F.mse_loss(v_pred, v_true)
            loss_pred = F.mse_loss(z_pred, z_target.detach())

            loss = lambda_vel * loss_vel + lambda_pred * loss_pred

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            model.update_target()

            vel_losses.append(loss_vel.item())
            pred_losses.append(loss_pred.item())
            
            total_loss += loss.item()
            total_vel_loss += loss_vel.item()
            total_pred_loss += loss_pred.item()

        # Record final-epoch losses
        if epoch >= n_epochs - 5:
            final_vel_losses.extend(vel_losses)
            final_pred_losses.extend(pred_losses)
            
        if return_model and (epoch + 1) % 20 == 0:
            print(f"Epoch {epoch+1}: loss={total_loss:.1f} (vel={total_vel_loss:.1f}, pred={total_pred_loss:.4f})")

    if return_model:
        return model
    return np.mean(final_vel_losses), np.mean(final_pred_losses)

def train_standard_jepa_multistep(model, data, n_epochs=100, n_rollout=3, lambda_vel=10.0, lambda_pred=0.1, dt=0.002, device='cuda'):
    """
    Train with multi-step latent rollout to improve long-range prediction.
    """
    if isinstance(device, str):
        device = torch.device(device if torch.cuda.is_available() else 'cpu')
        
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    
    batch_size = 256
    n_samples = len(data['obs'])
    
    for epoch in range(n_epochs):
        idx = torch.randperm(n_samples)
        total_loss = 0
        total_vel_loss = 0
        total_pred_loss = 0
        
        for i in range(0, n_samples - n_rollout, batch_size):
            b = idx[i:i+batch_size]
            
            # Single-step prediction loss
            z_t = model.encode(data['obs'][b], data['obs_prev'][b], dt)
            z_target = model.encode_target(data['obs_next'][b], data['obs_prev_next'][b], dt)
            
            delta_z = model.predict_residual(z_t, data['action'][b])
            z_pred = z_t + delta_z
            
            v_pred = model.decode_velocity(z_t)
            v_true = (data['obs'][b] - data['obs_prev'][b]) / dt
            
            loss_vel = F.mse_loss(v_pred, v_true)
            loss_pred = F.mse_loss(z_pred, z_target.detach())
            
            # Multi-step rollout loss
            loss_multistep = 0
            z_rollout = z_t.clone()
            for k in range(min(n_rollout, n_samples - i - batch_size)):
                idx_k = b + k + 1
                if idx_k.max() < n_samples:
                    delta_z_k = model.predict_residual(z_rollout, data['action'][idx_k - 1])
                    z_rollout = z_rollout + delta_z_k
                    
                    z_target_k = model.encode_target(
                        data['obs_next'][idx_k - 1], 
                        data['obs_prev_next'][idx_k - 1], 
                        dt
                    )
                    loss_multistep += F.mse_loss(z_rollout, z_target_k.detach())
            
            loss = lambda_vel * loss_vel + lambda_pred * (loss_pred + 0.1 * loss_multistep)
            
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            
            model.update_target()
            
            total_loss += loss.item()
            total_vel_loss += loss_vel.item()
            total_pred_loss += loss_pred.item()
        
        if (epoch + 1) % 20 == 0:
            print(f"Epoch {epoch+1}: loss={total_loss:.1f} (vel={total_vel_loss:.1f}, pred={total_pred_loss:.4f})")
    
    return model

def train_pano(model, data, n_epochs=100, lr=1e-3, batch_size=256, device='cuda'):
    """Train PANO velocity predictor."""
    if isinstance(device, str):
        device = torch.device(device if torch.cuda.is_available() else 'cpu')
        
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    n_samples = len(data['obs'])

    for epoch in range(n_epochs):
        idx = torch.randperm(n_samples, device=device)
        total_loss = 0
        n_batches = 0

        for i in range(0, n_samples, batch_size):
            b = idx[i:i+batch_size]
            v_pred = model(data['obs'][b], data['action_history'][b])
            loss = F.mse_loss(v_pred, data['velocity'][b])

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            n_batches += 1

        if (epoch + 1) % 25 == 0:
            print(f"  PANO training epoch {epoch+1}/{n_epochs}: loss={total_loss/n_batches:.4f}")

    return model
