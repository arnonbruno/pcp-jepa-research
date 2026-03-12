import torch
import torch.nn as nn
import torch.nn.functional as F

class StandardLatentJEPA(nn.Module):
    """
    Standard Latent JEPA: Residual latent dynamics z_next = z + Δz.
    
    Architecture:
    1. Residual predictor: z_next = z + Δz
    2. EMA target encoder with stop-gradient
    3. Aggressive velocity loss
    """
    def __init__(self, obs_dim=11, action_dim=3, latent_dim=64):
        super().__init__()

        self.encoder = nn.Sequential(
            nn.Linear(obs_dim * 2, 128), nn.ReLU(),
            nn.Linear(128, 128), nn.ReLU(),
            nn.Linear(128, latent_dim),
        )

        self.target_encoder = nn.Sequential(
            nn.Linear(obs_dim * 2, 128), nn.ReLU(),
            nn.Linear(128, 128), nn.ReLU(),
            nn.Linear(128, latent_dim),
        )
        self.target_encoder.load_state_dict(self.encoder.state_dict())

        self.velocity_decoder = nn.Sequential(
            nn.Linear(latent_dim, 64), nn.ReLU(),
            nn.Linear(64, obs_dim),
        )

        self.predictor = nn.Sequential(
            nn.Linear(latent_dim + action_dim, 128), nn.ReLU(),
            nn.Linear(128, 128), nn.ReLU(),
            nn.Linear(128, latent_dim),
        )
        
        self.obs_dim = obs_dim
        self.action_dim = action_dim
        self.latent_dim = latent_dim

    def encode(self, obs, obs_prev, dt=0.002):
        velocity = (obs - obs_prev) / dt
        inp = torch.cat([obs, velocity], dim=-1)
        return self.encoder(inp)

    def encode_target(self, obs, obs_prev, dt=0.002):
        velocity = (obs - obs_prev) / dt
        inp = torch.cat([obs, velocity], dim=-1)
        with torch.no_grad():
            return self.target_encoder(inp)

    def decode_velocity(self, z):
        return self.velocity_decoder(z)

    def predict_residual(self, z, action):
        return self.predictor(torch.cat([z, action], dim=-1))
        
    def forward_latent(self, z, action):
        """One-step latent rollout: z_next = z + Δz"""
        delta_z = self.predict_residual(z, action)
        return z + delta_z

    @torch.no_grad()
    def update_target(self, tau=0.996):
        for tp, ep in zip(self.target_encoder.parameters(), self.encoder.parameters()):
            tp.data.mul_(tau).add_(ep.data, alpha=1 - tau)
