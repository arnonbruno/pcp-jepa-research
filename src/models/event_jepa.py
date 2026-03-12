import torch
import torch.nn as nn
import torch.nn.functional as F
from src.models.jepa import StandardLatentJEPA

class EventConsistentJEPA(StandardLatentJEPA):
    """
    Event-Consistent JEPA explicitly models discrete events (like contacts)
    to prevent latent drift at boundaries.
    
    Theory:
    Naive JEPAs drift at contact boundaries because the latent dynamics
    assume continuous motion. Event-Consistent JEPA detects these boundaries
    in latent space and applies a learned constraint to the rollout, ensuring
    physics-consistent predictions. By explicitly identifying events, the model
    can apply corrective projections rather than letting the prediction drift
    uncontrolled.
    """
    def __init__(self, obs_dim=11, action_dim=3, latent_dim=64):
        super().__init__(obs_dim=obs_dim, action_dim=action_dim, latent_dim=latent_dim)
        
        # Module to identify discrete events from the latent state
        self.contact_detector = nn.Sequential(
            nn.Linear(latent_dim, 32),
            nn.ReLU(),
            nn.Linear(32, 1),
            nn.Sigmoid()
        )
        
        # Learned physics constraint to apply when an event is detected
        self.constraint_proj = nn.Linear(latent_dim, latent_dim)
        
    def detect_event(self, latent):
        """
        Returns event probability (0-1) based on latent discontinuity.
        """
        return self.contact_detector(latent)
        
    def constrain_prediction(self, latent, event_type):
        """
        Applies a physics constraint during rollout based on event probability.
        If event_type is high, it heavily constrains the latent representation to prevent drift.
        """
        constraint = self.constraint_proj(latent)
        return latent - event_type * constraint

    def forward_latent(self, z, action):
        """
        One-step latent rollout with event constraint:
        1. Predict continuous residual
        2. Detect event
        3. Apply constraint if event detected
        """
        delta_z = self.predict_residual(z, action)
        z_next_unconstrained = z + delta_z
        
        event_prob = self.detect_event(z_next_unconstrained)
        z_next_constrained = self.constrain_prediction(z_next_unconstrained, event_prob)
        
        return z_next_constrained

    def contrastive_event_loss(self, pred, target, events, margin=1.0):
        """
        Loss that aligns latent predictions with contact events using a contrastive approach.
        When event is 1, pred and target should be aligned (close).
        When event is 0, they can be further apart up to a margin.
        """
        dist = torch.norm(pred - target, p=2, dim=-1)
        events = events.view(-1)
        
        # Contrastive loss:
        # Pull together when event occurs
        loss_event = events * dist.pow(2)
        # Push apart when event does not occur, up to a margin
        loss_no_event = (1 - events) * F.relu(margin - dist).pow(2)
        
        return (loss_event + loss_no_event).mean()
