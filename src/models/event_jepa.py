import torch
import torch.nn as nn
import torch.nn.functional as F
from src.models.jepa import StandardLatentJEPA


class EventConsistentJEPA(StandardLatentJEPA):
    """
    Event-Consistent JEPA augments latent rollout with contact supervision.

    This variant uses MuJoCo-derived contact labels, normal-force targets,
    and a complementarity penalty based on contact distance to regularize
    the latent space around contact events.
    """
    def __init__(self, obs_dim=11, action_dim=3, latent_dim=64):
        super().__init__(obs_dim=obs_dim, action_dim=action_dim, latent_dim=latent_dim)

        # Predict whether the next latent state corresponds to a contact event.
        self.contact_detector = nn.Sequential(
            nn.Linear(latent_dim, 32),
            nn.ReLU(),
            nn.Linear(32, 1),
        )

        # Predict non-negative normal impulse magnitude from the next latent.
        self.impulse_head = nn.Sequential(
            nn.Linear(latent_dim, 32),
            nn.ReLU(),
            nn.Linear(32, 1),
        )

        # Contact-conditioned correction applied during rollout.
        self.constraint_proj = nn.Linear(latent_dim, latent_dim)

    def detect_event_logits(self, latent):
        """
        Returns raw event logits for BCE supervision.
        """
        return self.contact_detector(latent)

    def detect_event(self, latent):
        """
        Returns event probability (0-1) based on latent discontinuity.
        """
        return torch.sigmoid(self.detect_event_logits(latent))

    def predict_contact_impulse(self, latent):
        """
        Predict a non-negative normal impulse estimate.
        """
        return F.softplus(self.impulse_head(latent))

    def constrain_prediction(self, latent, event_type, contact_impulse=None):
        """
        Applies a contact-conditioned latent correction.

        The correction is scaled by both the inferred event probability and the
        predicted normal impulse, so free-flight states remain mostly
        unconstrained while contact states receive stronger updates.
        """
        if contact_impulse is None:
            contact_impulse = self.predict_contact_impulse(latent)
        constraint = self.constraint_proj(latent)
        return latent - event_type * contact_impulse * constraint

    def forward_latent(self, z, action, return_aux=False):
        """
        One-step latent rollout with event constraint:
        1. Predict continuous residual
        2. Detect event
        3. Predict normal impulse
        4. Apply constraint if event detected
        """
        delta_z = self.predict_residual(z, action)
        z_next_unconstrained = z + delta_z

        event_logits = self.detect_event_logits(z_next_unconstrained)
        event_prob = torch.sigmoid(event_logits)
        contact_impulse = self.predict_contact_impulse(z_next_unconstrained)
        z_next_constrained = self.constrain_prediction(
            z_next_unconstrained, event_prob, contact_impulse
        )

        if return_aux:
            return z_next_constrained, {
                'event_logits': event_logits,
                'event_prob': event_prob,
                'contact_impulse': contact_impulse,
                'z_next_unconstrained': z_next_unconstrained,
            }
        return z_next_constrained

    def event_supervision_loss(self, event_logits, events):
        """
        Supervise contact-event detection with binary labels.
        """
        return F.binary_cross_entropy_with_logits(
            event_logits.view(-1), events.view(-1)
        )

    def contact_impulse_loss(self, contact_impulse, target_impulse):
        """
        Match predicted impulse to the MuJoCo normal impulse proxy.
        """
        target_impulse = torch.clamp(target_impulse, min=0.0)
        return F.mse_loss(contact_impulse, target_impulse)

    def complementarity_loss(self, contact_impulse, contact_distance, events=None):
        """
        Enforce normal-force complementarity: lambda_n >= 0 and lambda_n * gap = 0.

        `contact_distance` is the minimum MuJoCo contact distance. Positive
        distance means separation; negative distance means penetration.
        """
        gap = torch.relu(contact_distance)
        penetration = torch.relu(-contact_distance)
        loss = (contact_impulse * gap).mean() + penetration.mean()

        if events is not None:
            free_flight = 1.0 - events
            loss = loss + (free_flight * contact_impulse).mean()

        return loss

    def contact_constraint_loss(self, contact_impulse, contact_distance, events=None):
        """
        Aggregate contact penalties.
        """
        return self.complementarity_loss(
            contact_impulse=contact_impulse,
            contact_distance=contact_distance,
            events=events,
        )

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
