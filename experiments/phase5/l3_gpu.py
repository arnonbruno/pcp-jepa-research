#!/usr/bin/env python3
"""
PHASE 5: L3 - Impact Consistency with GPU

Implements explicit impact map M(b_pre, theta) -> b_post
with L3a (loss-only coupling).
"""

import torch
import torch.nn as nn
import numpy as np

device = torch.device('cuda')
print(f"Using: {device}")


# ============================================================================
# ENVIRONMENT + DATA
# ============================================================================

class BouncingBall:
    def __init__(self, e=0.8):
        self.e = e
        
    def reset(self, seed):
        np.random.seed(seed)
        self.x = [0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5][seed % 7]
        self.v = 0.0
        
    def step(self, a):
        a = np.clip(a, -2, 2)
        self.v += (-9.81 + a) * 0.05
        self.x += self.v * 0.05
        bounced = False
        if self.x < 0:
            self.x = -self.x * self.e
            self.v = -self.v * self.e
            bounced = True
        elif self.x > 3:
            self.x = 3 - (self.x - 3) * self.e
            self.v = -self.v * self.e
            bounced = True
        return bounced


def generate_batched_data(n_episodes, e, seed):
    """Generate data in batched format for GPU."""
    np.random.seed(seed)
    env = BouncingBall(e)
    
    # Storage: (episode, time, data)
    all_obs = []      # (n, 30, 1)
    all_acts = []     # (n, 30, 1)
    all_bounces = []  # (n, 30)
    all_pre = []      # indices of pre-impact
    all_post = []      # indices of post-impact
    
    for ep in range(n_episodes):
        env.reset(seed=ep + seed)
        x = env.x
        v = env.v
        
        obs_list = []
        act_list = []
        bounce_list = []
        
        for step in range(30):
            # Obs: partial (position only)
            obs = np.array([x], dtype=np.float32)
            
            # Action: expert PD
            a = 1.5 * (2.0 - x) + (-2.0) * (-v)
            a = np.clip(a, -2, 2)
            
            # Step
            bounced = env.step(a)
            x = env.x
            v = env.v
            
            obs_list.append(obs)
            act_list.append(np.array([a], dtype=np.float32))
            bounce_list.append(1.0 if bounced else 0.0)
        
        # Find impact pairs
        pre_idx = []
        post_idx = []
        for i in range(1, 29):
            if bounce_list[i] == 1.0:
                pre_idx.append(i - 1)
                post_idx.append(i + 1)
        
        all_obs.append(np.array(obs_list))
        all_acts.append(np.array(act_list))
        all_bounces.append(np.array(bounce_list))
        
        if len(pre_idx) > 0:
            all_pre.append((ep, pre_idx))
            all_post.append((ep, post_idx))
    
    # Convert to tensors
    obs_tensor = torch.tensor(np.array(all_obs), dtype=torch.float32).to(device)
    act_tensor = torch.tensor(np.array(all_acts), dtype=torch.float32).to(device)
    bounce_tensor = torch.tensor(np.array(all_bounces), dtype=torch.float32).to(device)
    
    return obs_tensor, act_tensor, bounce_tensor, all_pre, all_post


# ============================================================================
# MODEL
# ============================================================================

class ImpactModel(nn.Module):
    """
    JEPA with Impact Map:
    - Encoder: obs -> latent
    - GRU: belief update  
    - Impact Map: predicts belief change at impact
    - Controller: linear feedback
    """
    def __init__(self, obs_dim=1, action_dim=1, latent_dim=8, belief_dim=16):
        super().__init__()
        
        # Encoder
        self.encoder = nn.Linear(obs_dim, latent_dim)
        
        # GRU
        self.gru = nn.GRUCell(latent_dim + action_dim, belief_dim)
        
        # Impact map: (belief, theta) -> delta_belief
        # theta = restitution (scalar)
        self.impact_net = nn.Sequential(
            nn.Linear(belief_dim + 1, 16),
            nn.Tanh(),
            nn.Linear(16, belief_dim)
        )
        
        # Controller
        self.controller = nn.Linear(belief_dim, action_dim)
    
    def forward(self, obs, acts, bounces, thetas=None):
        """
        Forward pass through trajectory.
        Returns: beliefs, pre_beliefs, post_beliefs (for impact pairs)
        """
        batch_size, T, _ = obs.shape
        
        # Init
        beliefs = []
        belief = torch.zeros(batch_size, self.gru.hidden_size, device=device)
        
        # Impact pairs storage
        pre_beliefs = []
        post_beliefs = []
        
        for t in range(T):
            # Encode
            z = torch.tanh(self.encoder(obs[:, t]))
            
            # Previous action
            a_prev = acts[:, t-1] if t > 0 else torch.zeros(batch_size, 1, device=device)
            
            # GRU step
            x = torch.cat([z, a_prev], dim=1)
            belief_new = torch.tanh(self.gru(x, belief))
            
            # Impact handling
            if thetas is not None and bounces is not None and t > 0:
                # Apply impact map where bounces occurred
                impact_mask = (bounces[:, t] == 1.0).float().unsqueeze(-1)
                theta = thetas.unsqueeze(-1).expand(-1, 1)
                
                # Predict delta
                delta = torch.tanh(self.impact_net(torch.cat([belief, theta], dim=1)))
                
                # Apply impact where bounce occurred
                belief = (1 - impact_mask) * belief_new + impact_mask * (belief + delta)
            else:
                belief = belief_new
            
            beliefs.append(belief)
        
        beliefs = torch.stack(beliefs, dim=1)  # (B, T, D)
        
        return beliefs
    
    def get_impacts(self, obs, acts, pre_indices, post_indices):
        """Get belief at pre and post impact for consistency loss."""
        beliefs = self.forward(obs, acts, None)
        
        pre_beliefs = []
        post_beliefs = []
        
        for (ep, pre_list), (_, post_list) in zip(pre_indices, post_indices):
            for pi, po in zip(pre_list, post_list):
                pre_beliefs.append(beliefs[ep, pi])
                post_beliefs.append(beliefs[ep, po])
        
        if len(pre_beliefs) == 0:
            return None, None
        
        return torch.stack(pre_beliefs), torch.stack(post_beliefs)


def train_l3(n_episodes=500, epochs=100, lr=0.001):
    """Train with L1 (prediction) + L3 (impact consistency)."""
    
    print("Generating data...")
    obs, acts, bounces, pre_idx, post_idx = generate_batched_data(n_episodes, 0.8, 42)
    print(f"  Obs shape: {obs.shape}")
    print(f"  Impact pairs: {len(pre_idx)}")
    
    # Model
    model = ImpactModel().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    
    # Restitution for impact model
    thetas = torch.full((n_episodes,), 0.8, device=device)
    
    print(f"Training L3 ({epochs} epochs)...")
    
    for epoch in range(epochs):
        # Forward pass
        beliefs = model.forward(obs, acts, bounces, thetas)
        
        # L1: Prediction loss (smooth dynamics)
        pred_loss = 0
        for t in range(beliefs.shape[1] - 1):
            pred_loss += ((beliefs[:, t+1] - beliefs[:, t]) ** 2).mean()
        pred_loss = pred_loss / (beliefs.shape[1] - 1)
        
        # L3: Impact consistency
        impact_loss = 0
        if len(pre_idx) > 0:
            pre_b, post_b = model.get_impacts(obs, acts, pre_idx, post_idx)
            
            # Predict delta and compare
            theta_exp = torch.full((pre_b.shape[0], 1), 0.8, device=device)
            delta_pred = torch.tanh(model.impact_net(torch.cat([pre_b, theta_exp], dim=1)))
            
            # Loss: predicted delta should match actual delta
            impact_loss = ((post_b - pre_b - delta_pred) ** 2).mean()
        
        # L4: Controller imitation
        ctrl_loss = 0
        for t in range(beliefs.shape[1]):
            a_pred = torch.tanh(model.controller(beliefs[:, t]))
            ctrl_loss += ((a_pred - acts[:, t]) ** 2).mean()
        ctrl_loss = ctrl_loss / beliefs.shape[1]
        
        # Total
        loss = 0.3 * pred_loss + 1.0 * impact_loss + 0.5 * ctrl_loss
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        if (epoch + 1) % 20 == 0:
            print(f"  Epoch {epoch+1}: loss={loss.item():.4f}, impact={impact_loss.item():.4f}")
    
    return model, obs, acts


def evaluate(model, n_test=200, e=0.8, seed=0):
    """Evaluate: just use the controller head learned during training."""
    model.eval()
    
    # Evaluate directly using model's controller
    print("  Evaluating...")
    env = BouncingBall(e)
    successes = 0
    
    with torch.no_grad():
        for ep in range(n_test):
            env.reset(seed=ep + seed + 1000)
            x = env.x
            v = env.v
            
            belief = torch.zeros(1, model.gru.hidden_size, device=device)
            last_a = torch.zeros(1, 1, device=device)
            
            for step in range(30):
                # Partial obs: (1, 1)
                obs_t = torch.tensor([[[x]]], dtype=torch.float32).to(device)
                
                # Encoder: (1, 1) -> (1, 8)
                z = torch.tanh(model.encoder(obs_t.squeeze(0).unsqueeze(0)))
                
                # GRU: (1, 9) -> (1, 16)
                belief = torch.tanh(model.gru(torch.cat([z, last_a.squeeze(0).unsqueeze(0)], dim=1), belief.squeeze(0).unsqueeze(0))).unsqueeze(0))
                
                a = model.controller(belief.squeeze(0)).item()
                a = float(np.clip(a, -2, 2))
                last_a = torch.tensor([[[a]]], device=device)
                
                env.step(a)
                x = env.x
                
                if abs(x - 2.0) < 0.3:
                    successes += 1
                    break
    
    return successes / n_test


def main():
    print("="*70)
    print("PHASE 5: L3 - Impact Consistency (GPU)")
    print("="*70)
    
    # Train L3
    model, obs, acts = train_l3(n_episodes=500, epochs=100, lr=0.001)
    
    # Evaluate
    print("\nEvaluating...")
    rate = evaluate(model, n_test=200, e=0.8, seed=42)
    
    print("\n" + "="*70)
    print("RESULTS")
    print("="*70)
    print(f"PD (full state):   71.0%")
    print(f"PD (partial):      56.5%")
    print(f"L3 + L4-A:        {rate:.1%}")
    
    gap = (rate - 0.565) / (0.71 - 0.565)
    print(f"\nGap closed: {gap:.1%}")
    
    if rate > 0.60:
        print("\n✓ L3 works! Impact consistency makes belief control-sufficient.")
    else:
        print("\n✗ L3 didn't close gap.")


if __name__ == '__main__':
    main()
