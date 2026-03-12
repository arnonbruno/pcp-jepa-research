import torch
import torch.nn as nn

class PANOVelocityPredictor(nn.Module):
    """
    PANO velocity predictor: learns to estimate state velocity from
    current observation + action history.

    Improvements added:
    1) Layer Normalization: Stabilizes training and gradients across layers.
    2) Kaiming Initialization: Provides proper weight scaling for ReLU networks.
    3) Residual Connections: Optional skip connections to help gradients flow in deeper networks.
    4) Dropout: Optional regularization to prevent overfitting on trajectories.

    During dropout: obs_est = frozen_obs + predicted_velocity * dt * steps
    """
    def __init__(self, obs_dim, action_dim, history_len=5, hidden_dim=128, 
                 depth=3, use_layer_norm=True, dropout=0.0, use_residual=False):
        super().__init__()
        self.history_len = history_len
        self.use_residual = use_residual
        
        input_dim = obs_dim + history_len * action_dim
        
        # Input layer
        layers = [nn.Linear(input_dim, hidden_dim)]
        if use_layer_norm:
            layers.append(nn.LayerNorm(hidden_dim))
        layers.append(nn.ReLU())
        if dropout > 0:
            layers.append(nn.Dropout(dropout))
            
        self.input_layer = nn.Sequential(*layers)
        
        # Hidden layers
        self.hidden_layers = nn.ModuleList()
        for _ in range(depth - 1):
            h_layer = [nn.Linear(hidden_dim, hidden_dim)]
            if use_layer_norm:
                h_layer.append(nn.LayerNorm(hidden_dim))
            h_layer.append(nn.ReLU())
            if dropout > 0:
                h_layer.append(nn.Dropout(dropout))
            self.hidden_layers.append(nn.Sequential(*h_layer))
            
        # Output layer
        self.output_layer = nn.Linear(hidden_dim, obs_dim)
        
        # Apply Kaiming initialization
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            nn.init.kaiming_normal_(m.weight, mode='fan_in', nonlinearity='relu')
            if m.bias is not None:
                nn.init.zeros_(m.bias)
        elif isinstance(m, nn.LayerNorm):
            nn.init.ones_(m.weight)
            nn.init.zeros_(m.bias)

    def forward(self, obs, action_history):
        if action_history.dim() == 2:
            action_history = action_history.unsqueeze(0)
        action_flat = action_history.reshape(action_history.shape[0], -1)
        inp = torch.cat([obs, action_flat], dim=-1)
        
        x = self.input_layer(inp)
        
        for layer in self.hidden_layers:
            if self.use_residual:
                x = x + layer(x)
            else:
                x = layer(x)
                
        return self.output_layer(x)
