import torch
import numpy as np
import gymnasium as gym

def generate_jepa_data(env, sac_model, n_transitions, dt, device='cuda'):
    """Generate exactly n_transitions from an environment using the SAC oracle."""
    if isinstance(device, str):
        device = torch.device(device if torch.cuda.is_available() else 'cpu')
        
    data = {'obs': [], 'obs_prev': [], 'action': [], 'obs_next': [], 'obs_prev_next': []}
    count = 0
    while count < n_transitions:
        obs, _ = env.reset()
        obs_prev = obs.copy()
        for _ in range(1000):
            action, _ = sac_model.predict(obs, deterministic=True)
            data['obs'].append(obs)
            data['obs_prev'].append(obs_prev)
            data['action'].append(action)
            obs_next, _, term, trunc, _ = env.step(action)
            data['obs_next'].append(obs_next)
            data['obs_prev_next'].append(obs)
            count += 1
            if count >= n_transitions:
                break
            obs_prev = obs.copy()
            obs = obs_next
            if term or trunc:
                break
    for k in data:
        data[k] = torch.tensor(np.array(data[k][:n_transitions]), dtype=torch.float32, device=device)
    return data

def generate_jepa_data_episodes(sac_model, n_episodes=200, env_id='Hopper-v4', device='cuda'):
    """Generate training data for a fixed number of episodes."""
    if isinstance(device, str):
        device = torch.device(device if torch.cuda.is_available() else 'cpu')
        
    env = gym.make(env_id)
    data = {
        'obs': [], 'obs_prev': [], 'action': [],
        'obs_next': [], 'obs_prev_next': []
    }
    
    for ep in range(n_episodes):
        obs, _ = env.reset()
        obs_prev = obs.copy()
        
        for step in range(1000):
            action, _ = sac_model.predict(obs, deterministic=True)
            
            data['obs'].append(obs)
            data['obs_prev'].append(obs_prev)
            data['action'].append(action)
            
            obs_next, _, term, trunc, _ = env.step(action)
            
            data['obs_next'].append(obs_next)
            data['obs_prev_next'].append(obs)
            
            obs_prev = obs.copy()
            obs = obs_next
            
            if term or trunc:
                break
    
    env.close()
    
    for k in data:
        data[k] = torch.tensor(np.array(data[k]), dtype=torch.float32, device=device)
    
    print(f"Generated {len(data['obs'])} transitions")
    return data

def generate_pano_data(sac_model, n_episodes=300, history_len=5, env_id='Hopper-v4', device='cuda'):
    """Generate training data for PANO velocity predictor."""
    if isinstance(device, str):
        device = torch.device(device if torch.cuda.is_available() else 'cpu')
        
    env = gym.make(env_id)
    obs_dim = env.observation_space.shape[0]
    action_dim = env.action_space.shape[0]

    data = {'obs': [], 'action_history': [], 'velocity': []}
    dt = env.unwrapped.dt if hasattr(env.unwrapped, 'dt') else 0.002

    for ep in range(n_episodes):
        obs, _ = env.reset()
        obs_prev = obs.copy()
        action_history = [np.zeros(action_dim) for _ in range(history_len)]

        for step in range(1000):
            action, _ = sac_model.predict(obs, deterministic=True)
            action_history.pop(0)
            action_history.append(action.copy())

            obs_next, _, term, trunc, _ = env.step(action)
            velocity = (obs - obs_prev) / dt

            data['obs'].append(obs.copy())
            data['action_history'].append(np.array(action_history))
            data['velocity'].append(velocity)

            obs_prev = obs.copy()
            obs = obs_next
            if term or trunc:
                break

    env.close()
    for k in data:
        data[k] = torch.tensor(np.array(data[k]), dtype=torch.float32, device=device)

    print(f"Generated {len(data['obs'])} transitions from {n_episodes} episodes")
    return data
