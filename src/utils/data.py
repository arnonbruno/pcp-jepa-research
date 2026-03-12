import torch
import numpy as np
import gymnasium as gym

from src.envs.contact_dropout import ContactDropoutEnv

def generate_jepa_data(env, sac_model, n_transitions, dt, device='cuda', seed=42):
    """Generate exactly n_transitions from an environment using the SAC oracle."""
    if isinstance(device, str):
        device = torch.device(device if torch.cuda.is_available() else 'cpu')
        
    data = {
        'obs': [],
        'obs_prev': [],
        'action': [],
        'obs_next': [],
        'obs_prev_next': [],
        'episode_id': [],
        'timestep': [],
    }
    count = 0
    ep = 0
    while count < n_transitions:
        obs, _ = env.reset(seed=seed + ep)
        ep += 1
        obs_prev = obs.copy()
        for step in range(1000):
            action, _ = sac_model.predict(obs, deterministic=True)
            data['obs'].append(obs)
            data['obs_prev'].append(obs_prev)
            data['action'].append(action)
            obs_next, _, term, trunc, _ = env.step(action)
            data['obs_next'].append(obs_next)
            data['obs_prev_next'].append(obs)
            data['episode_id'].append(ep - 1)
            data['timestep'].append(step)
            count += 1
            if count >= n_transitions:
                break
            obs_prev = obs.copy()
            obs = obs_next
            if term or trunc:
                break
    for k in data:
        dtype = torch.long if k in {'episode_id', 'timestep'} else torch.float32
        data[k] = torch.tensor(np.array(data[k][:n_transitions]), dtype=dtype, device=device)
    return data

def generate_jepa_data_episodes(sac_model, n_episodes=200, env_id='Hopper-v4', device='cuda', seed=42):
    """Generate training data for a fixed number of episodes."""
    if isinstance(device, str):
        device = torch.device(device if torch.cuda.is_available() else 'cpu')
        
    if env_id == 'Ant-v4':
        env = gym.make(env_id, use_contact_forces=True)
    else:
        env = gym.make(env_id)
    data = {
        'obs': [], 'obs_prev': [], 'action': [],
        'obs_next': [], 'obs_prev_next': [],
        'episode_id': [], 'timestep': [],
    }
    
    for ep in range(n_episodes):
        obs, _ = env.reset(seed=seed + ep)
        obs_prev = obs.copy()
        
        for step in range(1000):
            action, _ = sac_model.predict(obs, deterministic=True)
            
            data['obs'].append(obs)
            data['obs_prev'].append(obs_prev)
            data['action'].append(action)
            
            obs_next, _, term, trunc, _ = env.step(action)
            
            data['obs_next'].append(obs_next)
            data['obs_prev_next'].append(obs)
            data['episode_id'].append(ep)
            data['timestep'].append(step)
            
            obs_prev = obs.copy()
            obs = obs_next
            
            if term or trunc:
                break
    
    env.close()
    
    for k in data:
        dtype = torch.long if k in {'episode_id', 'timestep'} else torch.float32
        data[k] = torch.tensor(np.array(data[k]), dtype=dtype, device=device)
    
    print(f"Generated {len(data['obs'])} transitions")
    return data

def generate_pano_data(sac_model, n_episodes=300, history_len=5, env_id='Hopper-v4', device='cpu', seed=42):
    """Generate training data for the forward PANO velocity predictor."""
    if isinstance(device, str):
        device = torch.device(device if torch.cuda.is_available() else 'cpu')
        
    if env_id == 'Ant-v4':
        env = gym.make(env_id, use_contact_forces=True)
    else:
        env = gym.make(env_id)
    obs_dim = env.observation_space.shape[0]
    action_dim = env.action_space.shape[0]

    data = {'obs': [], 'action_history': [], 'velocity': []}
    dt = env.unwrapped.dt if hasattr(env.unwrapped, 'dt') else 0.002

    for ep in range(n_episodes):
        obs, _ = env.reset(seed=seed + ep)
        action_history = [np.zeros(action_dim) for _ in range(history_len)]

        for step in range(1000):
            action, _ = sac_model.predict(obs, deterministic=True)
            action_history.pop(0)
            action_history.append(action.copy())

            obs_next, _, term, trunc, _ = env.step(action)
            velocity = (obs_next - obs) / dt

            data['obs'].append(obs.copy())
            data['action_history'].append(np.array(action_history))
            data['velocity'].append(velocity)

            obs = obs_next
            if term or trunc:
                break

    env.close()
    for k in data:
        data[k] = torch.tensor(np.array(data[k]), dtype=torch.float32, device=device)

    print(f"Generated {len(data['obs'])} transitions from {n_episodes} episodes")
    return data


def generate_event_jepa_data(
    sac_model,
    n_episodes=300,
    env_id='Hopper-v4',
    velocity_threshold=0.1,
    device='cuda',
    seed=42,
):
    """
    Generate transition data plus MuJoCo contact semantics for EventConsistentJEPA.

    We use the contact wrapper with `dropout_duration=0` so observations stay
    uncorrupted while the wrapper still exposes ground-truth contact metadata.
    """
    if isinstance(device, str):
        device = torch.device(device if torch.cuda.is_available() else 'cpu')

    env = ContactDropoutEnv(
        env_id=env_id,
        dropout_duration=0,
        velocity_threshold=velocity_threshold,
    )
    dt = env.env.unwrapped.dt if hasattr(env.env.unwrapped, 'dt') else 0.002

    data = {
        'obs': [],
        'obs_prev': [],
        'action': [],
        'obs_next': [],
        'obs_prev_next': [],
        'episode_id': [],
        'timestep': [],
        'contact': [],
        'contact_force': [],
        'contact_impulse': [],
        'contact_distance': [],
    }

    for ep in range(n_episodes):
        obs, _ = env.reset(seed=seed + ep)
        obs_prev = obs.copy()

        for step in range(1000):
            action, _ = sac_model.predict(obs, deterministic=True)
            obs_next, _, term, trunc, info = env.step(action)
            true_next = info['true_obs'].copy()

            contact_force = float(info.get('contact_normal_force_max', info.get('contact_force_max', 0.0)))
            contact_distance = float(info.get('contact_distance_min', np.inf))
            if not np.isfinite(contact_distance):
                contact_distance = 1.0

            data['obs'].append(obs.copy())
            data['obs_prev'].append(obs_prev.copy())
            data['action'].append(action.copy())
            data['obs_next'].append(true_next)
            data['obs_prev_next'].append(obs.copy())
            data['episode_id'].append(ep)
            data['timestep'].append(step)
            data['contact'].append([1.0 if info.get('contact_detected', False) else 0.0])
            data['contact_force'].append([contact_force])
            data['contact_impulse'].append([contact_force * dt])
            data['contact_distance'].append([contact_distance])

            obs_prev = obs.copy()
            obs = true_next
            if term or trunc:
                break

    env.close()

    for key in data:
        dtype = torch.long if key in {'episode_id', 'timestep'} else torch.float32
        data[key] = torch.tensor(np.array(data[key]), dtype=dtype, device=device)

    print(f"Generated {len(data['obs'])} event-aware transitions from {n_episodes} episodes")
    return data
