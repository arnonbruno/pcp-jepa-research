import torch
import numpy as np
import gymnasium as gym
from stable_baselines3 import SAC
from huggingface_sb3 import load_from_hub
import time

from src.models.pano import PANOVelocityPredictor
from src.envs.contact_dropout import ContactDropoutEnv
from src.utils.data import generate_pano_data
from src.utils.training import train_pano
from src.evaluation.stats import summarize_results

def main():
    import os
    os.environ["OMP_NUM_THREADS"] = "1"
    os.environ["MKL_NUM_THREADS"] = "1"
    torch.set_num_threads(1)
    
    device = torch.device('cpu')
    env_id = 'Ant-v4'
    
    print('Loading expert...')
    checkpoint = load_from_hub(repo_id='sb3/sac-Ant-v3', filename='sac-Ant-v3.zip')
    env = gym.make(env_id, use_contact_forces=True)
    model = SAC.load(checkpoint, env=env, device=device)
    
    print('Generating data...')
    data = generate_pano_data(model, n_episodes=1, env_id=env_id, device=device)
    
    print('Training PANO...')
    obs_dim = env.observation_space.shape[0]
    action_dim = env.action_space.shape[0]
    pano_model = PANOVelocityPredictor(obs_dim, action_dim, history_len=5).to(device)
    pano_model = train_pano(pano_model, data, n_epochs=1, device=device)
    
    print('Evaluating PANO...')
    eval_env = ContactDropoutEnv(env_id, dropout_duration=5, velocity_threshold=0.1)
    dt = eval_env.env.unwrapped.dt if hasattr(eval_env.env.unwrapped, 'dt') else 0.002
    
    rewards, state_errors = [], []
    for ep in range(1):
        obs, _ = eval_env.reset(seed=42 + ep)
        action_history = [np.zeros(action_dim) for _ in range(5)]
        total_reward = 0.0
        
        for step in range(1000):
            action, _ = model.predict(obs, deterministic=True)
            action_history.pop(0)
            action_history.append(action.copy())
            
            obs_raw, reward, term, trunc, info = eval_env.step(action)
            total_reward += reward
            
            if info['dropout_active']:
                obs_t = torch.tensor(obs, dtype=torch.float32, device=device).unsqueeze(0)
                ah_t = torch.tensor(np.array(action_history), dtype=torch.float32, device=device).unsqueeze(0)
                with torch.no_grad():
                    v_pred = pano_model(obs_t, ah_t).squeeze().cpu().numpy()
                    if np.any(np.isnan(v_pred)) or np.any(np.isinf(v_pred)):
                        v_pred = np.zeros_like(v_pred)
                
                dropout_step = info['dropout_step']
                frozen_obs = info['frozen_obs']
                obs = frozen_obs + v_pred * dt * (dropout_step + 1)
                
                state_errors.append(np.mean(np.abs(obs - info['true_obs'])))
            else:
                obs = obs_raw
                
            if term or trunc:
                break
        rewards.append(total_reward)
        
    res = summarize_results('PANO', np.array(rewards), np.array(state_errors) if state_errors else None)
    print("PANO Reward:", res['reward_mean'])

if __name__ == '__main__':
    main()
