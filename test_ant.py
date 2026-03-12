import gymnasium as gym
import time
from stable_baselines3 import SAC
from huggingface_sb3 import load_from_hub

print("Loading model...")
checkpoint = load_from_hub(repo_id='sb3/sac-Ant-v3', filename='sac-Ant-v3.zip')
env = gym.make('Ant-v4', use_contact_forces=True)
model = SAC.load(checkpoint, env=env)

print("Running predict...")
obs, _ = env.reset(seed=42)
start_time = time.time()
for step in range(1000):
    action, _ = model.predict(obs, deterministic=True)
    obs, _, term, trunc, _ = env.step(action)
    if term or trunc:
        obs, _ = env.reset()

print(f"Done in {time.time() - start_time:.2f}s")
