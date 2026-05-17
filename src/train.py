import sys, os
sys.path.insert(0, os.path.dirname(__file__))
os.environ['MUJOCO_GL'] = 'egl'

import numpy as np
from mujoco_cartpole_env import MujocoCartpoleEnv
from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env

SAVE_PATH = os.path.join(os.path.dirname(__file__), "../results/cartpole_ppo.zip")
os.makedirs(os.path.dirname(SAVE_PATH), exist_ok=True)

print("=== MuJoCo CartPole — PPO Training ===\n")

vec_env = make_vec_env("MujocoCartpole-v0", n_envs=16)

model = PPO(
    "MlpPolicy", vec_env,
    verbose=1,
    learning_rate=3e-4,
    n_steps=512,
    batch_size=64,
    n_epochs=10,
    gamma=0.99,
    device='cpu',
)

print("Training 300,000 steps...\n")
model.learn(total_timesteps=300_000)
model.save(SAVE_PATH)
print(f"\nSaved: {SAVE_PATH}")

print("\nEvaluating...")
env = MujocoCartpoleEnv()
rewards = []
for ep in range(10):
    obs, _ = env.reset(seed=ep)
    total  = 0
    for _ in range(500):
        action, _ = model.predict(obs, deterministic=True)
        obs, r, done, trunc, _ = env.step(action)
        total += r
        if done or trunc: break
    rewards.append(total)
    print(f"  ep {ep+1}: {total:.0f} steps")
print(f"\nMean: {np.mean(rewards):.0f}/500")
