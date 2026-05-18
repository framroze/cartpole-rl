"""
PPO Training for MuJoCo CartPole.

Algorithm : PPO (Proximal Policy Optimization)
Network   : MLP [64, 64]
Envs      : 16 parallel
Steps     : 300,000
"""
import os
import sys
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../.."))
os.environ['MUJOCO_GL'] = 'egl'

from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
from src.env.cartpole import CartPoleEnv

SAVE_PATH = os.path.join(
    os.path.dirname(__file__), "../../results/cartpole_ppo.zip"
)


def train(total_timesteps=300_000, n_envs=16):
    os.makedirs(os.path.dirname(SAVE_PATH), exist_ok=True)

    print("=" * 50)
    print("  CartPole — PPO Training")
    print("=" * 50)
    print(f"\n  Environments : {n_envs} parallel")
    print(f"  Total steps  : {total_timesteps:,}")
    print(f"  Network      : MLP [64, 64]\n")

    vec_env = make_vec_env("MujocoCartpole-v0", n_envs=n_envs)

    model = PPO(
        "MlpPolicy", vec_env,
        verbose=1,
        learning_rate=3e-4,
        n_steps=512,
        batch_size=64,
        n_epochs=10,
        gamma=0.99,
        device="cpu",
        policy_kwargs=dict(net_arch=[64, 64]),
    )

    model.learn(total_timesteps=total_timesteps)
    model.save(SAVE_PATH)
    print(f"\n  Model saved: {SAVE_PATH}")
    return model


def evaluate(model, n_episodes=10):
    env     = CartPoleEnv()
    rewards = []

    print(f"\n  Evaluating {n_episodes} episodes...")
    for ep in range(n_episodes):
        obs, _ = env.reset(seed=ep)
        total  = 0
        for _ in range(500):
            action, _ = model.predict(obs, deterministic=True)
            obs, r, done, trunc, _ = env.step(action)
            total += r
            if done or trunc:
                break
        rewards.append(total)
        print(f"    Episode {ep+1:2d}: {int(total):>3} steps")

    print(f"\n  Mean : {np.mean(rewards):.0f} / 500")
    print(f"  Best : {np.max(rewards):.0f} / 500")
    return rewards
