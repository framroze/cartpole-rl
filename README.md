# CartPole RL

CartPole balance using MuJoCo and PPO (Stable Baselines3).

## Setup
```bash
conda create -n dexterous python=3.11
conda activate dexterous
pip install mujoco stable-baselines3 gymnasium mujoco-playground
```

## Run
```bash
# Train policy
python src/train.py

# View trained policy in MuJoCo viewer
python src/view.py
```
