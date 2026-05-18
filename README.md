# 🛒 CartPole RL — Pole Balancing with PPO

[![Python](https://img.shields.io/badge/Python-3.11-3776AB?style=for-the-badge&logo=python&logoColor=white)](https://python.org)
[![MuJoCo](https://img.shields.io/badge/MuJoCo-3.8-00B4D8?style=for-the-badge)](https://mujoco.org)
[![Algorithm](https://img.shields.io/badge/PPO-Stable--Baselines3-orange?style=for-the-badge)](https://stable-baselines3.readthedocs.io)
[![License](https://img.shields.io/badge/License-MIT-yellow?style=for-the-badge)](LICENSE)

**A reinforcement learning agent that learns to balance a pole on a moving cart using MuJoCo physics and PPO.**

---

## ✨ What It Does

The trained agent controls a cart moving left and right on a track to keep a pole balanced upright:

1. 🧠 Observes cart position, cart velocity, pole angle, and pole velocity
2. ⚡ Decides whether to push the cart left or right
3. ⚖️ Keeps the pole balanced indefinitely
4. 🎯 Scores **499 / 500 steps** — near perfect balance

---

## 🧰 Tech Stack

| Component | Used For |
|-----------|----------|
| **MuJoCo 3.8** | Physics simulation |
| **PPO (Stable Baselines3)** | Reinforcement learning algorithm |
| **Gymnasium** | Environment interface |
| **DeepMind Control Suite** | Cartpole XML model |
| **Python 3.11** | Implementation |

---

## 📋 Requirements

- Python 3.11
- Ubuntu / WSL2
- Conda (recommended)

---

## ⚙️ Installation

### 1️⃣ Create environment

```bash
conda create -n dexterous python=3.11
conda activate dexterous
```

### 2️⃣ Install dependencies

```bash
pip install -r requirements.txt
```

---

## 🚀 How to Run

### 🖥️ Train the policy

```bash
python main.py train
```

### 🖥️ View trained policy in MuJoCo viewer

```bash
python main.py view
```

> Double-click the pole to grab and disturb it — watch the agent recover!

---

## 📊 Results

| Policy | Steps | Score |
|--------|-------|-------|
| Random | ~20 | 4% |
| **PPO (ours)** | **499 / 500** | **~100%** ✅ |

---

## 📂 Project Structure

| Path | Description |
|------|-------------|
| `main.py` | Entry point — train or view |
| `requirements.txt` | Dependencies |
| `config/ppo_config.yaml` | PPO hyperparameters |
| `src/env/cartpole.py` | MuJoCo Gymnasium environment |
| `src/training/train.py` | PPO training + evaluation |
| `src/visualization/viewer.py` | MuJoCo real-time viewer |

---

## 🧠 Concepts Demonstrated

- ✅ Custom Gymnasium environment with MuJoCo physics
- ✅ PPO reinforcement learning (same algorithm as OpenAI Dactyl)
- ✅ Parallel environment training (16 envs)
- ✅ Sim-to-real physics tuning (damping, timestep)
- ✅ Real-time policy visualization in MuJoCo viewer

---

## 👤 Author

**framroze**
🌐 GitHub: [@framroze](https://github.com/framroze)

---

## 📜 License

This project is licensed under the **MIT License**.
