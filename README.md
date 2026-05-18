# 🛒 CartPole RL — Pole Balancing with Reinforcement Learning

<p align="center">
  <img src="https://img.shields.io/badge/Python-3.11-3776AB?style=for-the-badge&logo=python&logoColor=white" alt="Python"/>
  <img src="https://img.shields.io/badge/MuJoCo-3.8-00B4D8?style=for-the-badge" alt="MuJoCo"/>
  <img src="https://img.shields.io/badge/PPO-Stable--Baselines3-orange?style=for-the-badge" alt="PPO"/>
  <img src="https://img.shields.io/badge/License-MIT-yellow?style=for-the-badge" alt="License"/>
</p>
<p align="center">
  <b>A reinforcement learning agent that learns — entirely on its own through trial and error — to balance a pole on a moving cart using MuJoCo physics.</b>
</p>

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
| **PPO (Stable Baselines3)** | Reinforcement learning |
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

### 1️⃣ Clone the repository

```bash
git clone https://github.com/framroze/cartpole-rl.git
cd cartpole-rl
```

### 2️⃣ Create environment

```bash
conda create -n dexterous python=3.11
conda activate dexterous
```

### 3️⃣ Install dependencies

```bash
pip install mujoco stable-baselines3 gymnasium mujoco-playground
```

---

## 🚀 How to Run

### 🖥️ Train the policy

```bash
python src/main.py train
```

### 🖥️ View trained policy in MuJoCo viewer

```bash
python src/main.py view
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
| 📁 [`src/`](src/) | All source files — environment, training, and viewer |
| 📄 [`src/main.py`](src/main.py) | Entry point — `python src/main.py train` or `python src/main.py view` |
| 📄 [`src/cartpole_env.py`](src/cartpole_env.py) | MuJoCo CartPole Gymnasium environment |
| 📄 [`src/train.py`](src/train.py) | PPO training + evaluation |
| 📄 [`src/viewer.py`](src/viewer.py) | MuJoCo real-time policy viewer |

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

This project is licensed under the **MIT License** — see the [LICENSE](LICENSE) file for details.
