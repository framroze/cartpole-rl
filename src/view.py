import sys, os
sys.path.insert(0, os.path.dirname(__file__))
os.environ['MUJOCO_GL'] = 'egl'

import mujoco
import mujoco.viewer
import numpy as np
import time
from stable_baselines3 import PPO
from mujoco_cartpole_env import XML_PATH, assets

SAVE_PATH = os.path.join(os.path.dirname(__file__), "../results/cartpole_ppo.zip")

print("Loading trained CartPole policy...")
model = PPO.load(SAVE_PATH)

mj_model = mujoco.MjModel.from_xml_string(XML_PATH.read_text(), assets)
mj_model.opt.timestep = 0.005

# Real-life physics values
mj_model.dof_damping[0] = 0.05    # cart: smooth rolling (was 5e-4)
mj_model.dof_damping[1] = 0.01    # pole pivot: light friction (was 2e-6)

mj_data  = mujoco.MjData(mj_model)
mj_data.qpos[0] = 0.0
mj_data.qpos[1] = 0.05
mj_data.qvel[:] = 0.0
mujoco.mj_forward(mj_model, mj_data)

print("Viewer launching! Grab pole to disturb!\n")

POLICY_FREQ    = 4
current_action = 0.0
physics_step   = 0

with mujoco.viewer.launch_passive(mj_model, mj_data) as v:
    v.cam.azimuth   = 0
    v.cam.elevation = -15
    v.cam.distance  = 4.0
    v.cam.lookat[:] = [0.0, 0.0, 1.2]

    while v.is_running():
        cp = float(mj_data.qpos[0])
        cv = float(mj_data.qvel[0])
        pa = float(mj_data.qpos[1])
        pv = float(mj_data.qvel[1])

        if physics_step % POLICY_FREQ == 0:
            if abs(cp) > 1.3:
                current_action = -np.sign(cp) * 1.0
            else:
                obs = np.array([[cp, cv, np.cos(pa), np.sin(pa), pv]],
                               dtype=np.float32)
                action, _ = model.predict(obs, deterministic=True)
                current_action = float(np.clip(action[0][0], -1.0, 1.0))

            if physics_step % 400 == 0:
                mode = "CENTERING" if abs(cp) > 1.3 else "balancing"
                print(f"  cart:{cp:>+5.2f}m | "
                      f"pole:{np.degrees(pa):>+5.1f}° | "
                      f"ctrl:{current_action:>+4.2f} | {mode}",
                      flush=True)

        mj_data.ctrl[0] = current_action
        physics_step    += 1
        mujoco.mj_step(mj_model, mj_data)
        v.sync()
        time.sleep(mj_model.opt.timestep)
