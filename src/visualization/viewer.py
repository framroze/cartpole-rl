"""
MuJoCo Viewer for trained CartPole policy.

Controls:
    Double-click pole  → grab and disturb
    Right-click + drag → rotate camera
    Scroll             → zoom
    Close window       → exit
"""
import os
import sys
import numpy as np
import time

import mujoco
import mujoco.viewer

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../.."))
from mujoco_playground._src import mjx_env
from mujoco_playground._src.dm_control_suite import common

XML_PATH = mjx_env.ROOT_PATH / "dm_control_suite" / "xmls" / "cartpole.xml"
assets   = common.get_assets()

SAVE_PATH = os.path.join(
    os.path.dirname(__file__), "../../results/cartpole_ppo.zip"
)

PHYSICS_HZ       = 200
POLICY_HZ        = 50
POLICY_FREQ      = PHYSICS_HZ // POLICY_HZ
CART_DAMPING     = 0.05
POLE_DAMPING     = 0.01
CENTER_THRESHOLD = 1.3


def build_model():
    mj_model = mujoco.MjModel.from_xml_string(XML_PATH.read_text(), assets)
    mj_model.opt.timestep   = 1.0 / PHYSICS_HZ
    mj_model.dof_damping[0] = CART_DAMPING
    mj_model.dof_damping[1] = POLE_DAMPING
    return mj_model


def get_action(policy, mj_data):
    cp = float(mj_data.qpos[0])
    cv = float(mj_data.qvel[0])
    pa = float(mj_data.qpos[1])
    pv = float(mj_data.qvel[1])

    if abs(cp) > CENTER_THRESHOLD:
        return -np.sign(cp) * 1.0, "centering"

    obs    = np.array([[cp, cv, np.cos(pa), np.sin(pa), pv]], dtype=np.float32)
    action, _ = policy.predict(obs, deterministic=True)
    return float(np.clip(action[0][0], -1.0, 1.0)), "balancing"


def run():
    from stable_baselines3 import PPO

    if not os.path.exists(SAVE_PATH):
        print(f"No trained model found at:\n  {SAVE_PATH}")
        print("Run: python main.py train")
        return

    print("=" * 50)
    print("  CartPole — MuJoCo Viewer")
    print("=" * 50)
    print("\nLoading trained policy...")

    policy   = PPO.load(SAVE_PATH)
    mj_model = build_model()
    mj_data  = mujoco.MjData(mj_model)

    mj_data.qpos[0] = 0.0
    mj_data.qpos[1] = 0.05
    mj_data.qvel[:] = 0.0
    mujoco.mj_forward(mj_model, mj_data)

    print("\nControls:")
    print("  Double-click pole  → grab and disturb")
    print("  Right-click + drag → rotate camera")
    print("  Close window       → exit\n")

    physics_step   = 0
    current_action = 0.0

    with mujoco.viewer.launch_passive(mj_model, mj_data) as v:
        v.cam.azimuth   = 0
        v.cam.elevation = -15
        v.cam.distance  = 4.0
        v.cam.lookat[:] = [0.0, 0.0, 1.2]

        while v.is_running():
            if physics_step % POLICY_FREQ == 0:
                current_action, mode = get_action(policy, mj_data)

                if physics_step % 400 == 0:
                    cp = float(mj_data.qpos[0])
                    pa = np.degrees(float(mj_data.qpos[1]))
                    print(f"  cart: {cp:>+5.2f}m | "
                          f"pole: {pa:>+5.1f}° | "
                          f"ctrl: {current_action:>+4.2f} | {mode}",
                          flush=True)

            mj_data.ctrl[0] = current_action
            physics_step    += 1
            mujoco.mj_step(mj_model, mj_data)
            v.sync()
            time.sleep(1.0 / PHYSICS_HZ)
