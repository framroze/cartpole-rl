"""
MuJoCo CartPole Environment using Playground's exact XML.
"""
import mujoco
import numpy as np
import gymnasium as gym
from gymnasium import spaces
from mujoco_playground._src import mjx_env
from mujoco_playground._src.dm_control_suite import common

XML_PATH = mjx_env.ROOT_PATH / "dm_control_suite" / "xmls" / "cartpole.xml"
assets   = common.get_assets()

class MujocoCartpoleEnv(gym.Env):
    metadata = {"render_modes": ["rgb_array"], "render_fps": 50}

    def __init__(self, render_mode=None):
        super().__init__()
        self.model = mujoco.MjModel.from_xml_string(
            XML_PATH.read_text(), assets)
        self.model.opt.timestep = 0.005
        self.data  = mujoco.MjData(self.model)
        self.max_steps = 500

        self.action_space = spaces.Box(
            low=-1.0, high=1.0, shape=(1,), dtype=np.float32)
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(5,), dtype=np.float32)

        self.steps       = 0
        self.render_mode = render_mode
        self._renderer   = None

    def _get_obs(self):
        return np.array([
            float(self.data.qpos[0]),
            float(self.data.qvel[0]),
            float(np.cos(self.data.qpos[1])),
            float(np.sin(self.data.qpos[1])),
            float(self.data.qvel[1]),
        ], dtype=np.float32)

    def _is_done(self):
        return (abs(self.data.qpos[1]) > np.radians(30) or
                abs(self.data.qpos[0]) > 1.7)

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.data.qpos[0] = self.np_random.uniform(-0.5, 0.5)
        self.data.qpos[1] = self.np_random.uniform(-0.1, 0.1)
        self.data.qvel[0] = self.np_random.uniform(-0.1, 0.1)
        self.data.qvel[1] = self.np_random.uniform(-0.1, 0.1)
        self.data.ctrl[0] = 0.0
        mujoco.mj_forward(self.model, self.data)
        self.steps = 0
        return self._get_obs(), {}

    def step(self, action):
        act = float(np.clip(action[0], -1.0, 1.0))
        for _ in range(4):
            self.data.ctrl[0] = act
            mujoco.mj_step(self.model, self.data)
        self.steps += 1
        done   = self._is_done()
        reward = (1.0 - 0.5 * self.data.qpos[0]**2) if not done else -10.0
        return self._get_obs(), reward, done, self.steps >= self.max_steps, {}

    def render(self):
        if self.render_mode != "rgb_array": return None
        if self._renderer is None:
            self._renderer = mujoco.Renderer(self.model, 480, 640)
        self._renderer.update_scene(self.data, camera="fixed")
        return self._renderer.render()

    def close(self):
        if self._renderer is not None:
            self._renderer.close()

gym.register(
    id="MujocoCartpole-v0",
    entry_point="mujoco_cartpole_env:MujocoCartpoleEnv",
    max_episode_steps=500,
)
