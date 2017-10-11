"""
The base class for environments based on MuJoCo 1.5.
"""
import os
import sys
import numpy as np

import gym
from gym import error, spaces
from gym.utils import seeding

try:
    import mujoco_py
    from mujoco_py import load_model_from_path, MjSim, MjViewer
    import glfw
except ImportError as e:
    raise error.DependencyNotInstalled("{}. (HINT: you need to install mujoco_py, and also perform the setup instructions here: https://github.com/openai/mujoco-py/.)".format(e))

from pkg_resources import parse_version

if parse_version(mujoco_py.__version__) < parse_version('1.5'):
    raise error.DependencyNotInstalled(
        "RoboSumo requires mujoco_py of version 1.5 or higher. "
        "The installed version is {}. Please upgrade mujoco_py."
        .format(mujoco_py.__version__))


def _read_pixels(sim, width=None, height=None, camera_name=None):
    """Reads pixels w/o markers and overlay from the same camera as screen."""
    if width is None or height is None:
        resolution = glfw.get_framebuffer_size(
            sim._render_context_window.window)
        resolution = np.array(resolution)
        resolution = resolution * min(1000 / np.min(resolution), 1)
        resolution = resolution.astype(np.int32)
        resolution -= resolution % 16
        width, height = resolution

    img = sim.render(width, height, camera_name=camera_name)
    img = img[::-1, :, :] # Rendered images are upside-down.
    return img


class MujocoEnv(gym.Env):
    """Superclass for all MuJoCo environments.
    """
    def __init__(self, model_path, frame_skip):
        if model_path.startswith("/"):
            fullpath = model_path
        else:
            fullpath = os.path.join(os.path.dirname(__file__), "assets", model_path)
        if not os.path.exists(fullpath):
            raise IOError("File %s does not exist" % fullpath)
        self.frame_skip = frame_skip
        self.model = load_model_from_path(fullpath)
        self.sim = MjSim(self.model)
        self.data = self.sim.data
        self.viewer = None
        self.buffer_size = (1600, 1280)

        self.metadata = {
            'render.modes': ['human', 'rgb_array'],
            'video.frames_per_second': 60,
        }

        self.init_qpos = self.data.qpos.ravel().copy()
        self.init_qvel = self.data.qvel.ravel().copy()
        observation, _reward, done, _info = self._step(np.zeros(self.model.nu))
        assert not done
        self.obs_dim = np.sum([o.size for o in observation]) if (
            type(observation) is tuple) else observation.size

        bounds = self.model.actuator_ctrlrange.copy()
        low, high = bounds[:, 0], bounds[:, 1]
        self.action_space = spaces.Box(low, high)

        high = np.inf * np.ones(self.obs_dim)
        self.observation_space = spaces.Box(-high, high)

        self._seed()

    def _seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    # methods to override:
    # ------------------------------------------------------------------------

    def reset_model(self):
        """Reset the robot degrees of freedom (qpos and qvel).
        Implement this in each subclass.
        """
        raise NotImplementedError

    def viewer_setup(self):
        """Called when the viewer is initialized and after every reset.
        Optionally implement this method, if you need to tinker with camera
        position and so forth.
        """
        pass

    # ------------------------------------------------------------------------

    def _reset(self):
        self.sim.reset()
        self.sim.forward()
        ob = self.reset_model()
        return ob

    def set_state(self, qpos, qvel):
        assert qpos.shape == (self.model.nq,)
        assert qvel.shape == (self.model.nv,)
        state = self.sim.get_state()
        for i in range(self.model.nq):
            state.qpos[i] = qpos[i]
        for i in range(self.model.nv):
            state.qvel[i] = qvel[i]
        self.sim.set_state(state)
        self.sim.forward()

    @property
    def dt(self):
        return self.model.opt.timestep * self.frame_skip

    def do_simulation(self, ctrl, n_frames):
        for i in range(self.model.nu):
            self.sim.data.ctrl[i] = ctrl[i]
        for _ in range(n_frames):
            self.sim.step()

    def _render(self, mode='human', close=False):
        if close:
            if self.viewer is not None:
                self.viewer = None
            return

        if mode == 'rgb_array':
            self.viewer_setup()
            return _read_pixels(self.sim, *self.buffer_size)
        elif mode == 'human':
            self._get_viewer().render()

    def _get_viewer(self, mode='human'):
        if self.viewer is None and mode == 'human':
            self.viewer = MjViewer(self.sim)
            self.viewer_setup()
        return self.viewer

    def state_vector(self):
        state = self.sim.get_state()
        return np.concatenate([state.qpos.flat, state.qvel.flat])
