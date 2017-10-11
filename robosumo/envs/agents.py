import os
import numpy as np
import xml.etree.ElementTree as ET

import gym


class Agent(object):
    """
    Superclass for all agents in sumo MuJoCo environment.
    """

    CFRC_CLIP = 100.

    COST_COEFS = {
        'ctrl': 1e-1,
        # 'pain': 1e-4,
        # 'attack': 1e-1,
    }

    JNT_NPOS = {
        0: 7,
        1: 4,
        2: 1,
        3: 1,
    }

    def __init__(self, env, scope, xml_path, adjust_z=0.):
        self._env = env
        self._scope = scope
        self._xml_path = xml_path
        self._xml = ET.parse(xml_path)
        self._adjust_z = adjust_z

        self._set_body()
        self._set_joint()

    def setup_spaces(self):
        self._set_observation_space()
        self._set_action_space()

    def _in_scope(self, name):
        return name.startswith(self._scope)

    def _set_body(self):
        self.body_names = list(filter(
            lambda x: self._in_scope(x), self._env.model.body_names
        ))
        self.body_ids = [
            self._env.model.body_names.index(name) for name in self.body_names
        ]
        self.body_name_idx = {
            name.split('/')[-1]: idx
            for name, idx in zip(self.body_names, self.body_ids)
        }
        # Determine body params
        self.body_dofnum = self._env.model.body_dofnum[self.body_ids]
        self.body_dofadr = self._env.model.body_dofadr[self.body_ids]
        self.nv = self.body_dofnum.sum()
        # Determine qvel_start_idx and qvel_end_idx
        dof = list(filter(lambda x: x >= 0, self.body_dofadr))
        self.qvel_start_idx = int(dof[0])
        last_dof_body_id = self.body_dofnum.shape[0] - 1
        while self.body_dofnum[last_dof_body_id] == 0:
            last_dof_body_id -= 1
        self.qvel_end_idx = int(dof[-1] + self.body_dofnum[last_dof_body_id])

    def _set_joint(self):
        self.joint_names = list(filter(
            lambda x: self._in_scope(x), self._env.model.joint_names
        ))
        self.joint_ids = [
            self._env.model.joint_names.index(name) for name in self.joint_names
        ]

        # Determine joint params
        self.jnt_qposadr = self._env.model.jnt_qposadr[self.joint_ids]
        self.jnt_type = self._env.model.jnt_type[self.joint_ids]
        self.jnt_nqpos = [self.JNT_NPOS[int(j)] for j in self.jnt_type]
        self.nq = sum(self.jnt_nqpos)
        # Determine qpos_start_idx and qpos_end_idx
        self.qpos_start_idx = int(self.jnt_qposadr[0])
        self.qpos_end_idx = int(self.jnt_qposadr[-1] + self.jnt_nqpos[-1])

    def _set_observation_space(self):
        obs = self.get_obs()
        self.obs_dim = obs.size
        low = -np.inf * np.ones(self.obs_dim)
        high = np.inf * np.ones(self.obs_dim)
        self.observation_space = gym.spaces.Box(low, high)

    def _set_action_space(self):
        acts = self._xml.find('actuator')
        self.action_dim = len(list(acts))
        default = self._xml.find('default')
        range_set = False
        if default is not None:
            motor = default.find('motor')
            if motor is not None:
                ctrl = motor.get('ctrlrange')
                if ctrl:
                    clow, chigh = list(map(float, ctrl.split()))
                    high = chigh * np.ones(self.action_dim)
                    low = clow * np.ones(self.action_dim)
                    range_set = True
        if not range_set:
            high =  np.ones(self.action_dim)
            low = - np.ones(self.action_dim)
        for i, motor in enumerate(list(acts)):
            ctrl = motor.get('ctrlrange')
            if ctrl:
                clow, chigh = list(map(float, ctrl.split()))
                low[i], high[i] = clow, chigh
        self._low, self._high = low, high
        self.action_space = gym.spaces.Box(low, high)

    def set_xyz(self, xyz):
        """Set (x, y, z) position of the agent; any element can be None."""
        qpos = self._env.data.qpos.ravel().copy()
        start = self.qpos_start_idx
        if xyz[0]: qpos[start]     = xyz[0]
        if xyz[1]: qpos[start + 1] = xyz[1]
        if xyz[2]: qpos[start + 2] = xyz[2]
        qvel = self._env.data.qvel.ravel()
        self._env.set_state(qpos, qvel)

    def set_euler(self, euler):
        """Set euler angles the agent; any element can be None."""
        qpos = self._env.data.qpos.ravel().copy()
        start = self.qpos_start_idx
        if euler[0]: qpos[start + 4] = euler[0]
        if euler[1]: qpos[start + 5] = euler[1]
        if euler[2]: qpos[start + 6] = euler[2]
        qvel = self._env.data.qvel.ravel()
        self._env.set_state(qpos, qvel)

    def set_opponents(self, opponents):
        self._opponents = opponents

    def reset(self):
        pass

    # --------------------------------------------------------------------------
    # Various getters
    # --------------------------------------------------------------------------

    def get_body_com(self, body_name):
        idx = self.body_names.index(self._scope + '/' + body_name)
        return self._env.data.subtree_com[self.body_ids[idx]]

    def get_cfrc_ext(self, body_ids=None):
        if body_ids is None:
            body_ids = self.body_ids
        return self._env.data.cfrc_ext[body_ids]

    def get_qpos(self):
        """Note: relies on the qpos for one agent being contiguously located.
        """
        qpos = self._env.data.qpos[self.qpos_start_idx:self.qpos_end_idx].copy()
        qpos[2] += self._adjust_z
        return qpos

    def get_qvel(self):
        """Note: relies on the qvel for one agent being contiguously located.
        """
        qvel = self._env.data.qvel[self.qvel_start_idx:self.qvel_end_idx]
        return qvel

    def get_qfrc_actuator(self):
        start, end = self.qvel_start_idx, self.qvel_end_idx
        qfrc = self._env.data.qfrc_actuator[start:end]
        return qfrc

    def get_cvel(self):
        cvel = self._env.data.cvel[self.body_ids]
        return cvel

    def get_body_mass(self):
        body_mass = self._env.model.body_mass[self.body_ids]
        return body_mass

    def get_xipos(self):
        xipos = self._env.data.xipos[self.body_ids]
        return xipos

    def get_cinert(self):
        cinert = self._env.data.cinert[self.body_ids]
        return cinert

    def get_obs(self):
        # Observe self
        self_forces = np.abs(np.clip(
            self.get_cfrc_ext(), -self.CFRC_CLIP, self.CFRC_CLIP))
        obs  = [
            self.get_qpos().flat,           # self all positions
            self.get_qvel().flat,           # self all velocities
            self_forces.flat,               # self all forces
        ]
        # Observe opponents
        for opp in self._opponents:
            body_ids = [
                opp.body_name_idx[name]
                for name in ['torso']
                if name in opp.body_name_idx
            ]
            opp_forces = np.abs(np.clip(
                opp.get_cfrc_ext(body_ids), -self.CFRC_CLIP, self.CFRC_CLIP))
            obs.extend([
                opp.get_qpos()[:7].flat,    # opponent torso position
                opp_forces.flat,            # opponent torso forces
            ])
        return np.concatenate(obs)

    def before_step(self):
        self.posbefore = self.get_qpos()[:2].copy()

    def after_step(self, action):
        self.posafter = self.get_qpos()[:2].copy()
        # Control cost
        reward = - self.COST_COEFS['ctrl'] * np.square(action).sum()
        return reward


# ------------------------------------------------------------------------------
# Beasts
# ------------------------------------------------------------------------------

class Ant(Agent):
    """
    The 4-leg agent.
    """

    def __init__(self, env, scope="ant", **kwargs):
        xml_path = os.path.join(os.path.dirname(__file__),
                                "assets", "ant.xml")
        super(Ant, self).__init__(env, scope, xml_path, **kwargs)


class Bug(Agent):
    """
    The 6-leg agent.
    """

    def __init__(self, env, scope="bug", **kwargs):
        xml_path = os.path.join(os.path.dirname(__file__),
                                "assets", "bug.xml")
        super(Bug, self).__init__(env, scope, xml_path, **kwargs)


class Spider(Agent):
    """
    The 8-leg agent.
    """

    def __init__(self, env, scope="spider", **kwargs):
        xml_path = os.path.join(os.path.dirname(__file__),
                                "assets", "spider.xml")
        super(Spider, self).__init__(env, scope, xml_path, **kwargs)


# ------------------------------------------------------------------------------

_available_agents = {
    'ant': Ant,
    'bug': Bug,
    'spider': Spider,
}


def get(name, *args, **kwargs):
    if name not in _available_agents:
        raise ValueError("Class %s is not available." % name)
    return _available_agents[name](*args, **kwargs)
