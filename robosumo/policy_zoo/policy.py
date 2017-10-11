"""
Policy classes.
"""
import tensorflow as tf
import numpy as np
import gym
import logging
import copy

from tensorflow.contrib import layers

from .utils import *


class Policy(object):
    def reset(self, **kwargs):
        pass

    def act(self, observation):
        raise NotImplementedError


class MLPPolicy(Policy):
    def __init__(self, scope, *, ob_space, ac_space, hiddens,
                 normalize=False,
                 reuse=False):
        self.recurrent = False
        self.normalized = normalize

        with tf.variable_scope(scope, reuse=reuse):
            self.scope = tf.get_variable_scope().name

            self.observation_ph = tf.placeholder(
                tf.float32, [None] + list(ob_space.shape), name="observation")
            self.taken_action_ph = tf.placeholder(
                tf.float32, [None, ac_space.shape[0]], name="taken_action")
            self.stochastic_ph = tf.placeholder(tf.bool, (), name="stochastic")

            if self.normalized:
                if self.normalized != 'ob':
                    self.ret_rms = RunningMeanStd(scope="retfilter")
                self.ob_rms = RunningMeanStd(
                    scope="obsfilter", shape=ob_space.shape)

            # Observation filtering
            obz = self.observation_ph
            if self.normalized:
                obz = tf.clip_by_value((self.observation_ph - self.ob_rms.mean) / self.ob_rms.std, -5.0, 5.0)

            # Value
            last_out = obz
            for i, hid_size in enumerate(hiddens):
                last_out = tf.nn.tanh(
                    dense(last_out, hid_size, "vffc%i" % (i + 1)))
            self.vpredz = dense(last_out, 1, "vffinal")[:, 0]

            self.vpred = self.vpredz
            if self.normalized and self.normalized != 'ob':
                self.vpred = self.vpredz * self.ret_rms.std + self.ret_rms.mean

            # Policy
            last_out = obz
            for i, hid_size in enumerate(hiddens):
                last_out = tf.nn.tanh(
                    dense(last_out, hid_size, "polfc%i" % (i + 1)))
            mean = dense(last_out, ac_space.shape[0], "polfinal")
            logstd = tf.get_variable(
                name="logstd",
                shape=[1, ac_space.shape[0]],
                initializer=tf.zeros_initializer())

            self.pd = DiagonalGaussian(mean, logstd)
            self.sampled_action = switch(
                self.stochastic_ph, self.pd.sample(), self.pd.mode())

    def act(self, observation, stochastic=True):
        outputs = [self.sampled_action, self.vpred]
        feed_dict = {
            self.observation_ph: observation[None],
            self.stochastic_ph: stochastic,
        }
        a, v = tf.get_default_session().run(outputs, feed_dict)
        return a[0], {'vpred': v[0]}

    def get_variables(self):
        return tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, self.scope)


class LSTMPolicy(Policy):
    def __init__(self, scope, *, ob_space, ac_space, hiddens,
                 reuse=False, normalize=False):
        self.recurrent = True
        self.normalized = normalize

        with tf.variable_scope(scope, reuse=reuse):
            self.scope = tf.get_variable_scope().name

            self.observation_ph = tf.placeholder(
                tf.float32, [None, None] + list(ob_space.shape),
                name="observation")
            self.taken_action_ph = tf.placeholder(
                tf.float32, [None, None, ac_space.shape[0]],
                name="taken_action")
            self.stochastic_ph = tf.placeholder(tf.bool, (), name="stochastic")

            if self.normalized:
                if self.normalized != 'ob':
                    self.ret_rms = RunningMeanStd(scope="retfilter")
                self.ob_rms = RunningMeanStd(
                    scope="obsfilter",
                    shape=ob_space.shape)

            # Observation filtering
            obz = self.observation_ph
            if self.normalized:
                obz = tf.clip_by_value((self.observation_ph - self.ob_rms.mean) / self.ob_rms.std, -5.0, 5.0)

            # Embedding
            last_out = obz
            for hidden in hiddens[:-1]:
                last_out = tf.contrib.layers.fully_connected(last_out, hidden)

            self.zero_state = []
            self.state_in_ph = []
            self.state_out = []

            # Value
            cell = tf.contrib.rnn.BasicLSTMCell(hiddens[-1], reuse=reuse)
            size = cell.state_size
            self.zero_state.append(np.zeros(size.c, dtype=np.float32))
            self.zero_state.append(np.zeros(size.h, dtype=np.float32))
            self.state_in_ph.append(
                tf.placeholder(tf.float32, [None, size.c], name="lstmv_c"))
            self.state_in_ph.append(
                tf.placeholder(tf.float32, [None, size.h], name="lstmv_h"))
            initial_state = tf.contrib.rnn.LSTMStateTuple(
                self.state_in_ph[-2], self.state_in_ph[-1])
            last_out, state_out = tf.nn.dynamic_rnn(
                cell, last_out, initial_state=initial_state, scope="lstmv")
            self.state_out.append(state_out)

            self.vpredz = tf.contrib.layers.fully_connected(last_out, 1, activation_fn=None)[:, :, 0]
            self.vpred = self.vpredz
            if self.normalized and self.normalized != 'ob':
                self.vpred = self.vpredz * self.ret_rms.std + self.ret_rms.mean

            # Policy
            last_out = obz
            for hidden in hiddens[:-1]:
                last_out = tf.contrib.layers.fully_connected(last_out, hidden)
            cell = tf.contrib.rnn.BasicLSTMCell(hiddens[-1], reuse=reuse)
            size = cell.state_size
            self.zero_state.append(np.zeros(size.c, dtype=np.float32))
            self.zero_state.append(np.zeros(size.h, dtype=np.float32))
            self.state_in_ph.append(
                tf.placeholder(tf.float32, [None, size.c], name="lstmp_c"))
            self.state_in_ph.append(
                tf.placeholder(tf.float32, [None, size.h], name="lstmp_h"))
            initial_state = tf.contrib.rnn.LSTMStateTuple(
                self.state_in_ph[-2], self.state_in_ph[-1])
            last_out, state_out = tf.nn.dynamic_rnn(
                cell, last_out, initial_state=initial_state, scope="lstmp")
            self.state_out.append(state_out)

            mean = tf.contrib.layers.fully_connected(
                last_out, ac_space.shape[0], activation_fn=None)
            logstd = tf.get_variable(
                name="logstd",
                shape=[1, ac_space.shape[0]],
                initializer=tf.zeros_initializer())

            self.pd = DiagonalGaussian(mean, logstd)
            self.sampled_action = switch(
                self.stochastic_ph, self.pd.sample(), self.pd.mode())

            self.zero_state = np.array(self.zero_state)
            self.state_in_ph = tuple(self.state_in_ph)
            self.state = self.zero_state

    def act(self, observation, stochastic=True):
        outputs = [self.sampled_action, self.vpred, self.state_out]
        feed_dict = {
            self.observation_ph: observation[None, None],
            self.state_in_ph: list(self.state[:, None, :]),
            self.stochastic_ph: stochastic,
        }
        a, v, s = tf.get_default_session().run(outputs, feed_dict)
        self.state = []
        for x in s:
            self.state.append(x.c[0])
            self.state.append(x.h[0])
        self.state = np.array(self.state)
        return a[0, 0], {'vpred': v[0, 0], 'state': self.state}

    def get_variables(self):
        return tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, self.scope)

    def reset(self):
        self.state = self.zero_state
