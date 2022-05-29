#!/usr/bin/env python

import os
import time
import numpy as np
import keras.backend as K

from copy import deepcopy
from keras.layers import Input, Dense
from keras.optimizers import Adam, RMSprop
from keras.models import Model, model_from_json, load_model


class Memory:
  def __init__(self):
    self.batch_s = []
    self.batch_a = []
    self.batch_r = []
    self.batch_s_ = []
    self.batch_done = []

  def store(self, s, a, r, s_, done):
    self.batch_s.append(s)
    self.batch_a.append(a)
    self.batch_r.append(r)
    self.batch_s_.append(s_)
    self.batch_done.append(done)

  def clear(self):
    self.batch_s.clear()
    self.batch_a.clear()
    self.batch_r.clear()
    self.batch_s_.clear()
    self.batch_done.clear()

  @property
  def count_sample(self):
    return len(self.batch_s)


class Agent:
  def __init__(self, dic_agent_conf, dic_path, dic_env_conf):
    self.dic_agent_conf = dic_agent_conf
    self.dic_path = dic_path
    self.dic_env_conf = dic_env_conf

    self.n_actions = self.dic_agent_conf["ACTION_DIM"]

    self.actor_network = self._build_actor_network()
    self.actor_old_network = self.build_network_from_copy(self.actor_network)

    self.critic_network = self._build_critic_network()

    self.dummy_advantage = np.zeros((1,1))
    self.dummy_old_prediction = np.zeros((1, self.n_actions))

    self.memory = Memory()

  def choose_action(self, state):
    assert isinstance(state, np.ndarray), "state must be numpy.ndarray"
    state = np.reshape(state, [-1, self.dic_agent_conf["STATE_DIM"][0]])
    prob = self.actor_network.predict_on_batch([state, self.dummy_advantage, self.dummy_old_prediction]).flatten()
    action = np.random.choice(self.n_actions, p=prob)
    return action

  def train_network(self):
    n = self.memory.count_sample
    discounted_r = []
    if self.memory.batch_done[-1]:
      v = 0
    else:
      v = self.get_v(self.memory.batch_s_[-1])

    for r in self.memory.batch_r[::-1]:
      v = r + self.dic_agent_conf["GAMMA"] * v
      discounted_r.reverse()

    batch_s, batch_a, batch_discounted_r = np.vstack(self.memory.batch_s), np.vstack(self.memory.batch_a), np.vstack(discounted_r)

    batch_v = self.get_v(batch_s)

    batch_advantage = batch_discounted_r - batch_v
    batch_old_prediction = self.get_old_prediction(batch_s)

    batch_a_final = np.zeros(shape=(len(batch_a), self.n_actions))
    batch_a_final[:, batch_a.flatten()] = 1
    self.actor_network.fit(x=[batch_s, batch_advantage, batch_old_prediction], y=batch_a_final, verbose=0)
    self.critic_network.fit(x=batch_s, y=batch_discounted_r, epoch=2, verbose=0)
    self.memory.clear()
    self.update_targe_network()

  def get_old_prediction(self, s):
    s = np.reshape(s, (-1, self.dic_agent_conf["STATE_DIM"][0]))
    v = self.critic_network.predict_on_batch(s)
    return v

  def store_transition(self, s, a, s_, r, done):
    self.memory.store(s, a, s_, r, done)

  def get_v(self, s):
    s = np.reshape(s, (-1, self.dic_agent_conf["STATE_DIM"][0]))
    v = self.critic_network.predict_on_batch(s)
    return v

  def save_model(self, filename):
    self.actor_network.save(os.path.join(self.dic_path["PATH_TO_MODEL"], "%s_actor_network.h5" % filename))
    self.critic_network.save(os.path.join(self.dic_path["PATH_TO_MODEL"], "%s_critic_network.h5"))

  def load_model(self):
    self.actor_network = load_model(self.dic_path["PATH_TO_MODEL"], "%s_actor_network.h5")
    self.critic_network = load_model(self.dic_path["PATH_TO_MODEL"], "%s_critic_network.h5")
    self.actor_old_network = deepcopy(self.actor_network)

  def _build_actor_network(self):
    state = Input(shape=self.dic_agent_conf["STATE_DIM"], name="state")
    advantage = Input(shape=(1, ), name="Advantage")