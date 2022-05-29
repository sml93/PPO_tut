#!/usr/bin/env python

import torch
import numpy as np
import torch.nn as nn

from torch.optim import Adam
from network import FeedForwardNN
from torch.distributions import MultivariateNormal

class PPO:
  def __init__(self, env):
    
    # Initialize hyperparameters for training with PPO
    self._init_hyperparameters()

    # Extract environment information
    self.env = env
    self.obs_dim = env.observation_space.shape[0]
    self.act_dim = env.action_space.shape[0]

    # Initialise actor and critic networks (ALGO STEP 1)
    self.actor = FeedForwardNN(self.obs_dim, self.act_dim)
    self.critic = FeedForwardNN(self.obs_dim, 1)

    # Initialize the convariance matrix used to query the actor for actions
    self.cov_var = torch.full(size=(self.act_dim,), fill_value=0.5)
    self.cov_mat = torch.diag(self.cov_var)

    # Initialize the optimizers for actor and critic
    self.actor_optim = Adam(self.actor.parameters(), lr=self.lr)
    self.critic_optim = Adam(self.critic.parameters(), lr=self.lr)


  def _init_hyperparameters(self):
    # Default values for hyperparameters, to change later.
    self.max_timesteps_per_episode = 1600   # timesteps per episode
    self.timesteps_per_batch = 4800         # timesteps per batch
    
    self.n_updates_per_iteration = 5
    
    self.gamma = 0.95
    self.clip = 0.2
    self.lr = 0.005


  def learn(self, total_timesteps):
    t_so_far = 0  # Timesteps simulated so far (ALGO STEP 2)
    
    while t_so_far < total_timesteps:
      # ALGO STEP 3
      batch_obs, batch_acts, batch_log_probs, batch_rtgs, batch_lens = self.rollout()

      # Calculate how many timesteps we collected this batch
      t_so_far += np.sum(batch_lens)

      # Calculate V_{phi, k}
      V, _ = self.evaluate(batch_obs, batch_acts)

      # Calculate advantage (ALGO STEP 5)
      A_k = batch_rtgs - V.detach()

      # Normalize advantages
      A_k = (A_k - A_k.mean()) / (A_k.std() + 1e-10)

      # This is the loop where we update our network for some n epochs (ALGO STEP 6 & 7)
      for _ in range(self.n_updates_per_iteration):
        # Calculate V_phi and pi_theta(a_t | s_t)
        V, curr_log_probs = self.evaluate(batch_obs, batch_acts)

        # Calculate ratios
        ratios = torch.exp(curr_log_probs - batch_log_probs)

        # Calculate surrogate losses
        surr1 = ratios * A_k
        surr2 = torch.clamp(ratios, 1-self.clip, 1+self.clip)*A_k

        # Negative sign to MAXIMISE this loss, or performance/obj function (through stochastic gradient ascent) but optimizer used is Adam which MINIMISES the loss.
        actor_loss = (-torch.min(surr1, surr2)).mean()
        critic_loss = nn.MSELoss()(V, batch_rtgs)
        
        # Calculate gradients and perform backward propagation for actor network
        self.actor_optim.zero_grad()
        actor_loss.backward(retain_graph=True)
        self.actor_optim.step()

        # Calculate gradients and perform backward propagation for critic network
        self.critic_optim.zero_grad()
        critic_loss.backward()
        self.critic_optim.step()
  
 
  def rollout(self):
    # Batch data
    batch_obs = []        # batch observations
    batch_acts = []       # batch actions
    batch_log_probs = []  # log probs of each action
    batch_rewards = []    # batch rewards
    batch_rtgs = []       # batch rewards-to-go
    batch_lens = []       # episodic lengths in batch

    t = 0                 # Keeps track of how many timesteps have been ran for this batch.

    while t < self.timesteps_per_batch:
      # Reward this episode
      ep_rewards = []
      obs = self.env.reset()
      done = False

      for ep_t in range(self.max_timesteps_per_episode):
        # self.env.render()
        # Increment timesteps ran this batch so far
        t += 1

        # Collect observations
        batch_obs.append(obs)

        action, log_prob = self.get_action(obs)
        obs, rewards, done, _ = self.env.step(action)

        # Collect reward, action, and log prob
        ep_rewards.append(rewards)
        batch_acts.append(action)
        batch_log_probs.append(log_prob)
        print("t: ", t)

        if done:
          print("done")
          # print("ep_rewards", ep_rewards)
          break
      
      # Collect episodic length and rewards
      batch_lens.append(ep_t + 1)  # plus 1 because timestep starts at 0
      batch_rewards.append(ep_rewards)

    # Reshape data as tensors in the shape specified in function description, before returning
    batch_obs = torch.tensor(batch_obs, dtype=torch.float)
    batch_acts = torch.tensor(batch_acts, dtype=torch.float)
    batch_log_probs = torch.tensor(batch_log_probs, dtype=torch.float)
    batch_rtgs = self.compute_rtgs(batch_rewards)

    return batch_obs, batch_acts, batch_log_probs, batch_rtgs, batch_lens


  def get_action(self, obs):
    # Query the actor network for a mean action
    mean = self.actor(obs)

    # Create a distribution with the mean action and std dev from the covariance matrix above
    distrib = MultivariateNormal(mean, self.cov_mat)

    # Sample an action from the distrib and get its log prob
    action = distrib.sample()
    log_prob = distrib.log_prob(action)
    # Calling detach() since the action and log_prob are tensors with computation graphs, 
    return action.detach().numpy(), log_prob.detach()


  def compute_rtgs(self, batch_rewards):
    # The rewards-to-go (rtg) per episode per bathc to return.
    # The shape will be (num of timesteps per episode)
    batch_rtgs = []

    # Iterate through each episode backwards to maintain same order in batch_rtgs
    for ep_rewards in reversed(batch_rewards):
      discounted_reward = 0   # The discounted reward so far
      for reward in reversed(ep_rewards):
        discounted_reward = reward + discounted_reward*self.gamma
        batch_rtgs.insert(0, discounted_reward)

    # Convert the rewards-to-go into a tensor
    batch_rtgs = torch.tensor(batch_rtgs, dtype=torch.float)
    return batch_rtgs

  
  def evaluate(self, batch_obs, batch_acts):
    # Query critic network for a value V for each obs in batch_obs
    V = self.critic(batch_obs).squeeze()

    # Calculate the log prob of batch actions using the most recent actor network.
    # This segment of code is siimilar to that in get_action()
    mean = self.actor(batch_obs)
    distrib = MultivariateNormal(mean, self.cov_mat)
    log_probs = distrib.log_prob(batch_acts)

    return V, log_probs


if __name__ == "__main__":
  import gym
  # env = gym.make('Pendulum-v0')
  env = gym.make('BipedalWalker-v3')
  model = PPO(env)
  model.learn(10000)
