#!/usr/bin/env python

import torch

from network import FeedForwardNN
from torch.distributions import MultivariateNormal

class PPO:
  def __init__(self, env):
    
    # Initialize hyperparameters for training with PPO
    self._init_hyperparameters()

    # Initialize the convariance matrix used to query the actor for actions
    self.cov_var = torch.full(size=(self.act_dim,), fill_value=0.5)
    self.cov_mat = torch.diag(self.cov_var)

    # Extract environment information
    self.env = env
    self.obs_dim = env.observation_space.shape[0]
    self.act_dim = env.action_space.shape[0]

    # Initialise actor and critic networks
    self.actor = FeedForwardNN(self.obs_dim, self.act_dim)
    self.critic = FeedForwardNN(self.obs_dim, 1)


  def learn(self, total_timesteps):
    t_so_far = 0  # Timesteps simulated so far
    
    while t_so_far < total_timesteps:
      # Increment t_so_far somewhere below
      continue
  
  
  def _init_hyperparameters(self):
    # Default values for hyperparameters, to change later.
    self.timesteps_per_batch = 4800         # timesteps per batch
    self.max_timesteps_per_episode = 1600   # timesteps per episode


  def rollout(self):
    # Batch data
    batch_obs = []        # batch observations
    batch_acts = []       # batch actions
    batch_log_probs = []  # log probs of each action
    batch_rewards = []       # batch rewards
    batch_rtgs = []       # batch rewards-to-go
    batch_lens = []       # episodic lengths in batch

    t = 0                 # Keeps track of how many timesteps have been ran for this batch.

    while t < self.timesteps_per_batch:
      # Reward this episode
      ep_rewards = []
      obs = self.env.reset()
      done = False

      for ep_t in range(self.max_timesteps_per_episode):
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

        if done:
          break
      
      # Collect episodic length and rewards
      batch_lens.append(ep_t + 1)  # plus 1 because timestep starts at 0
      batch_rewards.append(ep_rewards)

    # Reshape data as tensors in the shape specified in function description, before returning
    batch_obs = torch.tensor(batch_obs, dtype=torch.float)
    batch_acts = torch.tensor(batch_acts, dtype=torch.float)
    batch_log_probs = torch.tensor(batch_log_probs, dtype=torch.float)
    batch_rtgs = self.compute_rtgs(batch_rewards)

    # Log the episodic returns and episodic lengths in this batch.
    self.logger['batch_rewards'] = batch_rewards
    self.logger['batch_lens'] = batch_lens

    return batch_obs, batch_acts, batch_log_probs, batch_rtgs, batch_lens


  def get_action(self, obs):
    # Query the actor network for a mean action
    mean = self.actor(obs)

    # Create a distribution with the mean action and std dev from the covariance matrix above
    distrib = MultivariateNormal(mean, self.cov_mat)

    # Sample an action from the distrib and get its log prob
    action = dist.sample()
    log_prob = dist.log_prob(action)

    return action.detach().numpy(), log_prob.detach()
