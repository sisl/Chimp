import os
import numpy as np
import chainer
import chainer.functions as F
from chainer import optimizers
from copy import deepcopy
import math
import random

import pickle # used to save the nets

class Learner(object):

  def __init__(self, net, forward, settings):

    self.net = net
    self.forward = forward
    self.target_net = deepcopy(net)

    self.learning_rate = settings['learning_rate']
    self.decay_rate = settings['decay_rate']
    self.discount = settings['discount']
    self.clip_err = settings['clip_err']
    self.clip_reward = settings['clip_reward']
    self.target_net_update = settings['target_net_update']
    self.double_DQN = settings['double_DQN']

    # setting up various possible gradient update algorithms
    if settings['optim_name'] == 'RMSprop':
      self.optimizer = optimizers.RMSprop(lr=self.learning_rate, alpha=self.decay_rate)
    elif settings['optim_name'] == 'ADADELTA':
      print("Supplied learning rate not used with ADADELTA gradient update method")
      self.optimizer = optimizers.AdaDelta()
    elif settings['optim_name'] == 'SGD':
      self.optimizer = optimizers.SGD(lr=self.learning_rate)
    else:
      print('The requested optimizer is not supported!!!')
      exit()

    self.optim_name = settings['optim_name']
    self.optimizer.setup(self.net)

    self.train_losses = []
    self.val_losses = []
    self.policy_rewards = []
    self.qval_avgs = []

  # sampling of one mini-batch and one update using it
  def gradUpdate(self, s0, a, r, s1):

    if self.clip_reward:
      r = np.clip(r,-self.clip_reward,self.clip_reward)

    # reset gradient storage to zero
    self.optimizer.zero_grads()
    # move forward through the net and produce output variable 
    # containing the loss gradient + MSE loss + average Q-value for taken actions
    approx_q_all, loss, qval_avg = self.forward_loss(s0, a, r, s1)
    # propagate the loss gradient through the net
    approx_q_all.backward()
    # carry out parameter updates based on the distributed gradients
    self.optimizer.update()

    return loss

  # function to get net output and to calculate the loss
  def forward_loss(self, s0, a, r, s1):

    # transfer states into Chainer format
    s0, s1 = chainer.Variable(s0), chainer.Variable(s1, volatile = True)
    # calculate target Q-values (from s1 and on)
    if not self.double_DQN:
      target_q_all = self.forward(self.target_net, s1)
      target_q_max = np.max(target_q_all.data, 1)
    else:
      target_q_all = self.forward(self.net, s1)
      target_argmax = np.argmax(target_q_all.data, 1)
      target_q_all = self.forward(self.target_net, s1)
      target_q_max = target_q_all.data[np.arange(target_q_all.data.shape[0]),target_argmax]
      
    target_q_value = r + self.discount * target_q_max

    # calculate expected Q-values for all actions
    approx_q_all = self.forward(self.net, s0)
    # extract expected Q-values for the actions we actually took
    approx_q_value = approx_q_all.data[np.arange(approx_q_all.data.shape[0]),a]
    # calculate the loss gradient
    gradLoss = approx_q_value - target_q_value

    # clip the loss gradient
    if self.clip_err:
      gradLoss = np.clip(gradLoss,-self.clip_err,self.clip_err)

    # distribute the loss gradient into the shape of the net's output
    gradLossAll = np.zeros_like(approx_q_all.data)
    gradLossAll[np.arange(gradLossAll.shape[0]),a] = gradLoss
    # transfer the loss gradient
    approx_q_all.grad = gradLossAll

    return approx_q_all, np.mean(gradLoss**2), np.mean(approx_q_value)

  # extract the optimal policy in the given state
  def policy(self, s):
    s = chainer.Variable(s, volatile = True)
    approx_q_all = self.forward(self.net, s)
    opt_a = np.argmax(approx_q_all.data,1)
    return opt_a

  # collect net parameters (coefs and grads)
  def params(self, net):
    self.net.collect_parameters()
