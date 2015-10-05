import os
import numpy as np
import chainer
import chainer.functions as F
from chainer import optimizers
from sample import Sampler
from copy import deepcopy
import math
import random

import pickle # used to save the nets

class DQN(object):

  def __init__(self, model, forward, settings):
    self.model = model
    self.forward = forward
    self.target_model = deepcopy(model)

    self.batch_size = settings['batch_size']
    self.n_epochs = settings['n_epochs']

    self.learning_rate = settings['learning_rate']
    self.decay_rate = settings['decay_rate']
    self.discount = settings['discount']
    self.clip_err = settings['clip_err']
    self.clip_reward = settings['clip_reward']
    self.target_net_update = settings['target_net_update']
    self.print_every = settings['print_every']
    self.save_every = settings['save_every']
    self.save_dir = settings['save_dir']
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
    self.optimizer.setup(self.model)

    self.train_losses = []
    self.val_losses = []
    self.policy_rewards = []
    self.qval_avgs = []

  # sampling of one mini-batch and one update using it
  def iteration(self, sampler):

    # load data
    s0, a, r, s1 = sampler.minibatch('train')

    if self.clip_reward:
      r = np.clip(r,-self.clip_reward,self.clip_reward)

    s0, s1 = chainer.Variable(s0), chainer.Variable(s1, volatile = True)

    # calculate target Q-values (from s1 and on)
    if not self.double_DQN:
      target_q_all = self.forward(self.target_model, s1)
      target_q_max = np.max(target_q_all.data, 1)
    else:
      target_q_all = self.forward(self.model, s1)
      target_argmax = np.argmax(target_q_all.data, 1)
      target_q_all = self.forward(self.target_model, s1)
      target_q_max = target_q_all.data[np.arange(target_q_all.data.shape[0]),target_argmax]
      
    target_q_value = r + self.discount * target_q_max

    # calculate expected Q-values for the actions we actually took
    approx_q_all = self.forward(self.model, s0)
    approx_q_value = approx_q_all.data[np.arange(approx_q_all.data.shape[0]),a]

    # resets gradient storage to zero
    self.optimizer.zero_grads()

    # calculate the loss
    gradLoss = approx_q_value - target_q_value
    loss = np.mean(gradLoss**2)

    # calculate and distribute the loss gradient (clipped)
    if self.clip_err:
      gradLoss = np.clip(gradLoss,-self.clip_err,self.clip_err)
    gradLossAll = np.zeros_like(approx_q_all.data)
    gradLossAll[np.arange(gradLossAll.shape[0]),a] = gradLoss
    approx_q_all.grad = gradLossAll
    approx_q_all.backward()

    # carry out parameter updates based on the distributed gradients
    self.optimizer.update()

    return loss

  # extract the optimal policy in the given state
  def policy(self, s):
    s = chainer.Variable(s, volatile = True)
    approx_q_all = self.forward(self.model, s)
    opt_a = np.argmax(approx_q_all.data,1)
    return opt_a

  # collect model parameters (coefs and grads)
  def params(self, model):
    self.model.collect_parameters()


  # function encapsulating the training process
  def train(self, sampler):

    # create "models" directory to save training output
    if not os.path.exists(self.save_dir):
      print("Creating '%s' directory to store training results..." % self.save_dir)
      os.makedirs(self.save_dir)

    if self.clip_reward:
      print("Rewards are clipped in training, but not in evaluation")

    # estimate number of mini-batches in a training data
    iterations_per_epoch = int(math.ceil(sampler.data_size['train'] / self.batch_size))
    # total number of iterations in training
    iterations = self.n_epochs * iterations_per_epoch

    for i in range(1,iterations):

      epoch = i / float(iterations_per_epoch)
      loss = self.iteration(sampler) # run an update iteration on one mini-batch
      self.train_losses.append(loss)

      # copy the updated net onto the target generating net
      if i % self.target_net_update == 0:
          self.target_model = deepcopy(self.model)

      # print overview metrics every fixed number of iterations
      if i % self.print_every == 0:   
        print('progress %.3f, epoch %.2f, iteration %d, train loss %.3f' % ((i/float(iterations)),epoch,i,loss))

      # every epoch run evaluation and save the net
      if i % iterations_per_epoch == 0:

        val_loss, qval_avg, policy_reward = self.evaluate(sampler, 'val')

        self.val_losses.append(val_loss)
        self.qval_avgs.append(qval_avg)
        self.policy_rewards.append(policy_reward)

        if epoch % self.save_every == 0:
          self.savemodel(self.model,'./%s/model_%d.p' % (self.save_dir,int(epoch)))

        print('%d, epoch %.2f, validation loss %.3f, average policy reward %.3f, average Q-value %.3f' % ((i/float(iterations)), 
          epoch, val_loss, policy_reward, qval_avg))

    # save the DQN object together with training history at the end of training
    self.savemodel(self,'./%s/DQN_final.p' % self.save_dir)

  # helper function to save model (or any object)
  def savemodel(self,model,name):
    pickle.dump(model, open(name, "wb"))

  # helper function to load model (or any object)
  def loadmodel(self,name):
    return pickle.load(open(name, "rb"))

  # function to do evaluation - similar to "iteration" function
  def evaluate(self,sampler,split,n=100000):
    
    sampler.reset(split) # reset the index used for mini-batch sampling
    n = min(n,int(math.ceil(sampler.data_size[split] / self.batch_size)))

    loss = qval_avg = policy_reward = 0

    for i in range(n):

      s0,a,r,s1 = sampler.minibatch(split)

      #if self.clip_reward:
      #  r = np.clip(r,-self.clip_reward,self.clip_reward)

      s0, s1 = chainer.Variable(s0, volatile=True), chainer.Variable(s1, volatile=True)
      
      if not self.double_DQN:
        target_q_all = self.forward(self.target_model, s1)
        target_q_max = np.max(target_q_all.data, 1)
      else:
        target_q_all = self.forward(self.model, s1)
        target_argmax = np.argmax(target_q_all.data, 1)
        target_q_all = self.forward(self.target_model, s1)
        target_q_max = target_q_all.data[np.arange(target_q_all.data.shape[0]),target_argmax]
      
      target_q_value = r + self.discount * target_q_max

      approx_q_all = self.forward(self.model, s0)
      approx_q_value = approx_q_all.data[np.arange(approx_q_all.data.shape[0]),a]

      opt_a = np.argmax(approx_q_all.data,1)
      ind = opt_a == a
      # if, in the mini-batch, at least one action complies with policy recommendations
      if sum(ind) > 0:
        policy_reward += np.mean(r[ind])

      qval_avg += np.mean(approx_q_value)
      gradLoss = approx_q_value - target_q_value
      loss += np.mean(gradLoss**2)

    policy_reward /= n
    qval_avg /= n
    loss /= n

    return loss, qval_avg, policy_reward