import os
import numpy as np
import chainer
import chainer.functions as F
from chainer import optimizers
from copy import deepcopy
import math
import random

import pickle # used to save the nets

class Agent(object):

  def __init__(self, settings):

    self.batch_size = settings['batch_size']
    self.n_epochs = settings['n_epochs']

    self.print_every = settings['print_every']
    self.save_dir = settings['save_dir']
    self.save_every = settings['save_every']

  # function encapsulating the training process
  def train(self, learner, memory):

    # create "nets" directory to save training output
    if not os.path.exists(self.save_dir):
      print("Creating '%s' directory to store training results..." % self.save_dir)
      os.makedirs(self.save_dir)

    if learner.clip_reward:
      print("Rewards are clipped in training, but not in evaluation")

    # estimate number of mini-batches in a training data
    iterations_per_epoch = int(math.ceil(memory.data_size['train'] / self.batch_size))
    # total number of iterations in training
    iterations = self.n_epochs * iterations_per_epoch

    for i in range(1,iterations):

      epoch = i / float(iterations_per_epoch)
      s0,a,r,s1 = memory.minibatch('train',self.batch_size)
      loss = learner.gradUpdate(s0,a,r,s1) # run an update iteration on one mini-batch
      learner.train_losses.append(loss)

      # copy the updated net onto the target generating net
      if i % learner.target_net_update == 0:
          learner.target_net = deepcopy(learner.net)

      # print overview metrics every fixed number of iterations
      if i % self.print_every == 0:   
        print('progress %.3f, epoch %.2f, iteration %d, train loss %.3f' % ((i/float(iterations)),epoch,i,loss))

      # every epoch run evaluation and save the net
      if i % iterations_per_epoch == 0:

        val_loss, qval_avg, policy_reward = self.evaluate(learner, memory, 'val')

        learner.val_losses.append(val_loss)
        learner.qval_avgs.append(qval_avg)
        learner.policy_rewards.append(policy_reward)

        if epoch % self.save_every == 0:
          self.save(learner.net,'./%s/net_%d.p' % (self.save_dir,int(epoch)))

        print('%d, epoch %.2f, validation loss %.3f, average policy reward %.3f, average Q-value %.3f' % ((i/float(iterations)), 
          epoch, val_loss, policy_reward, qval_avg))

    # save the DQN object together with training history at the end of training
    self.save(self,'./%s/DQN_final.p' % self.save_dir)

  # helper function to save net (or any object)
  def save(self,obj,name):
    pickle.dump(obj, open(name, "wb"))

  # helper function to load net (or any object)
  def load(self,name):
    return pickle.load(open(name, "rb"))

  # function to do evaluation - similar to "iteration" function
  def evaluate(self,learner,memory,split,n=float('nan')):

    if math.isnan(n):
      n = int(math.ceil(memory.data_size[split] / self.batch_size))

    loss = qval_avg = policy_reward = 0

    for i in range(n):

      s0,a,r,s1 = memory.minibatch(split,self.batch_size)

      # move forward through the net and estimate the loss gradient + loss + qval_avg
      approx_q_all, batch_loss, batch_qval_avg = learner.forward_loss(s0, a, r, s1)

      # avg. reward for values we 
      opt_a = np.argmax(approx_q_all.data,1)
      ind = opt_a == a
      # if, in the mini-batch, at least one action complies with policy recommendations
      if sum(ind) > 0:
        policy_reward += np.mean(r[ind])

      qval_avg += batch_qval_avg
      loss += batch_loss

    policy_reward /= n
    qval_avg /= n
    loss /= n

    return loss, qval_avg, policy_reward
