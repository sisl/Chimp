'''
(Double) Deep Q-Learning Algorithm Implementation
Supports double deep Q-learning with on either GPU and CPU

'''

import numpy as np
import chainer
import chainer.functions as F
from chainer import optimizers
from chainer import cuda
from copy import deepcopy

import pickle # used to save the nets

class Learner(object):

    def __init__(self, net, settings):

        self.settings = settings
        
        self.net = net
        
        self.gpu = settings['gpu']
        if self.gpu:
            cuda.get_device(0).use()
            self.net.to_gpu()
            print("Deep learning on GPU ...")
        else:
            print("Deep learning on CPU ...")

        self.target_net = deepcopy(self.net)

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

        elif settings['optim_name'] == 'ADAM':
            self.optimizer = optimizers.Adam()

        elif settings['optim_name'] == 'SGD':
            self.optimizer = optimizers.SGD(lr=self.learning_rate)

        else:

            print('The requested optimizer is not supported!!!')
            exit()

        self.optim_name = settings['optim_name']
        self.optimizer.setup(self.net)

        self.train_losses = []
        self.train_rewards = []
        self.train_qval_avgs = []
        self.train_times = []
        self.val_losses = []
        self.val_rewards = []
        self.val_qval_avgs = []
        self.val_times = []

        self.overall_time = 0

    # sampling one mini-batch and one update based on it
    def gradUpdate(self, s0, ahist0, a, r, s1, ahist1, episode_end_flag):

        if self.clip_reward:
            r = np.clip(r,-self.clip_reward,self.clip_reward)

        self.net.zerograds()  # reset gradient storage to zero

        # move forward through the net and produce output variable 
        # containing the loss gradient + MSE loss + average Q-value for the actions taken
        approx_q_all, loss, qval_avg = self.forwardLoss(s0, ahist0, a, r, s1, ahist1, episode_end_flag)

        approx_q_all.backward() # propagate the loss gradient through the net

        self.optimizer.update() # carry out parameter updates based on the distributed gradients

        return approx_q_all, loss, qval_avg

    # function to get net output and to calculate the loss
    def forwardLoss(self, s0, ahist0, a, r, s1, ahist1, episode_end_flag):

        if self.gpu:
            s0 = cuda.to_gpu(s0)
            s1 = cuda.to_gpu(s1)
            ahist0 = cuda.to_gpu(ahist0)
            ahist1 = cuda.to_gpu(ahist1)

        # transfer states into Chainer format
        s0, s1 = chainer.Variable(s0), chainer.Variable(s1, volatile = True)
        ahist0, ahist1 = chainer.Variable(ahist0), chainer.Variable(ahist1, volatile = True)

        # calculate target Q-values (from s1 and on)
        if not self.double_DQN:
            target_q_all = self.target_net(s1, ahist1)
            if self.gpu:
                target_q_max = np.max(target_q_all.data.get(), 1)
            else:
                target_q_max = np.max(target_q_all.data, 1)

        else:
            # when we do double Q-learning
            # use the current network to determine optimal action in the next state (argmax)
            target_q_all = self.net(s1, ahist1)
            if self.gpu:
                target_argmax = np.argmax(target_q_all.data.get(), 1)
            else:
                target_argmax = np.argmax(target_q_all.data, 1)

            # use the target network to determine the value of the selected optimal action in the next state
            target_q_all = self.target_net(s1, ahist1)
            if self.gpu:
                target_q_max = target_q_all.data.get()[np.arange(target_q_all.data.shape[0]),target_argmax]
            else:
                target_q_max = target_q_all.data[np.arange(target_q_all.data.shape[0]),target_argmax]
      
        # one reinforcement learning back-track
        # 'end' here tells us to zero out the step from the future in terminal states
        target_q_value = r + self.discount * target_q_max * (1-1*episode_end_flag)

        # calculate expected Q-values for all actions
        approx_q_all = self.net(s0, ahist0)

        # extract expected Q-values for the actions we actually took
        if self.gpu:
            approx_q_value = approx_q_all.data.get()[np.arange(approx_q_all.data.shape[0]),a]
        else:
            approx_q_value = approx_q_all.data[np.arange(approx_q_all.data.shape[0]),a]

        # calculate the loss gradient
        gradLoss = approx_q_value - target_q_value

        # clip the loss gradient
        if self.clip_err:
            gradLoss = np.clip(gradLoss,-self.clip_err,self.clip_err)

        # distribute the loss gradient into the shape of the net's output
        gradLossAll = np.zeros(approx_q_all.data.shape, dtype=np.float32)
        gradLossAll[np.arange(gradLossAll.shape[0]),a] = gradLoss

        # transfer the loss gradient
        if self.gpu:
            approx_q_all.grad = cuda.to_gpu(gradLossAll)
        else:
            approx_q_all.grad = gradLossAll

        return approx_q_all, np.mean(gradLoss**2), np.mean(approx_q_value)

    # extract the optimal policy in the given state
    def policy(self, s, ahist):

        if self.gpu:
            s = cuda.to_gpu(s)
            ahist = cuda.to_gpu(ahist)
        s = chainer.Variable(s, volatile = True)
        ahist = chainer.Variable(ahist, volatile = True)

        # get all Q-values for given state(s)
        approx_q_all = self.net(s, ahist)

        # pick actions that maximize Q-values in each state
        if self.gpu:
            opt_a = np.argmax(approx_q_all.data.get(),1)
        else:
            opt_a = np.argmax(approx_q_all.data,1)

        return opt_a

    # function to update target net with the current net
    def net_to_target_net(self):
        self.target_net = deepcopy(self.net)

    # function to load in a new net from a variable
    def load_net(self,net):
        self.net = deepcopy(net)
        if self.gpu:
            self.net.to_gpu()
        self.target_net = deepcopy(self.net)

    # collect net parameters (coefs and grads)
    def params(self, net):
        self.net.collect_parameters()
