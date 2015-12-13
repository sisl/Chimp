import numpy as np
import chainer
import chainer.functions as F
from chainer import optimizers
from chainer import cuda
from copy import deepcopy
from lspi import lstdq
import numpy.linalg as la

import pickle # used to save the nets

class LearnerLSPI(object):

    def __init__(self, settings):

        self.settings = settings

        self.maxexp = 1

        self.target_net_update=1

        self.n_actions = settings['n_actions']
        self.model_dims = settings['model_dims']
        self.learning_rate = settings['learning_rate']
        self.gamma = settings['discount']
        self.clip_reward = settings['clip_reward']

        self.train_losses = []
        self.train_rewards = []
        self.train_qval_avgs = []
        self.train_times = []
        self.val_losses = []
        self.val_rewards = []
        self.val_qval_avgs = []
        self.val_times = []

        self.net = np.zeros(self.n_actions*np.prod(self.model_dims), dtype=np.float32)

        self.overall_time = 0

    def linear_policy(self, w, s):
        return np.argmax([np.dot(w,self.phi(s,a)) for a in range(self.n_actions)])
    
    def phi(self, s, a):
        features = np.zeros(np.prod(s.shape)*self.n_actions)
        features[int(a * np.prod(s.shape)) : int(a * np.prod(s.shape)) + int(np.prod(s.shape))] = np.reshape(s, np.prod(s.shape))
        return features

    # sampling one mini-batch and one update based on it
    def gradUpdate(self, s0, ahist0, a, r, s1, ahist1, episode_end_flag):

        if self.clip_reward:
            r = np.clip(r,-self.clip_reward,self.clip_reward)

        # move forward through the net and produce output variable 
        # containing the loss gradient + MSE loss + average Q-value for the actions taken
        approx_q_all, loss, qval_avg = self.forwardLoss(s0, ahist0, a, r, s1, ahist1, episode_end_flag)

        return approx_q_all, loss, qval_avg

    # function to get net output and to calculate the loss
    def forwardLoss(self, s0, ahist0, a, r, s1, ahist1, episode_end_flag):

        D = []
        for i in range(s0.shape[0]):
            D.append((s0[i], a[i], r[i], s1[i], None))

        self.A,self.b,self.net,info = lstdq.LSTDQ(D,env=self,w=self.net,damping=0.001)

        approx_q_all = np.zeros((s0.shape[0],self.n_actions))
        for i in range(s0.shape[0]):
            for j in range(self.n_actions):
                approx_q_all[i,j] = np.dot(self.phi(s0[i],j), self.net)

        return approx_q_all, 0, np.mean(np.max(approx_q_all,1))

    # extract the optimal policy in the given state
    def policy(self, s, ahist):

        approx_q_all = np.zeros((s.shape[0],self.n_actions))
        for i in range(s.shape[0]):
            for j in range(self.n_actions):
                approx_q_all[i,j] = np.dot(self.phi(s[i],j), self.net)

        opt_a = np.argmax(approx_q_all,1)

        return opt_a

    # function to update target net with the current net
    def net_to_target_net(self):
        None

    # function to load in a new net from a variable
    def load_net(self,net):
        self.net = deepcopy(net)

    # collect net parameters (coefs and grads)
    def params(self, net):
        self.net
