'''
(Double) Deep Q-Learning Algorithm Implementation
Supports double deep Q-learning with on either GPU and CPU

'''

import numpy as np
import pickle # used to save the nets
from copy import deepcopy

class DQNLearner(object):

    def __init__(self, settings, backend):

        """
        Functions that must be defined by the custom learner:
        - forward_loss(obs, a, r, obsp, term) # computes scores and loss
        - forward(obs) # computes scores
        - update(obs, a, r, obsp) # update the params
        - get_net() # returns the network object
        - set_net(net) # sets the source and target nets and moves to gpu (if needed)
        Fields owned by the learner:
        - source_net: generates source Q-vals
        - target_net: generates target Q-vals
        """

        self.backend = backend

        self.clip_reward = settings.get('clip_reward', False)
        self.reward_rescale = settings.get('reward_rescale', False)
        self.r_max = 1 # keep the default value at 1


    def update(self, obs, a, r, obsp, term):
        r = self.pre_process_reward(r)
        return self.backend.update(obs, a, r, obsp, term)

    def forward_loss(self, obs, a, r, obsp, term):
        return self.backend.forward_loss(obs, a, r, obsp, term)

    def forward(self, obs):
        return self.backend.forward(obs)

    def copy_net_to_target_net(self):
        ''' update target net with the current net '''
        self.backend.target_net = deepcopy(self.backend.source_net)

    def save(self,obj,name):
        pickle.dump(obj, open(name, "wb"))

    def load(self,name):
        return pickle.load(open(name, "rb"))

    def save_net(self,name):
        ''' save a net to a path '''
        self.save(self.backend.source_net,name)

    def load_net(self,net):
        ''' load in a net from path or a variable'''
        if isinstance(net, str): # if it is a string, load the net from the path
            net = self.load(net)
        self.backend.set_net(net)


    def save_training_history(self, path='.'):
        ''' save training history '''
        train_hist = np.array([range(len(self.train_rewards)),self.train_losses,self.train_rewards, self.train_qval_avgs, self.train_episodes, self.train_times]).T
        eval_hist = np.array([range(len(self.val_rewards)),self.val_losses,self.val_rewards, self.val_qval_avgs, self.val_episodes, self.val_times]).T
        # TODO: why is this here and not in agent?
        np.savetxt(path + '/training_hist.csv', train_hist, delimiter=',')
        np.savetxt(path + '/evaluation_hist.csv', eval_hist, delimiter=',')

    def params(self):
        """
        Returns an iterator over netwok parameters 
        Note: different back-ends will return different param containers
        """
        # TODO: return a dictionary here?
        self.backend.params()
        

    def pre_process_reward(self, r):
        """
        Clips and re-scales the rewards 
        """
        if self.clip_reward:
            r = np.clip(r,-self.clip_reward,self.clip_reward)
        if self.reward_rescale:
            self.r_max = max(np.amax(np.absolute(r)),self.r_max) 
            r = r / self.r_max
        return r

