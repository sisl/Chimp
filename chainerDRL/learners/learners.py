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

    def __init__(self, settings):

        self.settings = settings

        self.gpu = settings['gpu']
        self.learning_rate = settings['learning_rate']
        self.decay_rate = settings['decay_rate']
        self.discount = settings['discount']
        self.clip_err = settings['clip_err']
        self.clip_reward = settings['clip_reward']
        self.target_net_update = settings['target_net_update']
        self.double_DQN = settings['double_DQN']
        self.reward_rescale = settings['reward_rescale']
        self.r_max = 1 # keep the default value at 1

        # setting up various possible gradient update algorithms
        if settings['optim_name'] == 'RMSprop':
            self.optimizer = optimizers.RMSprop(lr=self.learning_rate, alpha=self.decay_rate)

        elif settings['optim_name'] == 'ADADELTA':
            print("Supplied learning rate not used with ADADELTA gradient update method")
            self.optimizer = optimizers.AdaDelta()

        elif settings['optim_name'] == 'ADAM':
            self.optimizer = optimizers.Adam(alpha=self.learning_rate)

        elif settings['optim_name'] == 'SGD':
            self.optimizer = optimizers.SGD(lr=self.learning_rate)

        else:
            print('The requested optimizer is not supported!!!')
            exit()

        self.optim_name = settings['optim_name']

        self.train_losses = []
        self.train_rewards = []
        self.train_qval_avgs = []
        self.train_episodes = []
        self.train_times = []

        self.val_losses = []
        self.val_rewards = []
        self.val_qval_avgs = []
        self.val_episodes = []
        self.val_times = []

        self.overall_time = 0


    def gradUpdate(self, s0, ahist0, a, r, s1, ahist1, episode_end_flag):
        ''' one gradient update step based on a given mini-batch '''

        if self.clip_reward:
            r = np.clip(r,-self.clip_reward,self.clip_reward)

        if self.reward_rescale:
            self.r_max = max(np.amax(np.absolute(r)),self.r_max) # r_max originally equal to 1

        self.net.zerograds()  # reset gradient storage to zero

        # move forward through the net and produce output variable 
        # containing the loss gradient + MSE loss + average Q-value for the actions taken
        approx_q_all, loss, qval_avg = self.forwardLoss(s0, ahist0, a, r/self.r_max, s1, ahist1, episode_end_flag)

        approx_q_all.backward() # propagate the loss gradient through the net

        self.optimizer.update() # carry out parameter updates based on the distributed gradients

        return approx_q_all, loss, qval_avg


    def forwardLoss(self, s0, ahist0, a, r, s1, ahist1, episode_end_flag):
        ''' get network output, calculate the loss and the average Q-value'''

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

        # calculate the expected Q-values for all actions
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

    def policy(self, s, ahist):
        ''' extract the optimal policy in the given state and action history '''

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

    def copy_net_to_target_net(self):
        ''' update target net with the current net '''
        self.target_net = deepcopy(self.net)

    def save(self,obj,name):
        pickle.dump(obj, open(name, "wb"))

    def load(self,name):
        return pickle.load(open(name, "rb"))

    def save_net(self,name):
        ''' save a net to a path '''
        self.save(self.net,name)

    def load_net(self,net):
        ''' load in a net from path or a variable'''
        if isinstance(net, str): # if it is a string, load the net from the path
            net = self.load(net)

        self.net = deepcopy(net)
        if self.gpu:
            cuda.get_device(0).use()
            self.net.to_gpu()
        self.target_net = deepcopy(self.net)

        self.optimizer.setup(self.net)

    def save_training_history(self, path='.'):
        ''' save training history '''
        train_hist = np.array([range(len(self.train_rewards)),self.train_losses,self.train_rewards, self.train_qval_avgs, self.train_episodes, self.train_times]).T
        eval_hist = np.array([range(len(self.val_rewards)),self.val_losses,self.val_rewards, self.val_qval_avgs, self.val_episodes, self.val_times]).T
        # TODO: why is this here and not in agent?
        np.savetxt(path + '/training_hist.csv', train_hist, delimiter=',')
        np.savetxt(path + '/evaluation_hist.csv', eval_hist, delimiter=',')

    def params(self):
        ''' collect net parameters (coefs and grads) '''
        self.net.collect_parameters()
