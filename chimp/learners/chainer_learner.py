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

class ChainerLearner(object):

    def __init__(self, settings, net = None):

        self.set_params(settings)

        self.source_net = None
        self.target_net = None
        if net is not None:
            self.set_net(net)


    def update(self, obs, a, r, obsp, term):
        """
        Performs a single mini-batch update
        """

        self.source_net.zerograds()  # reset gradient storage to zero

        # compute loss and qval output layer
        loss, qvals = self.forward_loss(obs, a, r, obsp, term)

        qvals.backward() # propagate the loss gradient through the net
        self.optimizer.update() # carry out parameter updates based on the distributed gradients

        return loss, qvals 


    def forward_loss(self, obs, a, r, obsp, term):
        """
        Computes the loss and gradients
        """
        if self.gpu:
            return self.forward_loss_gpu(obs, a, r, obsp, term)
        else:
            return self.forward_loss_cpu(obs, a, r, obsp, term)


    def forward_loss_gpu(self, obs, a, r, obsp, term):
        # unpack
        ohist, ahist = obs
        ophist, aphist = obsp

        # move to GPU
        ohist, ahist = self.to_gpu(ohist), self.to_gpu(ahist)
        ophist, aphist = self.to_gpu(ophist), self.to_gpu(aphist)

        # transfer inputs into Chainer format
        ohist, ophist = chainer.Variable(ohist), chainer.Variable(ophist, volatile = True)
        ahist, aphist = chainer.Variable(ahist), chainer.Variable(aphist, volatile = True)

        # get target Q
        target_q_all = self.target_net(ophist, aphist) # forward prop
        target_q_max = np.max(target_q_all.data.get(), axis=1) # max Q for each entry in mini-batch

        # compute the target values for each entry in mini-batch 
        target_q_vals = r + self.discount * target_q_max * term 

        # compute the source q-vals
        source_q_all = self.source_net(ohist, ahist) # forward prop
        source_q_vals = source_q_all.data.get()[np.arange(source_q_all.data.shape[0]), a]

        # compute the loss grads
        qdiff = source_q_vals - target_q_vals 

        # distribute the loss gradient into the shape of the net's output
        dQ = np.zeros(approx_q_all.data.shape, dtype=np.float32) 
        dQ[np.arange(dQ.shape[0]), a] = qdiff

        # set as the output grad layer
        source_q_all.grad = self.to_gpu(dQ)

        # compute loss
        loss = np.mean(dQ**2)

        return loss, source_q_all


    def forward_loss_cpu(self, obs, a, r, obsp, term):
        # unpack
        ohist, ahist = obs
        ophist, aphist = obsp

        # transfer inputs into Chainer format
        ohist, ophist = self.chainer_var(ohist), self.chainer_var(ophist, volatile = True)
        ahist, aphist = self.chainer_var(ahist), self.chainer_var(aphist, volatile = True)

        # get target Q
        target_q_all = self.target_net(ophist, aphist)
        target_q_max = np.max(target_q_all.data, axis=1)

        # compute the target values for each entry in mini-batch 
        target_q_vals = r + self.discount * target_q_max * term 

        # compute the source q-vals
        source_q_all = self.source_net(ohist, ahist) # forward prop
        source_q_vals = source_q_all.data[np.arange(source_q_all.data.shape[0]),a]

        # compute the loss
        qdiff = source_q_vals - target_q_vals 

        # distribute the loss gradient into the shape of the net's output
        dQ = np.zeros(source_q_all.data.shape, dtype=np.float32) 
        dQ[np.arange(dQ.shape[0]), a] = qdiff

        # set as the output grad layer
        source_q_all.grad = dQ

        # compute loss
        loss = np.mean(dQ**2)

        return loss, source_q_all


    def forward(self, obs):
        """
        Returns the Q-values for the network input obs
        """
        # turn train off for bn, dropout, etc
        self.source_net.train = False
        if self.gpu:
            return self.forward_gpu(obs)
        else:
            return self.forward_cpu(obs)


    def forward_cpu(self, obs):
        """
        Performs forward pass on CPU, returns Q values
        """
        # unpack
        ohist, ahist = obs
        # transfer inputs into Chainer format
        ohist, ahist = self.chainer_var(ohist, volatile=True), self.chainer_var(ahist, volatile=True)
        # evaluate
        qvals = self.source_net(ohist, ahist)
        return qvals.data

    def forward_gpu(self, obs):
        """
        Performs forward pass on CPU, returns Q values
        """
        # unpack
        ohist, ahist = obs
        # move to gpu
        ohist, ahist = self.to_gpu(ohist), self.to_gpu(ahist)
        # transfer inputs into Chainer format
        ohist, ahist = self.chainer_var(ohist, volatile=True), self.chainer_var(ahist, volatile=True)
        # evaluate
        qvals = self.source_net(ohist, ahist)
        return qvals.data.get()

    #################################################################  
    #################### Utility Functions ##########################
    #################################################################  

    def to_gpu(self, var):
        if obs is None:
            return None
        return cuda.to_gpu(var)

    def chainer_var(self, var, volatile=False):
        if var is None:
            return None
        return chainer.Variable(var, volatile=volatile)

    def set_net(self, net):
        self.source_net = deepcopy(net)
        self.target_net = deepcopy(net)
        if self.gpu:
            cuda.get_device(0).use()
            self.source_net.to_gpu()
            self.target_net.to_gpu()
        self.optimizer.setup(self.source_net)
        self.target_net.train = False


    def params(self):
        ''' collect net parameters (coefs and grads) '''
        self.source_net.params()

    
    def set_params(self, params):

        self.gpu = params['gpu']
        self.learning_rate = params['learning_rate']
        self.decay_rate = params['decay_rate']
        self.discount = params['discount']
        self.clip_err = params['clip_err']
        self.target_net_update = params['target_net_update']
        self.double_DQN = params['double_DQN']

        # setting up various possible gradient update algorithms
        if params['optim_name'] == 'RMSprop':
            self.optimizer = optimizers.RMSprop(lr=self.learning_rate, alpha=self.decay_rate)

        elif params['optim_name'] == 'ADADELTA':
            print("Supplied learning rate not used with ADADELTA gradient update method")
            self.optimizer = optimizers.AdaDelta()

        elif params['optim_name'] == 'ADAM':
            self.optimizer = optimizers.Adam(alpha=self.learning_rate)

        elif params['optim_name'] == 'SGD':
            self.optimizer = optimizers.SGD(lr=self.learning_rate)

        else:
            print('The requested optimizer is not supported!!!')
            exit()

        if self.clip_err is not False:
            self.optimizer.add_hook(chainer.optimizer.GradientClipping(self.clip_err))

        self.optim_name = params['optim_name']
