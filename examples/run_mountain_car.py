"""
File to initialize training.
Contains settings, network definition for Chainer.
Creates the simulator, replay memory, DQN learner, and passes these to the agent framework for training.
"""

import numpy as np

import chainer
import chainer.functions as F
import chainer.links as L
from chainer import cuda, Function, gradient_check, Variable, optimizers, serializers, utils
from chainer import Link, Chain, ChainList

# Memory
from chimp.memories import ReplayMemoryHDF5

# Learner (Brain)
from chimp.learners.dqn_learner import DQNLearner
from chimp.learners.chainer_backend import ChainerBackend

# Agent Framework
from chimp.agents import DQNAgent

# Simulator
from chimp.simulators.mdp.mdp_simulator import MDPSimulator
from chimp.simulators.mdp.mountain_car import MountainCar 

# Rollout Policy
from chimp.utils.policies import RandomPolicy


settings = {

    # agent settings
    'batch_size' : 32,
    'print_every' : 1000,
    'save_dir' : 'results/mountain_car',
    'iterations' : 200000,
    'eval_iterations' : 100,
    'eval_every' : 1000,
    'save_every' : 20000,
    'initial_exploration' : 50000,
    'epsilon_decay' : 0.000001, # subtract from epsilon every step
    'eval_epsilon' : 0, # epsilon used in evaluation, 0 means no random actions
    'epsilon' : 1.0,  # Initial exploratoin rate
    'learn_freq' : 1,
    'history_sizes' : (1, 0, 0), # sizes of histories to use as nn inputs (o, a, r)
    'model_dims' : (1,2), 

    # simulator settings
    'viz' : False,

    # replay memory settings
    'memory_size' : 100000,  # size of replay memory
    'n_frames' : 1,  # number of frames

    # learner settings
    'learning_rate' : 0.00001, 
    'decay_rate' : 0.99, # decay rate for RMSprop, otherwise not used
    'discount' : 0.95, # discount rate for RL
    'clip_err' : False, # value to clip loss gradients to
    'clip_reward' : False, # value to clip reward values to
    'target_net_update' : 1000, # update the update-generating target net every fixed number of iterations
    'double_DQN' : False, # use Double DQN (based on Deep Mind paper)
    'optim_name' : 'ADAM', # currently supports "RMSprop", "ADADELTA", "ADAM" and "SGD"'
    'gpu' : False,
    'reward_rescale': False,

    # general
    'seed_general' : 1723,
    'seed_simulator' : 5632,
    'seed_agent' : 9826,
    'seed_memory' : 7563

    }

mdp = MountainCar()
simulator = MDPSimulator(mdp)

class CarNet(Chain):

    def __init__(self):
        super(CarNet, self).__init__(
            l1=F.Linear(settings['model_dims'][1], 20, bias=0.0),
            l2=F.Linear(20, 10, bias=0.0),
            bn1=L.BatchNormalization(10),
            l3=F.Linear(10, 10),
            l4=F.Linear(10, 10),
            bn2=L.BatchNormalization(10),
            lout=F.Linear(10, simulator.n_actions)
        )
        self.train = True
        # initialize avg_var to prevent divide by zero
        self.bn1.avg_var.fill(0.1),
        self.bn2.avg_var.fill(0.1),


    def __call__(self, ohist, ahist):
        h = F.relu(self.l1(ohist))
        h = F.relu(self.l2(h))
        h = self.bn1(h, test=not self.train)
        h = F.relu(self.l3(h))
        h = F.relu(self.l4(h))
        h = self.bn2(h, test=not self.train)
        output = self.lout(h)
        return output


net = CarNet()

# Initialize Learner with a Chainer backend
backend = ChainerBackend(settings)
backend.set_net(net)
learner = DQNLearner(settings, backend)

# Initialize memory
memory = ReplayMemoryHDF5(settings)

# Initialize Agent Framework
agent = DQNAgent(learner, memory, simulator, settings)

# Start training
agent.train(verbose=True)
