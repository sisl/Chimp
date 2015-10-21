import argparse
import numpy as np
import random
import math
from copy import deepcopy
import scipy.misc as spm

import chainer
import chainer.functions as F
from chainer import optimizers

from ale_python_interface import ALEInterface

from replay import ReplayMemory
from learners import Learner
from agents import Agent

print('Setting training parameters...')
# Set training settings
settings = {
    # agent settings
    'batch_size' : 32, # mini-batch size
    'print_every' : 1000, # print out update every 5000 iterations
    'save_dir' : 'nets', # directory where we save the net
	'eval_every' : 10,
	'eval_episodes' : 5,
	'save_every' : 100,
	'n_episodes' : 10000,

    # Atari simulator settings
    'epsilon' : 1.0,  # Initial exploratoin rate
    'initial_exploration' : 10000,
    'frame_skip' : 4,
    'viz' : True,
    #'rom' : "./roms/breakout.bin",
    'rom' : "./roms/boxing.bin",
    'screen_dims_new' : (84,84),
    'display_dims' : (320,420),
    'pad' : 15,

    # replay memory settings
    'memory_size' : 100000,  # size of replay memory
    'n_frames' : 4,  # number of frames

    # learner settings
    'learning_rate' : 0.0001, 
    'decay_rate' : 0.99, # decay rate for RMSprop, otherwise not used
    'discount' : 0.95, # discount rate for RL
    'clip_err' : False, # value to clip loss gradients to
    'clip_reward' : False, # value to clip reward values to
    'target_net_update' : 10000, # update the update-generating target net every fixed number of iterations
    'double_DQN' : True, # use Double DQN (based on Deep Mind paper)
    'optim_name' : 'RMSprop', # currently supports "RMSprop", "ADADELTA" and "SGD"'

    # general
    'seed' : 1234
    }

print(settings)

# Initialize random seed for reproducibility
np.random.seed(settings["seed"])
random.seed(settings["seed"])


# SETTING UP ATARI SIMULATOR

print('Firing up Atari...')

ale = ALEInterface()

ale.setInt("frame_skip",settings["frame_skip"])
ale.setInt("random_seed",settings["seed"])
ale.loadROM(settings["rom"])

settings['actions'] = ale.getLegalActionSet() #Minimal
settings['n_actions'] = settings['actions'].size
settings['screen_dims'] = ale.getScreenDims()

print("Original screen width/height: " +str(settings['screen_dims'][0]) + "/" + str(settings['screen_dims'][1]))
print("Modified screen width/height: " +str(settings['screen_dims_new'][0]) + "/" + str(settings['screen_dims_new'][1]))

# SETTING UP THE REPLAY MEMORY

print('Initializing replay memory...')

# Generate sampler object to sample mini-batches
memory = ReplayMemory(settings)


# SETTING UP THE LEARNER
# Define the network to be used by the learner

print('Setting up networks...')
# Set parameters that define the network

# Define the core layer structure of the network
net = chainer.FunctionSet(
    l1=F.Convolution2D(settings['n_frames'], 32, ksize=8, stride=4, nobias=False, wscale=np.sqrt(2)),
    l2=F.Convolution2D(32, 64, ksize=4, stride=2, nobias=False, wscale=np.sqrt(2)),
    l3=F.Convolution2D(64, 64, ksize=3, stride=1, nobias=False, wscale=np.sqrt(2)),
    l4=F.Linear(3136, 512, wscale=np.sqrt(2)),
    l5=F.Linear(512, settings['n_actions'], wscale = np.sqrt(2))
    )

# Define forward pass that specifies all extra activation functions and how the net produces output
# on the way, the network also memorizes how to run the backward pass through all the layers
# this memory is stored in the output variable
def forward(net, s):
    h1 = F.relu(net.l1(s))
    h2 = F.relu(net.l2(h1))
    h3 = F.relu(net.l3(h2))    
    h4 = F.relu(net.l4(h3))
    output = net.l5(h4)
    return output

# initialize Learner object that will handle the net updates and policy extraction
print('Initializing the learner...')
learner = Learner(net, forward, settings)

# SETTING UP THE EXPERIMENT

# initialize Agent
print('Initializing the agent framework...')
agent = Agent(settings)

print('Training...')
# starting training
agent.train(learner, memory, ale)

