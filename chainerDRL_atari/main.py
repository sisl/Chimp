'''
File to initialize training.
Contains settings, network definition for Chainer.
Creates the simulator, replay memory, DQN learner, and passes these to the agent framework for training.
'''

import numpy as np
import random

import chainer
import chainer.functions as F

from memories import ReplayMemoryHDF5
from memories import ReplayMemory
from learners import Learner
from agents import Agent
from simulators import Atari


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
    'rom' : "./roms/breakout.bin",
    'screen_dims_new' : (84,84), # size to which the image shall be cropped
    'display_dims' : (170,170), # size of the pygame display
    'pad' : 15, # padding parameter - for image cropping - only along the length of the image, to obtain a square

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
    'double_DQN' : False, # use Double DQN (based on Deep Mind paper)
    'optim_name' : 'RMSprop', # currently supports "RMSprop", "ADADELTA" and "SGD"'
    'cuda' : False,

    # general
    'seed' : 1234

    }

print(settings)

# Initialize random seed for reproducibility
np.random.seed(settings["seed"])
random.seed(settings["seed"])


# SETTING UP ATARI SIMULATOR

print('Firing up Atari...')

ale = Atari(settings)

# SETTING UP THE REPLAY MEMORY
 
print('Initializing replay memory...')

# Generate sampler object to sample mini-batches
# memory = ReplayMemoryHDF5(settings)
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
    l5=F.Linear(512, ale.n_actions, wscale = np.sqrt(2))
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
