import argparse
import numpy as np
import random
import math
from copy import deepcopy

import chainer
import chainer.functions as F
from chainer import optimizers

from replay import ReplayMemory
from learners import Learner
from agents import Agent


# Initialize random seed for reproducibility
np.random.seed(1234)
random.seed(1234)

print('Setting training parameters...')
# Set training settings
settings = {
    # agent settings
    'batch_size' : 200, # mini-batch size
    'n_epochs' : 200, # number of training epochs
    'print_every' : 5000, # print out update every 5000 iterations
    'save_every' : 1, # every number of epochs that we save the net
    'save_dir' : 'nets', # directory where we save the net

    # learner settings
    'learning_rate' : 0.0001, 
    'decay_rate' : 0.99, # decay rate for RMSprop, otherwise not used
    'discount' : 0.95, # discount rate for RL
    'clip_err' : False, # value to clip loss gradients to
    'clip_reward' : False, # value to clip reward values to
    'target_net_update' : 10000, # update the update-generating target net every fixed number of iterations
    'double_DQN' : True, # use Double DQN (based on Deep Mind paper)
    'optim_name' : 'RMSprop' # currently supports "RMSprop", "ADADELTA" and "SGD"'
    }
print(settings)


print('Loading data...')
# DATA LOADING (THIS SECTION AND A SAMPLER OBJECT WILL BE REPLACED WITH EXP. REPLAY)
train_data = np.genfromtxt('./data/kdd1998tuples_train.csv', delimiter=',',skip_header=0)
val_data =  np.genfromtxt('./data/kdd1998tuples_val.csv', delimiter=',',skip_header=0)
test_data =  np.genfromtxt('./data/kdd1998tuples_test.csv', delimiter=',',skip_header=0)

# Chainer currently supports only 32 architecture
train_data = train_data.astype(np.float32)
val_data = val_data.astype(np.float32)
test_data = test_data.astype(np.float32)

data = {}
data['train'] = train_data
data['val'] = val_data 
data['test'] = test_data

# Generate sampler object to sample mini-batches
memory = ReplayMemory(data)


# Define the network to be used by the learner

print('Setting up networks...')
# Set parameters that define the network
n_units = 100
n_states = 5
n_actions = 12

# Define the core layer structure of the network
net = chainer.FunctionSet(l1=F.Linear(n_states, n_units,wscale = np.sqrt(2)),
    l2=F.Linear(n_units, n_units/2,wscale = np.sqrt(2)),
    l3=F.Linear(n_units/2, n_units,wscale = np.sqrt(2)),
    l4=F.Linear(n_units, n_units/2,wscale = np.sqrt(2)),
    l5=F.Linear(n_units/2, n_actions,wscale = np.sqrt(2)))

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

# initialize Agent
print('Initializing the agent framework...')
agent = Agent(settings)

# launch training using data from the sampler object
print('Starting training...')
agent.train(learner,memory)
