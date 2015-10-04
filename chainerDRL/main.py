import argparse
import numpy as np
import random
import math
from copy import deepcopy

import chainer
import chainer.functions as F
from chainer import optimizers

from sample import Sampler
from DQNs import DQN


# Initialize random seed for reproducibility
np.random.seed(1234)

print('Setting training parameters...')
# Set training settings
settings = { 'batch_size' : 200, # mini-batch size
    'n_epochs' : 200, # number of training epochs
    'learning_rate' : 0.001, 
    'decay_rate' : 0.99, # decay rate for RMSprop, otherwise not used
    'discount' : 0.99, # discount rate for RL
    'clip_err' : 0.1, # value to clip loss gradients to
    'clip_reward' : False, # value to clip reward values to
    'target_net_update' : 10000, # update the update-generating target net every fixed number of iterations
    'print_every' : 5000, # print out update every 5000 iterations
    'save_every' : 1, # every # of epochs to save the net
    'save_dir' : 'models', # every # of epochs to save the net
    'double_DQN' : False, # use Double Deep Q-learning ? (CURRENTLY NOT SUPPORTED)
    'optim_name' : 'RMSprop' # currently supports "RMSprop" and "SGD"'
    }


print('Loading data...')
# DATA LOADING (THIS SECTION AND A SAMPLER OBJECT WILL BE REPLACED WITH EXP. REPLAY)
train_data = np.genfromtxt('./data/kdd1998tuples_train.csv', delimiter=',',skip_header=0)
val_data =  np.genfromtxt('./data/kdd1998tuples_val.csv', delimiter=',',skip_header=0)
test_data =  np.genfromtxt('./data/kdd1998tuples_test.csv', delimiter=',',skip_header=0)

# chainer currently supports only 32 architecture
train_data = train_data.astype(np.float32)
val_data = val_data.astype(np.float32)
test_data = test_data.astype(np.float32)

data = {}
data['train'] = train_data
data['val'] = val_data 
data['test'] = test_data

# generate sampler object to sample mini-batches
sampler = Sampler(data, settings['batch_size'])


# NETWORK DEFINITION

print('Setting up network...')
# set parameters that define the network
n_units = 100
n_states = 5
n_actions = 12

# define the core layer structure of the network
model = chainer.FunctionSet(l1=F.Linear(n_states, n_units),
    l2=F.Linear(n_units, n_units/2),
    l3=F.Linear(n_units/2, n_units),
    l4=F.Linear(n_units, n_units/2),
    l5=F.Linear(n_units/2, n_actions))

# define forward pass that specifies all extra activation functions and how the net produces output
# on the way, the network also memorizes how to run the backward pass through all the layers
# this memory is stored in the output variable
def forward(model, s):
    h1 = F.relu(model.l1(s))
    h2 = F.relu(model.l2(h1))
    h3 = F.relu(model.l3(h2))    
    h4 = F.relu(model.l4(h3))
    output = model.l5(h4)
    return output

# form a DQN object that will handle the training
print('Initializing controller...')
dqn = DQN(model, forward, settings)

# launch training using data from the sampler object
print('Starting training...')
dqn.train(sampler)
