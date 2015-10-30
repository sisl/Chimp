'''
File to initialize training.
Contains settings, network definition for Chainer.
Creates the simulator, replay memory, DQN learner, and passes these to the agent framework for training.
'''

import numpy as np

import chainer
import chainer.functions as F

from memories.memory_hdf5 import ReplayMemoryHDF5

from learners import Learner
from agents import DQNAgent

from simulators.pomdp import POMDPSimulator
from simulators.pomdp import TigerPOMDP

print('Setting training parameters...')
# Set training settings
settings = {
    # agent settings
    'batch_size' : 32,
    'print_every' : 5000,
    'save_dir' : 'results/nets_tiger',
    'iterations' : 200000,
    'eval_iterations' : 5000,
    'eval_every' : 5000,
    'save_every' : 25000,
    'initial_exploration' : 10000,
    'epsilon_decay' : 0.0001, # subtract from epsilon every step
    'eval_epsilon' : 0.05, # epsilon used in evaluation, 0 means no random actions

    # simulator settings
    'epsilon' : 1.0,  # Initial exploratoin rate
    'viz' : False,
    'model_dims' : (2,1), # size to which the image shall be cropped

    # replay memory settings
    'memory_size' : 100000,  # size of replay memory
    'n_frames' : 1,  # number of frames

    # learner settings
    'learning_rate' : 0.00025, 
    'decay_rate' : 0.95, # decay rate for RMSprop, otherwise not used
    'discount' : 0.95, # discount rate for RL
    'clip_err' : False, # value to clip loss gradients to
    'clip_reward' : False, # value to clip reward values to
    'target_net_update' : 1000, # update the update-generating target net every fixed number of iterations
    'double_DQN' : False, # use Double DQN (based on Deep Mind paper)
    'optim_name' : 'RMSprop', # currently supports "RMSprop", "ADADELTA" and "SGD"'
    'gpu' : False,

    # general
    'seed' : 1234
    }

print(settings)

np.random.seed(settings["seed"])


print('Initializing replay memory...')
memory = ReplayMemoryHDF5(settings)

print('Setting up simulator...')
pomdp = TigerPOMDP()
simulator = POMDPSimulator(pomdp, settings)

print('Setting up networks...')

net = chainer.FunctionSet(
    l1=F.Linear(2*settings["n_frames"], 200, wscale=np.sqrt(2)),
    l2=F.Linear(200, 100, wscale=np.sqrt(2)),
    l3=F.Linear(100, 100, wscale=np.sqrt(2)),
    l4=F.Linear(100, 50, wscale=np.sqrt(2)),
    l5=F.Linear(50, simulator.n_actions, wscale = np.sqrt(2))
    )

def forward(net, s):
    h1 = F.relu(net.l1(s))
    h2 = F.relu(net.l2(h1))
    h3 = F.relu(net.l3(h2))    
    h4 = F.relu(net.l4(h3))
    output = net.l5(h4)
    return output

print('Initializing the learner...')
learner = Learner(net, forward, settings)

print('Initializing the agent framework...')
agent = DQNAgent(settings)

print('Training...')
agent.train(learner, memory, simulator)

print('Loading the net...')
learner = agent.load('./nets_tiger/learner_final.p')


np.random.seed(359)

print('Evaluating DQN agent...')
print('(reward, MSE loss, mean Q-value, episodes - NA, time)')
print(agent.evaluate(learner, simulator, 50000))

print('Evaluating optimal policy...')
print('(reward, NA, NA, episodes - NA, time)')
print(agent.evaluate(learner, simulator, 50000, custom_policy=pomdp.optimal_policy()))

