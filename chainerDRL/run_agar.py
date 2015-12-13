'''
File to initialize training.
Contains settings, network definition for Chainer.
Creates the simulator, replay memory, DQN learner, and passes these to the agent framework for training.
'''

import numpy as np

import chainer
import chainer.functions as F
import chainer.links as L
from chainer import cuda, Function, gradient_check, Variable, optimizers, serializers, utils
from chainer import Link, Chain, ChainList

from memories import ReplayMemoryHDF5

from learners import Learner
from agents import DQNAgent

from simulators.agar import AgarIODriver

print('Setting training parameters...')
# Set training settings
settings = {
    # agent settings
    'batch_size' : 32, # mini-batch size
    'print_every' : 100, # print out update every 5000 iterations
    'save_dir' : './results/nets_agar', # directory where we save the net
    'iterations' : 10000000,
    'eval_iterations' : 5000,
    'eval_every' : 10000,
    'save_every' : 50000,
    'initial_exploration' : 1000,
    'epsilon_decay' : 1.0/10**5, # subtract 1.0/10**6 every step
    'eval_epsilon' : 0.01, # epsilon used in evaluation, 0 means no random actions
    'learn_freq' : 1,

    # Atari simulator settings
    'epsilon' : 1.0,  # Initial exploratoin rate
    'viz' : True,
    'viz_cropped' : True, # visualize only what an agent sees? vs. the whole screen
    'model_dims' : (84,84), # size to which the image shall be cropped
    'pad' : 15, # padding parameter - for image cropping - only along the length of the image, to obtain a square

    # replay memory settings
    'memory_size' : 10000,  # size of replay memory
    'n_frames' : 4,  # number of frames

    # learner settings
    'learning_rate' : 0.00025, 
    'decay_rate' : 0.95, # decay rate for RMSprop, otherwise not used
    'discount' : 0.99, # discount rate for RL
    'clip_err' : False, # value to clip loss gradients to
    'clip_reward' : 1, # value to clip reward values to
    'target_net_update' : 10000, # update the update-generating target net every fixed number of iterations
    'double_DQN' : False, # use Double DQN (based on Deep Mind paper)
    'optim_name' : 'RMSprop', # currently supports "RMSprop", "ADADELTA" and "SGD"'
    'gpu' : True,
    'reward_rescale': False,

    # general
    'seed_general' : 1723,
    'seed_simulator' : 5632,
    'seed_agent' : 9826,
    'seed_memory' : 7563
    }

print(settings)

np.random.seed(settings["seed_general"])

print('Initializing replay memory...')
memory = ReplayMemoryHDF5(settings)

print('Firing up Atari...')
simulator = AgarIODriver(settings)

print('Setting up networks...')
# Set random seed + parameters that define the network (you cannot pass the network a random number generator)

class Convolution(Chain):

    def __init__(self):
        super(Convolution, self).__init__(
            l1=F.Convolution2D(settings['n_frames'], 32, ksize=8, stride=4, nobias=False, wscale=np.sqrt(2)),
            l2=F.Convolution2D(32, 64, ksize=4, stride=2, nobias=False, wscale=np.sqrt(2)),
            l3=F.Convolution2D(64, 64, ksize=3, stride=1, nobias=False, wscale=np.sqrt(2)),
            l4=F.Linear(3136, 512, wscale = np.sqrt(2)),
            l5=F.Linear(512, simulator.n_actions, wscale = np.sqrt(2)),
        )

    def __call__(self, s, action_history):
        h1 = F.relu(self.l1(s/255.0))
        h2 = F.relu(self.l2(h1))
        h3 = F.relu(self.l3(h2))
        h4 = F.relu(self.l4(h3))
        output = self.l5(h4)
        return output

net = Convolution()

print('Initializing the learner...')
learner = Learner(net, settings)

print('Initializing the agent framework...')
agent = DQNAgent(settings)

print('Training...')
agent.train(learner, memory, simulator)

print('Loading the net...')
learner = agent.load('./results/nets_agar/learner_final.p')

print('Evaluating DQN agent...')
print('(reward, MSE loss, mean Q-value, episodes)')
agent.evaluate(learner, simulator, 50000)
