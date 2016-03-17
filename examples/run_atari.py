# be sure to have run ' python setup.py ' from chimp directory


# # Training DeepMind's Atari DQN with Chimp

# First, we load all the Chimp modules.

from chimp.memories import ReplayMemoryHDF5

from chimp.learners.dqn_learner import DQNLearner
from chimp.learners.chainer_backend import ChainerBackend

from chimp.simulators.atari import AtariSimulator

from chimp.agents import DQNAgent


# Then we load Python packages.

import matplotlib.pyplot as plt

import numpy as np
import chainer
import chainer.functions as F
import chainer.links as L
from chainer import Chain
import os

import pandas as ps


# Finally, we set training parameters in a params dictionary that will be passed to the modules.

# Define training settings

settings = {

    # agent settings
    'batch_size' : 32,
    'print_every' : 10000,
    'save_dir' : './results_atari',
    'iterations' : 5000000,
    'eval_iterations' : 5000,
    'eval_every' : 50000,
    'save_every' : 50000,
    'initial_exploration' : 50000,
    'epsilon_decay' : 0.000005, # subtract from epsilon every step
    'eval_epsilon' : 0.05, # epsilon used in evaluation, 0 means no random actions
    'epsilon' : 1.0,  # Initial exploratoin rate
    'learn_freq' : 4,
    'history_sizes' : (4, 0, 0), # sizes of histories to use as nn inputs (o, a, r)
    'model_dims' : (84,84),
    
    # Atari settings
    'rom' : "Breakout.bin",
    'rom_dir' :  './roms',
    'pad' : 15, # padding parameter - for image cropping - only along the length of the image, to obtain a square
    'action_history' : True,

    # simulator settings
    'viz' : True,
    'viz_cropped' : False,

    # replay memory settings
    'memory_size' : 500000,  # size of replay memory
    'frame_skip' : 4,  # number of frames to skip

    # learner settings
    'learning_rate' : 0.00025, 
    'decay_rate' : 0.95, # decay rate for RMSprop, otherwise not used
    'discount' : 0.99, # discount rate for RL
    'clip_err' : False, # value to clip loss gradients to
    'clip_reward' : 1, # value to clip reward values to
    'target_net_update' : 10000, # update the update-generating target net every fixed number of iterations
    'optim_name' : 'RMSprop', # currently supports "RMSprop", "ADADELTA", "ADAM" and "SGD"'
    'gpu' : True, # NO GPU FOR THIS EXAMPLE 
    'reward_rescale': False,

    # general
    'seed_general' : 1723,
    'seed_simulator' : 5632,
    'seed_agent' : 9826,
    'seed_memory' : 7563

    }


# set random seed
np.random.seed(settings["seed_general"])


# initialize the simulator

simulator = AtariSimulator(settings)

# Define the network
class Convolution(Chain):

    def __init__(self):
        super(Convolution, self).__init__(
            l1=F.Convolution2D(settings['history_sizes'][0], 32, ksize=8, stride=4, nobias=False, wscale=np.sqrt(2)),
            l2=F.Convolution2D(32, 64, ksize=4, stride=2, nobias=False, wscale=np.sqrt(2)),
            l3=F.Convolution2D(64, 64, ksize=3, stride=1, nobias=False, wscale=np.sqrt(2)),
            l4=F.Linear(3136, 512, wscale = np.sqrt(2)),
            l5=F.Linear(512, simulator.n_actions, wscale = np.sqrt(2)),
        )

    def __call__(self, ohist, ahist):
        if len(ohist.data.shape) < 4:
            ohist = F.reshape(ohist,(1,4,84,84))
        h1 = F.relu(self.l1(ohist/255.0))
        h2 = F.relu(self.l2(h1))
        h3 = F.relu(self.l3(h2))
        h4 = F.relu(self.l4(h3))
        output = self.l5(h4)
        return output

net = Convolution()


# initialize the learner + chainer backend, replay memory, and agent modules

backend = ChainerBackend(settings)
backend.set_net(net)
learner = DQNLearner(settings, backend)

memory = ReplayMemoryHDF5(settings)

agent = DQNAgent(learner, memory, simulator, settings)

# launch training

agent.train()


# Visualizing results

train_stats = ps.read_csv('%s/training_history.csv' % settings['save_dir'],delimiter=' ',header=None)
train_stats.columns = ['Iteration','MSE Loss','Average Q-Value']

eval_stats = ps.read_csv('%s/evaluation_history.csv' % settings['save_dir'],delimiter=' ',header=None)
eval_stats.columns = ['Iteration','Total Reward','Reward per Episode']


plt.plot(eval_stats['Iteration'], eval_stats['Reward per Episode'])
plt.xlabel("Iteration")
plt.ylabel("Avg. Reward per Episode")
plt.grid(True)
plt.savefig(settings['save_dir'] + '_' + "evaluation_reward.svg", bbox_inches='tight')
#plt.show()
plt.close()


plt.plot(train_stats['Iteration'], train_stats['Average Q-Value'])
plt.xlabel("Iteration")
plt.ylabel("Avg. Q-Values")
plt.grid(True)
plt.savefig(settings['save_dir'] + '_' + "training_q_values.svg", bbox_inches='tight')
#plt.show()
plt.close()


# Evaluating the best policy

# load the network that collected the highest reward per game episode

best_iteration_index = np.argmax(eval_stats['Reward per Episode'])
best_iteration = str(int(eval_stats['Iteration'][best_iteration_index]))

agent.learner.load_net(settings['save_dir']+'/net_' + best_iteration + '.p')


# evaluate policy performance

r_tot, r_per_episode, runtime = agent.simulate(10000, epsilon=0.05, viz=True)

r_per_episode


