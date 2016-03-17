
# LOAD CHIMP COMPONENTS

# Memory
from chimp.memories import ReplayMemoryHDF5

# Learner (Brain)
from chimp.learners.dqn_learner import DQNLearner
from chimp.learners.chainer_backend import ChainerBackend

# Simulator
from chimp.simulators.atari import AtariSimulator

# Agent Framework
from chimp.agents import DQNAgent


# Request necessary packages

import numpy as np
import random

import chainer
import chainer.functions as F
import chainer.links as L
from chainer import Chain


# Set the settings

settings = {

    # agent settings
    'batch_size' : 32,
    'print_every' : 1000,
    'save_dir' : './results_atari',
    'iterations' : 10000,
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
    'memory_size' : 1000000,  # size of replay memory
    'frame_skip' : 4,  # number of frames to skip

    # learner settings
    'learning_rate' : 0.00025, 
    'decay_rate' : 0.95, # decay rate for RMSprop, otherwise not used
    'discount' : 0.99, # discount rate for RL
    'clip_err' : False, # value to clip loss gradients to
    'clip_reward' : 1, # value to clip reward values to
    'target_net_update' : 10000, # update the update-generating target net every fixed number of iterations
    'optim_name' : 'RMSprop', # currently supports "RMSprop", "ADADELTA", "ADAM" and "SGD"'
    'gpu' : True,
    'reward_rescale': False,

    # general
    'seed_general' : 1723,
    'seed_simulator' : 5632,
    'seed_agent' : 9826,
    'seed_memory' : 7563

    }


# Initialize random seed 

np.random.seed(settings["seed_general"])
random.seed(settings["seed_general"])

# Initialize Atari simulator
simulator = AtariSimulator(settings)

# Initialize a deep net structure

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


# Initialize Learner with a Chainer backend

backend = ChainerBackend(settings)
backend.set_net(net)

learner = DQNLearner(settings, backend)


# Initialize memory

memory = ReplayMemoryHDF5(settings)


# Initialize Agent Framework

agent = DQNAgent(learner, memory, simulator, settings)


# Start training

agent.train()


# load net

# agent.learner.load_net(settings['save_dir']+'/net_750000.p')

# '''Plot training history'''

# plt.plot(range(len(learner.val_rewards)), learner.val_rewards)
# plt.xlabel("Evaluation episode (1000 transitions long, every 5000 training iter.)")
# plt.ylabel("Accumulated reward per episode")
# plt.xlim(0, 100)
# plt.ylim(-200, 200)
# plt.grid(True)
# plt.savefig(settings['save_dir'] + '_' + "evaluation_reward.svg", bbox_inches='tight')
# plt.close()

# plt.plot(range(len(learner.val_qval_avgs)), learner.val_qval_avgs)
# plt.xlabel("Evaluation episode  (1000 transitions long, every 5000 training iter.)")
# plt.ylabel("Average Q-value")
# plt.xlim(0, 100)
# plt.ylim(0, 20)
# plt.grid(True)
# plt.savefig(settings['save_dir'] + '_' + "evaluation_q_value.svg", bbox_inches='tight')
# plt.close()

# '''Plot value surface for beliefs'''
# ind_max = learner.val_rewards[0::5].index(max(learner.val_rewards[0::5]))
# ind_net = settings['save_every'] + ind_max * settings['save_every']
# learner.load_net(settings['save_dir']+'/net_%d.p' % int(ind_net))

# print(ind_net)
# print(ind_max*5)
# print(learner.val_rewards)
# print(learner.val_rewards[ind_max*5])

# '''Evaluate learned policy against the optimal heuristic policy'''
# np.random.seed(settings["seed_general"])

# print('Evaluating DQN agent...')
# print('(reward, MSE loss, mean Q-value, episodes - NA, time)')
# reward, MSE_loss, mean_Q_value, episodes, time, paths, actions, rewards = agent.evaluate(learner, simulator, 50000)
# print(reward, MSE_loss, mean_Q_value, episodes, time)
