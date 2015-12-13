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

from simulators.atari import AtariSimulator

import matplotlib.pyplot as plt

print('Setting training parameters...')
# Set training settings
settings = {
    # agent settings
    'batch_size' : 32, # mini-batch size
    'print_every' : 5000, # print out update every 5000 iterations
    'save_dir' : './results/nets_atari', # directory where we save the net
    'iterations' : 10000000,
    'eval_iterations' : 5000,
    'eval_every' : 10000,
    'save_every' : 50000,
    'initial_exploration' : 10000,
    'epsilon_decay' : 1.0/10**6, # subtract 1.0/10**6 every step
    'eval_epsilon' : 0.05, # epsilon used in evaluation, 0 means no random actions
    'learn_freq' : 4,

    # Atari simulator settings
    'epsilon' : 1.0,  # Initial exploratoin rate
    'frame_skip' : 4,
    'viz' : True,
    'viz_cropped' : False, # visualize only what an agent sees? vs. the whole screen
    'rom' : "Breakout.bin",
    'rom_dir' : "./simulators/atari/roms",
    'model_dims' : (84,84), # size to which the image shall be cropped
    'pad' : 15, # padding parameter - for image cropping - only along the length of the image, to obtain a square
    'action_history' : True,

    # replay memory settings
    'memory_size' : 100000,  # size of replay memory
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

print('Firing up Atari...')
simulator = AtariSimulator(settings)

print('Initializing replay memory...')
memory = ReplayMemoryHDF5(settings)

print('Setting up networks...')

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

print('Loading the net...')
learner = agent.load(settings['save_dir']+'/learner_final.p')

'''Plot training history'''

plt.plot(range(len(learner.val_rewards)), learner.val_rewards)
plt.xlabel("Evaluation episode (1000 transitions long, every 5000 training iter.)")
plt.ylabel("Accumulated reward per episode")
plt.xlim(0, 100)
plt.ylim(-200, 200)
plt.grid(True)
plt.savefig(settings['save_dir'] + '_' + "evaluation_reward.svg", bbox_inches='tight')
plt.close()

plt.plot(range(len(learner.val_qval_avgs)), learner.val_qval_avgs)
plt.xlabel("Evaluation episode  (1000 transitions long, every 5000 training iter.)")
plt.ylabel("Average Q-value")
plt.xlim(0, 100)
plt.ylim(0, 20)
plt.grid(True)
plt.savefig(settings['save_dir'] + '_' + "evaluation_q_value.svg", bbox_inches='tight')
plt.close()

'''Plot value surface for beliefs'''
ind_max = learner.val_rewards[0::5].index(max(learner.val_rewards[0::5]))
ind_net = settings['save_every'] + ind_max * settings['save_every']
agent.load_net(learner,settings['save_dir']+'/net_%d.p' % int(ind_net))

print(ind_net)
print(ind_max*5)
print(learner.val_rewards)
print(learner.val_rewards[ind_max*5])

'''Evaluate learned policy against the optimal heuristic policy'''
np.random.seed(settings["seed_general"])

print('Evaluating DQN agent...')
print('(reward, MSE loss, mean Q-value, episodes - NA, time)')
reward, MSE_loss, mean_Q_value, episodes, time, paths, actions, rewards = agent.evaluate(learner, simulator, 50000)
print(reward, MSE_loss, mean_Q_value, episodes, time)
