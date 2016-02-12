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

from simulators.pomdp import POMDPSimulator
from simulators.pomdp import TigerPOMDP

import matplotlib.pyplot as plt

print('Setting training parameters...')
# Set training settings
settings = {
    # agent settings
    'batch_size' : 32,
    'print_every' : 5000,
    'save_dir' : 'results/nets_tiger_belief',
    'iterations' : 200000,
    'eval_iterations' : 5000,
    'eval_every' : 5000,
    'save_every' : 5000,
    'initial_exploration' : 10000,
    'epsilon_decay' : 0.0001, # subtract from epsilon every step
    'eval_epsilon' : 0, # epsilon used in evaluation, 0 means no random actions
    'epsilon' : 1.0,  # Initial exploratoin rate
    'model_dims': (2,1),
    'learn_freq' : 1,

    # simulator settings
    'viz' : False,

    # replay memory settings
    'memory_size' : 100000,  # size of replay memory
    'n_frames' : 1,  # number of frames

    # learner settings
    'learning_rate' : 0.001, 
    'decay_rate' : 0.99, # decay rate for RMSprop, otherwise not used
    'discount' : 0.95, # discount rate for RL
    'clip_err' : False, # value to clip loss gradients to
    'clip_reward' : False, # value to clip reward values to
    'target_net_update' : 1000, # update the update-generating target net every fixed number of iterations
    'double_DQN' : False, # use Double DQN (based on Deep Mind paper)
    'optim_name' : 'RMSprop', # currently supports "RMSprop", "ADADELTA" and "SGD"'
    'gpu' : False,
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

print('Setting up simulator...')
pomdp = TigerPOMDP( seed=settings['seed_simulator'] )
simulator = POMDPSimulator(pomdp)

print('Setting up networks...')

class Linear(Chain):

    def __init__(self):
        super(Linear, self).__init__(
            l1=F.Linear(simulator.model_dims[0] * settings["n_frames"], 200, wscale=np.sqrt(2)),
            l2=F.Linear(200, 100, wscale=np.sqrt(2)),
            l3=F.Linear(100, 100, wscale=np.sqrt(2)),
            l4=F.Linear(100, 50, wscale=np.sqrt(2)),
            l5=F.Linear(50, simulator.n_actions, wscale = np.sqrt(2))
        )

    def __call__(self, s, action_history):
        h1 = F.relu(net.l1(s))
        h2 = F.relu(net.l2(h1))
        h3 = F.relu(net.l3(h2))    
        h4 = F.relu(net.l4(h3))
        output = net.l5(h4)
        return output

net = Linear()

print('Initializing the learner...')
learner = Learner(net, settings)

print('Initializing the agent framework...')
agent = DQNAgent(settings)

print('Loading the net...')
learner = agent.load(settings['save_dir']+'/learner_final.p')

'''Plot training history'''

plt.plot(range(len(learner.val_rewards)), learner.val_rewards)
plt.xlabel("Evaluation episode (5000 transitions long, every 5000 training iter.)")
plt.ylabel("Accumulated reward per episode")
plt.xlim(0, 100)
plt.ylim(-10000, 10000)
plt.grid(True)
plt.savefig(settings['save_dir'] + '_' + "evaluation_reward.svg", bbox_inches='tight')
plt.close()

plt.plot(range(len(learner.val_qval_avgs)), learner.val_qval_avgs)
plt.xlabel("Evaluation episode  (5000 transitions long, every 5000 training iter.)")
plt.ylabel("Average Q-value")
plt.xlim(0, 100)
plt.ylim(0, 40)
plt.grid(True)
plt.savefig(settings['save_dir'] + '_' + "evaluation_q_value.svg", bbox_inches='tight')
plt.close()



csv_rq = np.array([range(len(learner.val_rewards)),learner.val_rewards, learner.val_qval_avgs]).T


'''Plot value surface for beliefs'''
beliefs = np.array([[0,1],[0.025,0.975],[0.05,0.95], [0.15,0.85], [0.25,0.75], [0.35,0.65], [0.5,0.5],
    [0.65,0.35], [0.75,0.25], [0.85,0.15], [0.95,0.05], [0.975, 0.025], [1,0] ],dtype=np.float32)
beliefs = chainer.Variable(beliefs)

l = len(learner.val_rewards)

ind_min = learner.val_rewards[-10:].index(min(learner.val_rewards[-10:]))
ind_net = settings['initial_exploration'] + (l - 10 + ind_min) * settings['eval_every']
agent.load_net(learner,settings['save_dir']+'/net_%d.p' % int(ind_net))

all_a = learner.forward(learner.net, beliefs, None)

plt.plot(beliefs.data[:,0], all_a.data[:,0],color='b',linestyle='-',label='Open the left door') 
plt.plot(beliefs.data[:,0], all_a.data[:,1],color='g',linestyle='--',label='Open the right door')
plt.plot(beliefs.data[:,0], all_a.data[:,2],color='r',linestyle=':',label='Listen')
plt.xlabel("Belief that tiger is behind the left door")
plt.ylabel("Q-value")
plt.xlim(0, 1)
plt.ylim(0, 40)
plt.grid(True)
plt.legend(loc=4);
plt.savefig(settings['save_dir'] + '_' + str(ind_net) + '_' + "value_surface_bad.svg", bbox_inches='tight')
plt.close()

csv_val_bad = np.array([beliefs.data[:,0], all_a.data[:,0], all_a.data[:,1], all_a.data[:,2]]).T

ind_max = learner.val_rewards.index(max(learner.val_rewards))
ind_net = settings['initial_exploration'] + ind_max * settings['eval_every']
agent.load_net(learner,settings['save_dir']+'/net_%d.p' % int(ind_net))

all_a = learner.forward(learner.net, beliefs, None)

plt.plot(beliefs.data[:,0], all_a.data[:,0],color='b',linestyle='-',label='Open the left door') 
plt.plot(beliefs.data[:,0], all_a.data[:,1],color='g',linestyle='--',label='Open the right door')
plt.plot(beliefs.data[:,0], all_a.data[:,2],color='r',linestyle=':',label='Listen')
plt.xlabel("Belief that tiger is behind the left door")
plt.ylabel("Q-value")
plt.xlim(0, 1)
plt.ylim(0, 40)
plt.grid(True)
plt.legend(loc=4);
plt.savefig(settings['save_dir'] + '_' + str(ind_net) + '_' + "value_surface_best.svg", bbox_inches='tight')
plt.close()

csv_val_good = np.array([beliefs.data[:,0], all_a.data[:,0], all_a.data[:,1], all_a.data[:,2]]).T

np.savetxt('./training.csv', csv_rq, delimiter=',')
np.savetxt('./val_good.csv', csv_val_good, delimiter=',')
np.savetxt('./val_bad.csv', csv_val_bad, delimiter=',')

'''Evaluate learned policy against the optimal heuristic policy'''

np.random.seed(settings["seed_general"])

print('Evaluating DQN agent...')
print('(reward, MSE loss, mean Q-value, episodes - NA, time)')
reward, MSE_loss, mean_Q_value, episodes, time, paths, actions, rewards = agent.evaluate(learner, simulator, 50000)
print(reward, MSE_loss, mean_Q_value, episodes, time)
# 52828 / 50000 = 1.06

print('Evaluating optimal policy...')
print('(reward, NA, NA, episodes - NA, time)')
reward, MSE_loss, mean_Q_value, episodes, time, paths, actions, rewards = agent.evaluate(learner, simulator, 50000, custom_policy=pomdp.optimal_policy())
print(reward, MSE_loss, mean_Q_value, episodes, time)
# 49682 / 50000 = 0.99
