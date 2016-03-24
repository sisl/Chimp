"""
This is a place holder for real unit testing.
Right now we just overfit a simple control problem:
    - the agent tries to get to the top right corner (1,1) of a 2D map
    - action 0 takes it towards (0,0), action 1 takes it toward (1,1)
    - action 1 is optimal for all states
"""

# Memory
from chimp.memories import ReplayMemoryHDF5

# Learner (Brain)
from chimp.learners.dqn_learner import DQNLearner
from chimp.learners.chainer_backend import ChainerBackend

# Agent Framework
from chimp.agents import DQNAgent

# Simulator
from chimp.simulators.mdp.mdp_simulator import MDPSimulator
from chimp.simulators.mdp.cart_pole import CartPole 

# Rollout Policy
from chimp.utils.policies import RandomPolicy

import numpy as np
import pickle
import pylab as p

import chainer
import chainer.functions as F
import chainer.links as L
from chainer import Chain

settings = {

    # agent settings
    'batch_size' : 32,
    'print_every' : 1000,
    'save_dir' : 'results',
    'iterations' : 500000,
    'eval_iterations' : 200,
    'eval_every' : 1000,
    'save_every' : 20000,
    'initial_exploration' : 10000,
    'epsilon_decay' : 0.000005, # subtract from epsilon every step
    'eval_epsilon' : 0, # epsilon used in evaluation, 0 means no random actions
    'epsilon' : 1.0,  # Initial exploratoin rate
    'learn_freq' : 1,
    'history_sizes' : (1, 0, 0), # sizes of histories to use as nn inputs (o, a, r)
    'model_dims' : (1,4),

    # simulator settings
    'viz' : False,

    # replay memory settings
    'memory_size' : 10000,  # size of replay memory
    'n_frames' : 1,  # number of frames

    # learner settings
    'learning_rate' : 0.00001,
    'decay_rate' : 0.95, # decay rate for RMSprop, otherwise not used
    'discount' : 0.99, # discount rate for RL
    'clip_err' : False, # value to clip loss gradients to
    'clip_reward' : False, # value to clip reward values to
    'target_net_update' : 2000, # update the update-generating target net every fixed number of iterations
    'double_DQN' : False, # use Double DQN (based on Deep Mind paper)
    'optim_name' : 'ADAM', # currently supports "RMSprop", "ADADELTA", "ADAM" and "SGD"'
    'gpu' : False,
    'reward_rescale': False,

    # general
    'seed_general' : 1723,
    'seed_simulator' : 5632,
    'seed_agent' : 9826,
    'seed_memory' : 7563

    }

mdp = CartPole()
simulator = MDPSimulator(mdp)

class CartNet(Chain):

    def __init__(self):
        super(CartNet, self).__init__(
            l1=F.Linear(4, 20, bias=0.0),
            l2=F.Linear(20, 10, bias=0.0),
            bn1=L.BatchNormalization(10),
            l3=F.Linear(10, 10),
            l4=F.Linear(10, 10),
            bn2=L.BatchNormalization(10),
            lout=F.Linear(10, simulator.n_actions)
        )
        self.train = True
        # initialize avg_var to prevent divide by zero
        self.bn1.avg_var.fill(0.1),
        self.bn2.avg_var.fill(0.1),

    def __call__(self, ohist, ahist):
        h = F.relu(self.l1(ohist))
        h = F.relu(self.l2(h))
        h = self.bn1(h, test=not self.train)
        h = F.relu(self.l3(h))
        h = F.relu(self.l4(h))
        h = self.bn2(h, test=not self.train)
        output = self.lout(h)
        return output


def pole_sim(nsteps, simulator, policy, verbose=False):
    mdp = simulator.model

    # re-initialize the model
    simulator.reset_episode()

    rtot = 0.0
    xpos = np.zeros(nsteps)
    thetas = np.zeros(nsteps)
    # run the simulation
    input_state = np.zeros((1,4), dtype=np.float32)
    for i in xrange(nsteps):
        state = simulator.get_screenshot()
        input_state[0] = state
        #a = policy.action((input_state,None))
        a = policy.action(state)
        simulator.act(a)
        r = simulator.reward()
        rtot += r
        xpos[i], thetas[i] = state[0], state[2]
        print state, r
        if simulator.episode_over():
            break
    return rtot, xpos, thetas


class PoleCartHeuristic():

    def __inti__(self):
        self.a = 0

    def action(self, state):
        if state[2] > 0:
            return 1
        else:
            return 0


net = CartNet()

# Initialize Learner with a Chainer backend
backend = ChainerBackend(settings)
backend.set_net(net)
learner = DQNLearner(settings, backend)

# Initialize memory
memory = ReplayMemoryHDF5(settings)

# Initialize Agent Framework
agent = DQNAgent(learner, memory, simulator, settings)

# Start training
agent.train(verbose=True)

#policy = RandomPolicy(simulator.n_actions)
#policy = PoleCartHeuristic()

#r, xs, ts = pole_sim(100, simulator, policy, verbose=True)

#p.plot(xs); p.plot(10.0*ts)
#p.show()
