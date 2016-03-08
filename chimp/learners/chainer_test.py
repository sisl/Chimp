"""
This is a place holder for real unit testing.
Right now we just overfit a simple control problem:
    - the agent tries to get to the top right corner (1,1) of a 2D map
    - action 0 takes it towards (0,0), action 1 takes it toward (1,1)
    - action 1 is optimal for all states
"""

from chimp.learners.chainer_backend import ChainerBackend
from chimp.learners.dqn_learner import DQNLearner
from chimp.learners.dqn_learner import DQNPolicy

import numpy as np

import chainer
import chainer.functions as F
import chainer.links as L
from chainer import Chain

settings = {

    # agent settings
    'batch_size' : 32,
    'print_every' : 500,
    'save_dir' : 'results/nets_rocksample_belief_rmsprop',
    'iterations' : 100000,
    'eval_iterations' : 100,
    'eval_every' : 1000,
    'save_every' : 500,
    'initial_exploration' : 500,
    'epsilon_decay' : 0.00001, # subtract from epsilon every step
    'eval_epsilon' : 0, # epsilon used in evaluation, 0 means no random actions
    'epsilon' : 1.0,  # Initial exploratoin rate
    'learn_freq' : 1,
    'history_sizes' : (1, 0, 0), # sizes of histories to use as nn inputs (o, a, r)
    'model_dims' : (1,2),

    # simulator settings
    'viz' : False,

    # replay memory settings
    'memory_size' : 1000,  # size of replay memory
    'n_frames' : 1,  # number of frames

    # learner settings
    'learning_rate' : 0.00025, 
    'decay_rate' : 0.99, # decay rate for RMSprop, otherwise not used
    'discount' : 0.95, # discount rate for RL
    'clip_err' : False, # value to clip loss gradients to
    'clip_reward' : 1, # value to clip reward values to
    'target_net_update' : 1000, # update the update-generating target net every fixed number of iterations
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

n_actions = 2
o_dims = settings['model_dims']
n_samples = settings['batch_size']

class TestNet(Chain):

    def __init__(self):
        super(TestNet, self).__init__(
            #l1=F.Bilinear(settings["history_sizes"][0], settings["history_sizes"][1], 20),
            l1=F.Linear(o_dims[1], 20, bias=0.0),
            l2=F.Linear(20, 10, bias=0.0),
            bn1=L.BatchNormalization(10),
            lout=F.Linear(10, n_actions)
        )
        self.train = True
        # initialize avg_var to prevent divide by zero
        self.bn1.avg_var.fill(0.1),

    def __call__(self, ohist, ahist):
        h = F.relu(self.l1(ohist))
        h = F.relu(self.l2(h))
        h = self.bn1(h, test=not self.train)
        output = self.lout(h)
        return output

def make_batch(n_samples, o_dims, n_actions):
    obs = np.zeros((n_samples,)+o_dims, dtype=np.float32)
    obsp = np.zeros((n_samples,)+o_dims, dtype=np.float32)
    a = np.zeros(n_samples, dtype=np.int32)
    r = np.zeros(n_samples, dtype=np.float32)
    term = np.zeros(n_samples, dtype=np.bool)
    for i in xrange(n_samples):
        obs[i] = np.random.uniform(0.0, 1.0, o_dims)
        a[i] = np.random.randint(n_actions)
        obsp[i] = (obs[i] + 0.25) if a[i] == 1 else (obs[i] - 0.25)
        obsp[i] = np.clip(obsp[i], 0.0, 1.0)
        r[i] = np.sum(obs[i])
    return obs, a, r, obsp, term


net = TestNet()
custom_learner = ChainerBackend(settings)
custom_learner.set_net(net)

learner = DQNLearner(settings, custom_learner)

policy = DQNPolicy(learner)

obst, a, r, obsp, term = make_batch(10, o_dims, n_actions)

for i in xrange(10):
    ohist = (obst[i], None)
    a = policy.action(ohist)
    print "Test: ", i, " ", obst[i], " ", a, " ", learner.forward((obst[i], None))

print "TRAINING"
for i in xrange(3000):
    obs, a, r, obsp, term = make_batch(n_samples, o_dims, n_actions)
    ohist = (obs, None)
    ophist = (obsp, None)
    #loss, q_all = custom_learner.forward_loss(ohist, a, r, ophist, term)
    loss, q_all = learner.update(ohist, a, r, ophist, term)
    if i % 500 == 0:
        print loss
    

for i in xrange(10):
    ohist = (obst[i], None)
    a = policy.action(ohist)
    print "Test: ", i, " ", obst[i], " ", a, " ", learner.forward((obst[i], None))


