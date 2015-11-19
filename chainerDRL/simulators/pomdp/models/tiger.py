import numpy as np
from copy import deepcopy
from tools.belief import DiscreteBelief

#################################################################
# Implements the Tiger POMDP problem
#################################################################

class TigerPOMDP():

    # constructor
    def __init__(self, 
                 seed=999, # random seed
                 rlisten=-1.0, rtiger=-100.0, rescape=10.0, # reward values
                 pcorrect=0.85, # correct observation prob
                 discount=0.95): # discount

        self.random_state = np.random.RandomState(seed)
        self.rlisten = rlisten
        self.rtiger = rtiger
        self.rescape = rescape
        self.pcorrect = pcorrect
        self.discount = discount

        # transition arrs
        self.tstates = [0, 1] # left, right

        # actions
        self.tactions = [0, 1, 2] # open left, open right, listen

        # observations arrs
        self.tobs = [0, 1] # observed on the left, observed on the right

        # belief and observation shape
        self.belief_shape = (2,1)
        self.observation_shape = (1,1)

    ################################################################# 
    # Setters
    ################################################################# 
    def set_discount(self, d):
        self.discount = d

    def set_rewards(self, rl, rt, re):
        self.rlisten = rl
        self.rtiger = rt
        self.rescape = re

    def set_listen_prob(self, pc):
        self.pcorrect = pc

    ################################################################# 
    # S, A, O Spaces
    ################################################################# 
    def states(self):
        return self.tstates

    def actions(self):
        return self.tactions

    def observations(self):
        return self.tobs

    ################################################################# 
    # Reward Function
    ################################################################# 
    def reward(self, s, a):
        r = 0.0
        rt = self.rtiger
        re = self.rescape
        if a == 2:
            r += self.rlisten
        elif a == 1:
            r = (r + rt) if s == 1 else (r + re) 
        else:
            r = (r + rt) if s == 0 else (r + re) 
        return r

    ################################################################# 
    # Distribution Functions
    ################################################################# 
    # returns the transtion distriubtion of s' from the (s,a) pair
    def transition(self, s, a, dist):
        if a == 0 or a == 1:
            dist[0] = 0.5
            dist[1] = 0.5
        elif s == 0:
            dist[0] = 1.0 
            dist[1] = 0.0
        else:
            dist[0] = 0.0
            dist[1] = 1.0
        return dist

    # sample the transtion distribution 
    def sample_state(self, d):
        sidx = self.categorical(d)
        return self.tstates[sidx]

    # returns the observation dsitribution of o from the (s,a) pair
    def observation(self, s, a, dist):
        p = self.pcorrect
        if a == 2:
            if s == 0:
                dist[0] = p
                dist[1] = 1.0 - p
            else:
                dist[0] = 1.0 - p
                dist[1] = p
        else:
            dist[0] = 0.5
            dist[1] = 0.5
        return dist

    # sample the observation distirbution
    def sample_observation(self, d):
        oidx = self.categorical(d)
        return self.tobs[oidx]

    # pdf should be in a distributions module
    def transition_pdf(self, d, dval):
        assert dval < 2, "Attempting to retrive pdf value larger than state size"
        return d[dval]

    def observation_pdf(self, d, dval):
        assert dval < 2, "Attempting to retrive pdf value larger than state size"
        return d[dval]

    # numpy categorical sampling hack
    def categorical(self, d):
        return np.flatnonzero( self.random_state.multinomial(1,d,1) )[0]

    ################################################################# 
    # Create functions
    ################################################################# 
    def create_transition_distribution(self):
        td = np.array([0.5, 0.5])
        return td

    def create_observation_distribution(self):
        od = np.array([0.5, 0.5])
        return od

    def create_belief(self):
        return DiscreteBelief(self.n_states())

    def initial_belief(self):
        return DiscreteBelief(self.n_states())

    def initial_state(self):
        return self.random_state.randint(2)

    ################################################################# 
    # Misc Functions
    ################################################################# 

    def isterminal(self, s):
        # no terminal state in model
        return False

    def index2action(self, ai):
        return ai

    def n_states(self):
        return 2

    def n_actions(self):
        return 3

    def n_obsevations(self):
        return 2

    ################################################################# 
    # Policies
    ################################################################# 

    def optimal_policy(self):
        def pol(b):
            if b[0] < 0.04:
                return 0
            elif b[0] > 0.96:
                return 1
            else:
                return 2
        return pol

