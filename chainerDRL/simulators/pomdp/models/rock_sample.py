import numpy as np
from copy import deepcopy
from tools.belief import DiscreteBelief

#################################################################
# Implements the Rock Sample POMDP problem
#################################################################

class RockSamplePOMDP():

    # constructor
    def __init__(self, 
                 xs=7, # size of grid y dim
                 ys=8, # size of grid x dim
                 rocks={(1,2):False, (3,4):True}, # rock locations and types
                 seed=999, # random seed
                 rbad=-10.0, rgood=10.0, rexit=10.0, # reward values
                 eta=0.5, # quality of rover observation,
                 discount=0.99):
        
        self.random_state = np.random.RandomState(seed) # used for sampling
        self.discount = discount

        self.xs = xs # y-size of the grid
        self.ys = ys # x-size of the grid

        self.rocks = rocks # dictionary mapping rock positions to their types (x,y) => good or bad
        k = len(rocks)
        self.k = k # number of rocks

        self.rbad = rbad
        self.rgood = rgood
        self.rexit = rexit

        # states: state is represented by the rover position and the rock types
        self.rover_states = [i for i in range(xs*ys)] # fully observable vars
        self.rock_states = [i for i in range(2**k)] # partially observable vars
        self.n_rock_states = xs*ys
        self.n_rover_states = 2**k
        
        # actions: total of 5+k
        self.ractions = [0, # move left
                         1, # move right
                         2, # move up
                         3, # move down
                         4] # sample
        for i in range(k):
            self.ractions.append(5+i) # sample rock i

        # observations
        self.robs = [0, 1, 2] # none, good, bad

        # pre-allocate state variables
        self.rover_state = np.zeros(2) # rover (x,y) position
        self.rock_state = np.zeros(k, dtype=np.bool) # (good, bad) type for each rock



    ################################################################# 
    # Setters
    ################################################################# 
    def set_discount(self, d):
        self.discount = d

    def set_rewards(self, rs, rg, rb, re, rm):
        self.rsample = rs
        self.rgood = rg
        self.rbad = rb
        self.rexit = re


    ################################################################# 
    # S, A, O Spaces
    ################################################################# 
    def fully_obs_states(self):
        return self.rover_states

    def partially_obs_states(self):
        return self.rock_states

    def actions(self):
        return self.ractions

    def observations(self):
        return self.robs

    ################################################################# 
    # Reward Function
    ################################################################# 
    def reward(self, x, y, a):
        # Rewarded:
        # sampling good or bad rocks
        # exiting the map
        rocks = self.rocks
        xpos, ypos = x

        # if in terminal state, no reward 
        if self.isterminal(x, y):
            return 0.0
        # if exit get exit reward
        if a == 1 and xpos == self.xs:
            return self.rexit 
        # if trying to sample
        if a == 4:
            # if in a space with a rock
            if x in rocks:
                # if rock is good
                if rocks[x]:
                    return self.rgood
                # if rock is bad
                else:
                    return self.rbad
        return 0.0

    ################################################################# 
    # Distribution Functions
    ################################################################# 
    # rover moves determinisitcally: distribution is just the position of rover 
    def fully_obs_transition(self, x, y, a, dist):
        xpos = dist[0]
        ypos = dist[1]
        # going left
        if a == 0 and xpos > 0:
            xpos -= 1
        # going right
        elif a == 1 and xpos < (xs+1):
            xpos += 1
        # going up
        elif a == 2 and ypos < ys:
            ypos += 1
        # going down
        elif a == 3 and ypos > 0:
            ypos -= 1
        dist[0] = xpos
        dist[1] = ypos
        return dist

    # the positions of rocks or their types (good or bad) don't change
    def partially_obs_transition(self, x, y, a, dist):
        return dist

    # sample the transtion distribution 
    def sample_fully_obs_state(self, d):
        # deterministic transition
        return d 

    def sample_partially_obs_state(self, d):
        # rock states do not change
        return d

    # returns the observation dsitribution of o from the (s,a) pair
    def observation(self, x, y, a, dist):



    # sample the observation distirbution
    def sample_observation(self, d):
        oidx = self.categorical(d)
        return self.tobs[oidx]

    # pdf should be in a distributions module
    def pdf(self, d, dval):
        assert dval < 2, "Attempting to retrive pdf value larger than state size"
        return d[dval]

    # numpy categorical sampling hack
    def categorical(self, d):
        return np.flatnonzero( self.random_state.multinomial(1,d,1) )[0]


    ################################################################# 
    # Create functions
    ################################################################# 
    def create_fully_obs_transition_distribution(self):
        td = np.array([0,0]) # position of rover
        return td

    def create_partially_obs_transition_distribution(self):
        td = deepcopy(self.rocks)
        return td

    def create_observation_distribution(self):
        od = np.array([0.5, 0.5]) # good or bad rock
        return od

    def create_belief(self):
        return DiscreteBelief(self.n_states())

    def initial_belief(self):
        return DiscreteBelief(self.n_states())

    def initial_fully_obs_state(self):
        # returns a (0, y) tuple
        return (0, self.random_state.randint(self.xs+1))

    def initial_partially_obs_state(self):
        return deepcopy(self.rocks)


    ################################################################# 
    # Misc Functions
    ################################################################# 

    def isterminal(self, x, y):
        xpos, ypos = x
        if xpos > self.xs:
            return True
        return False

    def index2action(self, ai):
        return ai

    def n_states(self):
        return 2

    def n_actions(self):
        return 3

    def n_obsevations(self):
        return 2


