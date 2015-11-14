import numpy as np
from copy import deepcopy
from tools.belief_momdp import MOMDPBelief
import math
import itertools

#################################################################
# Implements the Rock Sample POMDP problem
#################################################################

class RockSamplePOMDP():

    # constructor
    def __init__(self, 
                 xs=7, # size of grid y dim
                 ys=7, # size of grid x dim
                 rocks={(2,4):False, (3,4):True, (5,5):False, # (2,0):False, (0,1):True, (3,1):False, (6,3):True,
                     (1,6):True}, # rock locations and types
                 seed=1, # random seed
                 rbad=-10.0, rgood=10.0, rexit=10.0, rbump=-100.0, # reward values
                 d0=10, # quality of rover observation,
                 h_conf=0.5, # confidence level before moving in heuristic policy
                 discount=0.99):
        
        self.random_state = np.random.RandomState(seed) # used for sampling
        self.discount = discount

        self.xs = xs - 1 # y-size of the grid
        self.ys = ys - 1 # x-size of the grid

        self.rocks = rocks # dictionary mapping rock positions to their types (x,y) => good or bad
        self.rock_pos = [k for k in sorted(rocks.keys())]
        self.rock_types = [rocks[k] for k in sorted(rocks.keys())]
        self.rock_map = {(k):i for (i, k) in enumerate(sorted(rocks.keys()))}
        k = len(rocks)
        self.k = k # number of rocks

        self.rbad = rbad
        self.rgood = rgood
        self.rbump = rbump
        self.rexit = rexit

        # states: state is represented by the rover position and the rock types
        self.rover_states = [(j,i) for i in range(xs) for j in range(ys)] # fully observable vars
        rs = itertools.product(*(xrange(2) for i in xrange(k)))
        self.rock_states = [[bool(j) for j in i] for i in rs]
        self.n_rock_states = len(self.rock_states)
        self.n_rover_states = len(self.rover_states)
        
        # actions: total of 5+k
        self.ractions = [0, # move left
                         1, # move right
                         2, # move up
                         3, # move down
                         4] # sample
        for i in range(k):
            self.ractions.append(5+i) # sample rock i

        # observations
        self.robs = [0, # none
                     1, # good
                     2] # bad

        # pre-allocate state variables
        self.rover_state = np.zeros(2) # rover (x,y) position
        self.rock_state = np.zeros(k, dtype=np.bool) # (good, bad) type for each rock

        self.d0 = d0
        self.h_conf = h_conf

        self.action_vectors = [[-1, 0], [1, 0], [0, 1], [0, -1]]

        # belief and observation dimensions
        self.xdims = 2
        self.odims = 1

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
        # trying to move off the grid
        rocks = self.rocks
        xpos, ypos = x

        # if in terminal state, no reward 
        if self.isterminal(x, y):
            return 0.0
        # if exit get exit reward
        if a == 1 and xpos == self.xs:
            return self.rexit 
        # if trying to move off the grid
        if (a == 0 and xpos == 0) or (a == 2 and ypos == self.ys) or (a == 3 and ypos == 0):
            return self.rbump
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
        xpos = x[0]
        ypos = x[1]
        # going left
        if a == 0 and xpos > 0:
            xpos -= 1
        # going right
        elif a == 1 and xpos < (self.xs+1):
            xpos += 1
        # going up
        elif a == 2 and ypos < self.ys:
            ypos += 1
        # going down
        elif a == 3 and ypos > 0:
            ypos -= 1
        dist[0] = xpos
        dist[1] = ypos
        return dist

    # the positions of rocks don't change, good rocks turn bad after sampling
    def partially_obs_transition(self, x, y, a, dist):
        # fill the distribution with our y var
        for i in xrange(len(y)):
            dist[i] = y[i]
        # if a rock is sampled it becomes bad
        if a == 4:
            rocks = self.rocks
            # if we are on a rock state change type to bad
            if x in rocks:
                ri = self.rock_map[x] 
                self.rock_types[ri] = False
                rocks[x] = False
                dist[ri] = False
        return dist

    # sample the transtion distribution 
    def sample_fully_obs_state(self, d):
        # deterministic transition
        return (d[0], d[1]) 

    def sample_partially_obs_state(self, d):
        # rock states do not change
        return d

    # returns the observation dsitribution of o from the (x,y,a) 
    def observation(self, x, y, a, dist):
        prob = 0.0
        # if the action checks a rock 
        if self.is_check_action(a):
            xpos = x[0]
            ypos = x[1]

            ri = self.act2rock(a) # rock index
            rock_pos = self.rock_pos[ri] # rock position
            rock_type = y[ri] # rock type

            r = math.sqrt((xpos - rock_pos[0])**2 + (ypos - rock_pos[1])**2) 
            eta = math.exp(-r/self.d0)
            p_correct = 0.5 + 0.5 * eta # probability of correct measure

            dist.fill(0.0)
            # if rock is good
            if rock_type == True:
                dist[1] = p_correct
                dist[2] = 1.0 - p_correct
            # rock is bad
            else:
                dist[1] = 1 - p_correct
                dist[2] = p_correct
        else:
            dist.fill(0.0)
            dist[0] = 1.0
        return dist


    # sample the observation distirbution
    def sample_observation(self, d):
        oidx = self.categorical(d)
        return self.robs[oidx]

    def fully_obs_transition_pdf(self, d, x): 
        if d[0] == x[0] and d[1] == x[1]:
            return 1.0
        else:
            return 0.0

    # only single rock configuration, always return 1
    def partially_obs_transition_pdf(self, d, y):
        if y == d:
            return 1.0
        else:
            return 0.0

    # pdf for observation prob
    def observation_pdf(self, d, dval):
        assert dval < 3, "Attempting to retrive pdf value larger than observation size"
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
        return deepcopy(self.rock_types)

    def create_observation_distribution(self):
        od = np.zeros(3) + 1.0/3 # none, good, bad 
        return od

    def create_belief(self):
        return MOMDPBelief(self.n_rock_states)

    def initial_belief(self):
        return MOMDPBelief(self.n_rock_states)

    def initial_fully_obs_state(self):
        # returns a (0, y) tuple
        return (0, self.random_state.randint(self.xs+1))

    def initial_partially_obs_state(self):
        for (i, k) in enumerate(sorted(self.rocks.keys())):
            t = bool(self.random_state.randint(2))
            self.rock_types[i] = t 
            self.rocks[k] = t
        return deepcopy(self.rock_types)


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

    def is_check_action(self, a):
        return True if a > 4 else False

    def act2rock(self, a):
        return a - 5

    def n_xstates(self):
        return len(self.rover_states)

    def n_ystates(self):
        return len(self.rock_states)

    def n_actions(self):
        return len(self.ractions)

    def n_obsevations(self):
        return 2


    ################################################################# 
    # Policies
    ################################################################# 

    def heuristic_policy(self, sc):
        # takes in a screen shot, [x, b] array
        x = (sc[0], sc[1]) # x and y pos
        b = np.array(sc[2:-1]) # belief
        return self.heuristic(x, b)

    def heuristic(self, x, b):
        # if we are not confident, keep checking randomly
        if b.max() < self.h_conf:
            return self.random_state.randint(5, 5+self.k)
        else:
            ri = b.argmax() # index of highest confidence rock state 
            y = self.rock_states[ri] # rock state
            # find closest good rock
            c = float('inf')
            ci = -1
            for (i, t) in enumerate(y):
                # if rock is good
                if t:
                    # if on the rock sample
                    if x == self.rock_pos[i]:
                        return 4
                    xrover = x[0]
                    yrover = x[1]
                    xrock, yrock = self.rock_pos[i]
                    dist = math.sqrt((xrock-xrover)**2 + (yrock-yrover)**2)
                    if dist < c:
                        c = dist
                        ci = i
            if ci > -1:
                return self.move_to(x, self.rock_pos[ci])
        # if no good rocks left move right
        return 1
                    
    # action to move rover from origin o to target t
    def move_to(self, o, t):
        # vector components
        v = [t[0] - o[0], t[1] - o[1]]
        sa = float('inf')
        ai = 1
        # move in the direction that minimizes angle between action and target
        for (i, a) in enumerate(self.action_vectors):
            ang = angle(v, a)
            if ang < sa:
                sa = ang
                ai = i
        return ai

def dotproduct(v1, v2):
    return sum((a*b) for a, b in zip(v1, v2))

def length(v):
  return math.sqrt(dotproduct(v, v))

def angle(v1, v2):
    return math.acos(dotproduct(v1, v2) / (length(v1) * length(v2)))    


