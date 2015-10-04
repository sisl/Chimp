import numpy as np
from copy import deepcopy

#################################################################
# Implements Belief and Belief Updater
#################################################################

class DiscreteBelief():

    def __init__(self, n):
        self.bold = np.zeros(n)
        self.bnew = np.zeros(n)
        self.n = n

    def __getitem__(self, idx):
        return self.bnew[idx]

    def __setitem__(self, idx, val):
        self.bold[idx] = val
        self.bnew[idx] = val

    def update(self, pomdp, a, o):
        
        sspace = pomdp.states()
        
        td = pomdp.create_transition_distribution()
        od = pomdp.create_observation_distribution()

        # old belief is now new, new is fresh
        (bold, bnew) = (bnew, bold)
        self.empty_new()

        for (i, sp) in enumerate(sspace):
            # get the distributions
            od = pomdp.observation(s, a, od)
            # get the prob of o from the current distribution
            probo = pomdp.pdf(od, o)
            # 


    def length(self):
        return self.n

    def empty(self):
        self.bold.fill(0.0)
        self.bnew.fill(0.0)

    def empty_old(self):
        self.bold.fill(0.0)

    def empty_new(self):
        self.bnew.fill(0.0)

    def old_belief(self):
        return self.bold

    def new_belief(self):
        return self.bnew

