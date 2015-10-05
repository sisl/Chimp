import numpy as np
from copy import deepcopy

#################################################################
# Implements Belief and Belief Updater
#################################################################

class DiscreteBelief():

    def __init__(self, n):
        self.bold = np.zeros(n) + 1.0/n
        self.bnew = np.zeros(n) + 1.0/n
        self.n = n

    def __getitem__(self, idx):
        return self.bnew[idx]

    def __setitem__(self, idx, val):
        self.bold[idx] = val
        self.bnew[idx] = val

    def update(self, pomdp, a, o):
       
        # swap pointers
        (bnew, bold) = (self.bold, self.bnew)

        sspace = pomdp.states()
        
        td = pomdp.create_transition_distribution()
        od = pomdp.create_observation_distribution()

        # old belief is now new, new is fresh
        bnew.fill(0.0)

        for (i, sp) in enumerate(sspace):
            # get the distributions
            od = pomdp.observation(sp, a, od)
            # get the prob of o from the current distribution
            probo = pomdp.pdf(od, o)
            # if observation prob is 0.0, then skip rest of update b/c bnew[i] is zero
            if probo == 0.0:
                continue
            b_sum = 0.0 # belef for state sp
            for (j, s) in enumerate(sspace):
                td = pomdp.transition(s, a, td)
                pp = pomdp.pdf(td, sp)
                b_sum += pp * bold[j]
                print i, j, pp, bold[j]
            bnew[i] = probo * b_sum
        norm = sum(bnew)
        print norm
        for i in range(self.length()):
            bnew[i] /= norm
        (self.bnew, self.bold) = (bnew, bold)
        return self

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

