#!/usr/bin/env python 
'''
@author jstober

Simple class to track knowledge of states and actions. Based on 

L. Li, M. L. Littman, and C. R. Mansley, "Online exploration in least-squares policy iteration" AAMAS, 2009.
'''
import numpy as np
import pdb
import copy

class TrackKnown:
    """
    Track knowledge of states and actions.

    TODO: Generalize by adding epsilon and kd tree or approximation methods.
    """
    def __init__(self, samples, nstates, nactions, mcount):
        self.nstates = nstates
        self.nactions = nactions
        self.mcount = mcount
        self.counts = np.zeros((nstates, nactions))
        self.samples = samples
        for (s,a,r,ns,na) in samples:
            self.counts[s,a] += 1

    def __iter__(self):
        return iter(self.samples)

    def __getitem__(self,key):
        new = copy.copy(self) # note: only need a shallow copy here *do not modify the slice object*
        new.samples = self.samples[key]
        return new

    def __len__(self):
        return len(self.samples)

    def extend(self, samples, take_all=False):
        if take_all:
            for (s,a,r,ns,na) in samples:
                self.counts[s,a] += 1
                self.samples.append((s,a,r,ns,na))
        else:
            for (s,a,r,ns,na) in samples:
                if self.counts[s,a] < self.mcount:
                    self.counts[s,a] += 1
                    self.samples.append((s,a,r,ns,na))

    def all_known(self):
        if (self.counts >= self.mcount).all():
            return True
        else:
            return False

    def known_pair(self,s,a):
        if self.counts[s,a] >= self.mcount:
            return True
        else:
            return False

    def known_state(self,s):
        if np.greater_equal(self.counts[s,:],self.mcount).all():
            return True
        else:
            return False

    def unknown(self,s):
        # indices of actions with low counts.
        return np.where(self.counts[s,:] < self.mcount)[0]

    def diagnostics(self):
        states, actions = np.nonzero(self.counts < self.mcount)
        print "Unknown states: ", len(np.unique(states))