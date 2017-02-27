#################################################################
# Implements distriubtions for POMDP models
#################################################################

import numpy as np
from copy import deepcopy

class Categorical():

    def __init__(self, n):
        self.indices = np.zeros(n, dtype=np.int64)
        self.weights = np.zeros(n) + 1.0/n
        self.n = n

    def __getitem__(self, idx):
        return (self.indices[idx], self.weights[idx])

    def __setitem__(self, idx, val):
        self.

    def sample(self):
        idx = self.quantile(np.random.rand())
        return self.indices[idx]


    def quantile(self, p):
        k = self.n
        pv = self.weights
        i = 1
        v = pv[1]
        while v < p and i < k:
            i += 1
            v += pv[i]
        return i

