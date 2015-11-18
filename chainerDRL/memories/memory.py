'''
An alternative replay memory that does not utilize HDF5 - less efficient
'''

import numpy as np

class ReplayMemory(object):

    def __init__(self, settings):
        
        self.random_state = np.random.RandomState(settings['seed_memory'])
        self.memory_size = settings['memory_size']
        self.model_dims = settings['model_dims']
        self.n_frames = settings['n_frames']
        self.data = [np.zeros((self.memory_size, self.n_frames, self.model_dims[0], self.model_dims[1]), dtype=np.float32),
            np.zeros((self.memory_size, self.n_frames), dtype=np.float32),
            np.zeros(self.memory_size, dtype=np.int32),
            np.zeros(self.memory_size, dtype=np.float32),
            np.zeros((self.memory_size, self.n_frames, self.model_dims[0], self.model_dims[1]), dtype=np.float32),
            np.zeros((self.memory_size, self.n_frames), dtype=np.float32),
            np.zeros(self.memory_size, dtype=np.bool)]
        self.counter = 0

    # function to sample a mini-batch
    def minibatch(self, batch_size):
        # sampling a mini-batch of the given size with replacement
        ind = self.random_state.randint(0,min(self.counter,self.memory_size),batch_size)
        return self.data[0][ind], self.data[1][ind], self.data[2][ind], self.data[3][ind], self.data[4][ind], self.data[5][ind], self.data[6][ind]

    # function to store the observed experience and keep the count within the replay memory
    def store_tuple(self, s0, ahist0, a, r, s1, ahist1, episode_end_flag = False):

        # keep the most recent observations within the limit of the memory
        ind = self.counter % self.memory_size

        self.data[0][ind] = s0
        self.data[1][ind] = ahist0
        self.data[2][ind] = a
        self.data[3][ind] = r

        if not episode_end_flag:
            self.data[4][ind] = s1
            self.data[5][ind] = ahist1

        self.data[6][ind] = episode_end_flag
    
        self.counter += 1
