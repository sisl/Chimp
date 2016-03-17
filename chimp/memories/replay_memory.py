''' Implements class for reading/writing experiences to the replay dataset.

We assume
(1) Actions and rewards for the full history fit comfortably in memory,
(2) The belief state representation for the full history does not,
(3) A single sample of belief states fits comfortably in memory.

For instance, if the replay dataset stores the last 1 million experiences,
then the history of actions is 1 byte x 1 M = 1 MB. The same holds for the
history of rewards. However, a modest belief state representation might be
a dense vector with a maximum of 1,000 Float64 elements (typical state spaces
are on the order of millions). In this case the full history of 1 million
states would be (1,000 elem x 8 bytes x 1 M = 8 GB).

N.B.!
Memory is organized as (a, r, s', end_of_game_flag). We refer to s' 
simply as "state". To sample (s, a, r, s', end_of_game_flag) 
we take s' from the current location in memory, and (a, r, s', end_of_game_flag) 
from the location one step forward.
'''

import numpy as np
import h5py
import os

class ReplayMemoryHDF5(object):
    ''' Wrapper around a replay dataset residing on disk as HDF5. '''
    
    def __init__(self, settings, filename='memory.hdf5', overwrite=True, empty=-1):

        if not os.path.exists(settings['save_dir']):
            os.makedirs(settings['save_dir'])

        filename = settings['save_dir'] + '/' + filename
        self.random_state = np.random.RandomState(settings['seed_memory'])
        self.ohist_size, self.ahist_size, self.rhist_size = settings['history_sizes']

        self.ahist_size = 1 if self.ahist_size is 0 else self.ahist_size
        self.rhist_size = 1 if self.rhist_size is 0 else self.rhist_size

        self.max_size = max(settings['history_sizes'])
        self.batch_size = settings['batch_size']

        if overwrite:
            self.fp = h5py.File(filename, 'w')
        else:
            self.fp = h5py.File(filename, 'a')

        if all(x in self.fp for x in ('observations', 'actions', 'rewards', 'next_observations', 'terminals')):
            self.observations = self.fp['observations']
            self.memory_size = self.observations.shape[0]

            self.actions = np.empty(self.memory_size, dtype=np.int32)
            self.fp['actions'].read_direct(self.actions)

            self.rewards = np.empty(self.memory_size, dtype=np.float32)
            self.fp['rewards'].read_direct(self.rewards)

            self.next_observations = self.fp['next_observations']

            self.terminals = np.empty(self.memory_size, dtype=bool)
            self.fp['terminals'].read_direct(self.terminals)

            if self.memory_size != settings['memory_size']:
                print("Warning: dataset loaded from %s is of size %d, "
                    "not %d as indicated in |settings|. Using existing size."
                    % (filename, self.memory_size, settings['memory_size']))

        else:
            self.memory_size = settings['memory_size']
            obs_shape = settings['model_dims']

            self.observations = self.fp.create_dataset('observations', (self.memory_size,) + obs_shape, dtype=np.float32)
            self.next_observations = self.fp.create_dataset('next_observations', (self.memory_size,) + obs_shape, dtype=np.float32)

            self.fp.create_dataset('actions', (self.memory_size,), dtype='int32')
            self.fp.create_dataset('rewards', (self.memory_size,), dtype='float32')
            self.fp.create_dataset('terminals', (self.memory_size,), dtype=bool)

            self.actions = np.empty(self.memory_size, dtype=np.int32)
            self.rewards = np.empty(self.memory_size, dtype=np.float32)
            self.terminals = np.empty(self.memory_size, dtype=np.bool)

            self.observations.attrs['head'] = 0
            self.observations.attrs['valid'] = 0

        # index of current "write" location
        self.head = self.observations.attrs['head']

        # greatest index of any valid experience; i.e., [0, self.valid)
        self.valid = self.observations.attrs['valid']

        # initialize histories
        self.ohist = np.zeros((self.batch_size, self.ohist_size) + obs_shape, dtype=np.float32)
        self.ophist = np.zeros((self.batch_size, self.ohist_size) + obs_shape, dtype=np.float32)
        self.ahist = np.zeros((self.batch_size, self.ahist_size), dtype=np.int32)
        self.rhist = np.zeros((self.batch_size, self.rhist_size), dtype=np.float32)
        self.thist = np.zeros((self.batch_size, self.ohist_size), dtype=np.bool)

        self._emptyint = np.int32(empty)
        self._emptyfloat = np.float32(empty)

    def minibatch(self):
        ''' Uniformly sample (o,a,r,o') experiences from the replay dataset.

        Args:
            batch_size: size of mini-batch

        Returns:
            Five numpy arrays that corresponds to o, a, r, o', and the terminal
            state indicator.
        '''
        batch_size = self.batch_size
        if batch_size >= self.valid:
            raise ValueError("Can't draw sample of size %d from replay dataset of size %d"
                       % (batch_size, self.valid))

        ohist_size, ahist_size, rhist_size = self.ohist_size, self.ahist_size, self.rhist_size
        max_hist = self.max_size

        indices = self.get_indices(batch_size)

        self.clear_history()

        # TODO: can we get rid of this loop by sorting inidces and then reshaping? 
        for i in xrange(batch_size):
            # all end on the same index
            endi = indices[i]
            starti = endi - max_hist
            # starting indecies if no terminal states
            starto, starta, startr = endi-ohist_size, endi-ahist_size, endi-rhist_size

            # look backwards and find first terminal state
            termarr = np.where(self.terminals[starti:endi-1]==True)[0]
            termidx = starti
            if termarr.size is not 0:
                termidx = endi - (endi-starti - termarr[-1]) + 1

            # if starts before terminal, change start index
            starto = termidx if starto < termidx else starto
            starta = termidx if starta < termidx else starta
            startr = termidx if startr < termidx else startr

            ohl, ahl, rhl = (endi - starto), (endi - starta), (endi - startr)
        
            # load from memory
            self.ohist[i, ohist_size-ohl:] = self.observations[xrange(starto, endi)]
            self.ophist[i, ohist_size-ohl:] = self.next_observations[xrange(starto, endi)]
            self.ahist[i, ahist_size-ahl:] = self.actions[xrange(starta, endi)]
            self.rhist[i, rhist_size-rhl:] = self.rewards[xrange(startr, endi)]
            self.thist[i, ohist_size-ohl:] = self.terminals[xrange(starto, endi)]

        return self.ohist, self.ahist, self.rhist, self.ophist, self.thist


    def get_indices(self, batch_size):
        ohist_size, ahist_size, rhist_size = self.ohist_size, self.ahist_size, self.rhist_size
        max_hist = self.max_size

        # want to sample from valid history sets
        start_shift = self.random_state.randint(max_hist)

        # indices corresponding to ranges from which to sample
        indices = self.random_state.choice(xrange(1,self.valid/max_hist), size=batch_size, replace=False)
        # shift all the indices and offset
        indices *= max_hist
        indices += start_shift

        return indices


    def store_tuple(self, obs, action, reward, obsp, terminal):
        ''' Stores an experience tuple into the replay dataset, i.e., a 
        triple (obs, action, reward, obsp, terminal) where |obsp| is the observation
        made when the agent takes |action| and recieves |reward| 
        while |obs| is the observation made prior to taking |action|.
        The observation |obs| is assumed to be at index (self.head).

        Args:
            obs: observation made at time t of shape provided by user (obs_shape)
            action: index of action chosen
            reward: float value of reward recieved after taking action a
                or None if the input action ended the game
            terminal: indicates if obsp is terminal

        '''
        self.actions[self.head] = action
        self.rewards[self.head] = reward
        self.terminals[self.head] = terminal
        self.observations[self.head] = obs
        self.next_observations[self.head] = obsp

        # update head and valid pointers
        self.head = (self.head + 1) % self.memory_size
        self.valid = min(self.memory_size, self.valid + 1)
        
    def clear_history(self):
        self.ohist.fill(self._emptyfloat)
        self.ophist.fill(self._emptyfloat)
        self.ahist.fill(self._emptyint)
        self.rhist.fill(0.0)
        self.thist.fill(False)

    def close(self):
        ''' Stores the memory dataset into the file when program ends. '''
        self.fp['actions'][:] = self.actions
        self.fp['rewards'][:] = self.rewards
        self.fp['terminals'][:] = self.terminals
        self.observations.attrs['head'] = self.head
        self.observations.attrs['valid'] = self.valid
        self.fp.close()

    def __del__(self):
        try:
            self.close()
        except:
            pass # already closed
        