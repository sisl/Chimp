''' Implements type for reading/writing experiences to the replay dataset.

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

class ReplayMemoryHDF5(object):
    ''' Wrapper around a replay dataset residing on disk as HDF5. '''
    
    def __init__(self, settings, filename='memory.hdf5', overwrite=True):

        filename = settings['save_dir'] + '_' + filename

        self.random_state = np.random.RandomState(settings['seed_memory'])

        if overwrite:
            self.fp = h5py.File(filename, 'w')
        else:
            self.fp = h5py.File(filename, 'a')

        if all(x in self.fp for x in ('state', 'ahist', 'action', 'reward', 'terminal')):
            self.state = self.fp['state']
            self.memory_size = self.state.shape[0]

            self.ahist = np.empty((self.memory_size,settings['n_frames']), dtype=np.float32)
            self.fp['ahist'].read_direct(self.ahist)

            self.action = np.empty(self.memory_size, dtype=np.uint8)
            self.fp['action'].read_direct(self.action)

            self.reward = np.empty(self.memory_size, dtype=np.float32)
            self.fp['reward'].read_direct(self.reward)

            self.terminal = np.empty(self.memory_size, dtype=bool)
            self.fp['terminal'].read_direct(self.terminal)

            if self.memory_size != settings['memory_size']:
                print("Warning: dataset loaded from %s is of size %d, "
                    "not %d as indicated in |settings|. Using existing size."
                    % (filename, self.memory_size, settings['memory_size']))

        else:
            self.memory_size = settings['memory_size']
            state_shape = (settings['n_frames'],) + settings['model_dims']

            self.state = self.fp.create_dataset('state',
            (self.memory_size,) + state_shape, dtype='float32')

            self.fp.create_dataset('ahist', (self.memory_size,settings['n_frames']), dtype='float32')
            self.fp.create_dataset('action', (self.memory_size,), dtype='uint8')
            self.fp.create_dataset('reward', (self.memory_size,), dtype='float32')
            self.fp.create_dataset('terminal', (self.memory_size,), dtype=bool)

            self.ahist = np.empty((self.memory_size,settings['n_frames']), dtype=np.float32)
            self.action = np.empty(self.memory_size, dtype=np.uint8)
            self.reward = np.empty(self.memory_size, dtype=np.float32)
            self.terminal = np.empty(self.memory_size, dtype=np.bool)

            self.state.attrs['head'] = 0
            self.state.attrs['valid'] = 0

        # index of current "write" location
        self.head = self.state.attrs['head']

        # greatest index of any valid experience; i.e., [0, self.valid)
        self.valid = self.state.attrs['valid']


    #@profile 
    def minibatch(self, batch_size):
        ''' Uniformly sample (s,a,r,s') experiences from the replay dataset.

        Args:
            batch_size: (self-explanatory)

        Returns:
            Five numpy arrays that corresponds to s, a, r, s', and the terminal
            state indicator.
        '''
        if batch_size >= self.valid:
            raise ValueError("Can't draw sample of size %d from replay dataset of size %d"
                       % (batch_size, self.valid))

        # sampling without replacement
        indices = self.random_state.choice(xrange(0, self.valid), size=batch_size, replace=False).tolist()

        # can't include (head - 1)th state in sample because we don't know the next
        # state, so we simply resample; rare case if dataset is large
        while (self.head - 1) in indices:
            indices = self.random_state.choice(xrange(0, self.valid), size=batch_size, replace=False).tolist()

        indices.sort()  # slicing for hdf5 must be in increasing order

        next_indices = [idx + 1 for idx in indices]

        # handle case where next_state wraps around end of dataset
        if next_indices[-1] == self.memory_size:
            next_indices[-1] = 0
            shape = (batch_size,) + self.state[0].shape
            next_states = np.empty(shape, dtype=np.float32)
            next_states[0:-1] = self.state[next_indices[0:-1]]
            next_states[-1] = self.state[0]
        else:
            next_states = self.state[next_indices]

        return self.state[indices], self.ahist[indices], self.action[next_indices], self.reward[next_indices], next_states, self.ahist[next_indices], self.terminal[next_indices]

    #@profile
    def store_tuple(self, prevstate, prevahist, action, reward, state, ahist, terminal=False):
        ''' Stores an experience tuple into the replay dataset, i.e., a 
        triple (action, reward, state) where |state| is the result of the
        agent taking |action| and |reward| is from the agent taking |action|
        at the previous state. The previous state is assumed to be the state
        at index (self.head - 1).

        Args:
            action: index of action chosen
            reward: float value of reward
            state: numpy array of shape NUM_FRAMES x FRAME_WIDTH x FRAME_HEIGHT
                or None if the input action ended the game

        '''
        self.action[self.head] = action
        self.reward[self.head] = reward
        self.terminal[self.head] = terminal
        if not terminal:
            self.state[self.head] = state
            self.ahist[self.head] = ahist

        # update head and valid pointers
        self.head = (self.head + 1) % self.memory_size
        self.valid = min(self.memory_size, self.valid + 1)
        

    def __del__(self):
        ''' Stores the memory dataset into the file when program ends. '''
        self.fp['ahist'][:] = self.ahist
        self.fp['action'][:] = self.action
        self.fp['reward'][:] = self.reward
        self.fp['terminal'][:] = self.terminal
        self.state.attrs['head'] = self.head
        self.state.attrs['valid'] = self.valid
        self.fp.close()
