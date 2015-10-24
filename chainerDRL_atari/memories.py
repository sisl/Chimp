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
import random
import h5py

class ReplayMemoryHDF5(object):
  ''' Wrapper around a replay dataset residing on disk as HDF5. '''
  def __init__(self, settings, filename='memory.hdf5', overwrite=False):
    if overwrite:
      self.fp = h5py.File(filename, 'w')
    else:
      self.fp = h5py.File(filename, 'a')

    if all(x in self.fp for x in ('state', 'action', 'reward', 'terminal')):
      self.state = self.fp['state']
      self.memory_size = self.state.shape[0]

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
      state_shape = (settings['n_frames'],) + settings['screen_dims_new']

      self.state = self.fp.create_dataset(
        'state',
        (self.memory_size,) + state_shape,
        dtype='float32')
      self.fp.create_dataset('action', (self.memory_size,), dtype='uint8')
      self.fp.create_dataset('reward', (self.memory_size,), dtype='float32')
      self.fp.create_dataset('terminal', (self.memory_size,), dtype=bool)

      self.action = np.empty(self.memory_size, dtype=np.uint8)
      self.reward = np.empty(self.memory_size, dtype=np.float32)
      self.terminal = np.empty(self.memory_size, dtype=np.bool)

      self.state.attrs['head'] = 0
      self.state.attrs['valid'] = 0

    # count number of iterations
    self.counter = 0

    # index of current "write" location
    self.head = self.state.attrs['head']

    # greatest index of any valid experience; i.e., [0, self.valid)
    self.valid = self.state.attrs['valid']

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
    indices = random.sample(xrange(0, self.valid), batch_size)

    # can't include (head - 1)th state in sample because we don't know the next
    # state, so we simply resample; rare case if dataset is large
    while (self.head - 1) in indices:
      indices = random.sample(xrange(0, self.valid), batch_size)

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

    return self.state[indices], self.action[next_indices], self.reward[next_indices], next_states, self.terminal[next_indices]

  def store_tuple(self, prevstate, action, reward, state, terminal=False):
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

    # update head and valid pointers
    self.head = (self.head + 1) % self.memory_size
    self.valid = min(self.memory_size, self.valid + 1)

    self.counter += 1

  def __del__(self):
    ''' Stores the memory dataset into the file when program ends. '''
    self.fp['action'][:] = self.action
    self.fp['reward'][:] = self.reward
    self.fp['terminal'][:] = self.terminal
    self.state.attrs['head'] = self.head
    self.state.attrs['valid'] = self.valid
    self.fp.close()




'''
An alternative replay memory that does not utilize HDF5 - less eficient
'''

class ReplayMemory(object):

  def __init__(self, settings):
    self.memory_size = settings['memory_size']
    self.screen_dims_new = settings['screen_dims_new']
    self.n_frames = settings['n_frames']
    self.data = [np.zeros((self.memory_size, self.n_frames, self.screen_dims_new[0], self.screen_dims_new[1]), dtype=np.float32),
      np.zeros(self.memory_size, dtype=np.int32),
      np.zeros(self.memory_size, dtype=np.float32),
      np.zeros((self.memory_size, self.n_frames, self.screen_dims_new[0], self.screen_dims_new[1]), dtype=np.float32),
      np.zeros(self.memory_size, dtype=np.bool)]
    self.counter = 0

  # function to sample a mini-batch
  def minibatch(self, batch_size):
    # sampling a mini-batch of the given size with replacement
    ind = np.random.randint(0,min(self.counter,self.memory_size),batch_size)
    return self.data[0][ind], self.data[1][ind], self.data[2][ind], self.data[3][ind], self.data[4][ind]

  # function to store the observed experience and keep the count within the replay memory
  def store_tuple(self, s0, a, r, s1, episode_end_flag = False):

    # keep the most recent observations within the limit of the memory
    ind = self.counter % self.memory_size

    self.data[0][ind] = s0
    self.data[1][ind] = a
    self.data[2][ind] = r

    if not episode_end_flag:
      self.data[3][ind] = s1

    self.data[4][ind] = episode_end_flag
    
    self.counter += 1