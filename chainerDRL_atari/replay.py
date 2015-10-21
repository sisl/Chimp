import numpy as np

# object to store (s0,a,r,s1) data

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

  # function to sample a mini-batch from the desired data 'split' (train,val,test)
  def minibatch(self, batch_size):
    # sampling a mini-batch of the given size with replacement
    ind = np.random.randint(0,min(self.counter,self.memory_size),batch_size)
    return self.data[0][ind], self.data[1][ind], self.data[2][ind], self.data[3][ind], self.data[4][ind]

  def storeTuple(self, s0, a, r, s1, episode_end_flag = False):

        # while still space in the memory, put the new observation into the next empty spot
        if self.counter < self.memory_size:
          ind = self.counter
        # when the replay memory has been filed, choose a random stored observation and replace it
        else:
          ind = np.random.randint(0,self.memory_size)

        self.data[0][ind] = s0
        self.data[1][ind] = a
        self.data[2][ind] = r

        if not episode_end_flag:
          self.data[3][ind] = s1

        self.data[4][ind] = episode_end_flag

        self.counter += 1

