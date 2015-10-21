import numpy as np

# object to load (s0,a,r,s1) marketing data - to verify the system works
# THIS WILL BE REPLACED WITH ReplayMemory REPLAY MODULE

# NOTE: WE NEED TO COME UP WITH A CONSISTENT NOTION OF A TRAINING EPOCH
# FOR OUR PURPOSES

class ReplayMemory(object):

  def __init__(self, data):
    self.data = data # data passed to the ReplayMemory is assumed to be a dict. with train, val and test data sets
    self.data_size = {'train' : data['train'].shape[0],
         'val' : data['val'].shape[0],
         'test' : data['test'].shape[0]}

  # function to sample a mini-batch from the desired data 'split' (train,val,test)
  def minibatch(self, split, batch_size):
    # sampling a mini-batch of the given size with replacement
    ind = np.random.randint(0,self.data_size[split],batch_size)
    return self.data[split][ind,:5], self.data[split][ind,5].astype(np.int32)-1, self.data[split][ind,6], self.data[split][ind,7:]

