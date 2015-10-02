import numpy as np

# object to load (s0,a,r,s1) marketing data - to verify the system works
# THIS WILL BE REPLACED WITH EXPERIENCE REPLAY MODULE

# NOTE: WE NEED TO COME UP WITH A CONSISTENT NOTION OF A TRAINING EPOCH
# FOR OUR PURPOSES

class Sampler(object):

  def __init__(self, data, batch_size=200):
    self.data = data # data passed to the sampler is assumed to be a dict. with train, val and test data sets
    self.batch_size = batch_size
    self.data_size = {'train' : data['train'].shape[0],
         'val' : data['val'].shape[0],
         'test' : data['test'].shape[0]}
    # generate and shuffle index used in mini-batch sampling
    self.data_index = {'train' : np.arange(self.data_size['train']), 
          'val' : np.arange(self.data_size['val']),
          'test' : np.arange(self.data_size['test'])}
    # in-place shuffle
    np.random.shuffle(self.data_index['train'])
    np.random.shuffle(self.data_index['val'])
    np.random.shuffle(self.data_index['test'])
    # keeping track of the used mini-batch from the random index
    self.batch_ix = {'train' : 0, 'val' : 0, 'test' : 0}

  # function to sample a mini-batch from the desired data 'split' (train,val,test)
  def minibatch(self, split):
    # "split" indicates which data set to use (train, val, test)
    # if the unused part of the random index is enough to sample the next batch
    if self.batch_ix[split] + self.batch_size < self.data_size[split]:
      # generate the index for the next batch
      ind = self.data_index[split][range(self.batch_ix[split],self.batch_ix[split]+self.batch_size)]
      # shift the indicator by the batch size
      self.batch_ix[split] += self.batch_size
    else:
      # if not sufficient data for the full batch, use only the remaining part of index
      ind = self.data_index[split][range(self.batch_ix[split],self.data_size[split])]
      # reset the indicator
      self.batch_ix[split] = 0
      # reshuffle the random index for sampling
      np.random.shuffle(self.data_index[split])
    return self.data[split][ind,:5], self.data[split][ind,5].astype(np.int32)-1, self.data[split][ind,6], self.data[split][ind,7:]

  # function to reset (reshuffle) the random index used for mini-batch sampling
  # for the selected data set (train,val,test)
  def reset(self, split):
    self.batch_ix[split] = 0
    np.random.shuffle(self.data_index[split])