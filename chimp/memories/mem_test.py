import numpy as np
from replay_memory import ReplayMemoryHDF5


settings = {
        'save_dir' : 'results/test',
        'seed_memory' : 1,
        'history_sizes' : (5, 2, 0),
        'memory_size' : 1000,
        'model_dims' : (1,20),
        'batch_size' : 32
    }

mem = ReplayMemoryHDF5(settings)

o_dims = settings['model_dims']

for i in xrange(1000):
    obs = np.random.random(o_dims) + i # random obs
    a = np.random.randint(10) + i# 10 actions
    r = np.random.rand() + i
    obsp = np.random.random(o_dims) + i
    term = bool(np.random.binomial(1,0.1)) # 10% chance reach terminal state
    mem.store_tuple(obs, a, r, obsp, term)

o,a,r,op,terms=mem.minibatch()
#mem.close()
