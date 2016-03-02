import numpy as np

#################################################################
# Implements multi-agent controllers
#################################################################


class RandomPolicy():

    # constructor
    def __init__(self, n_actions, rng = np.random.RandomState()):
        self.rng = rng
        self.n_actions = n_actions

    def action(self, obs):
        return self.rng.randint(self.n_actions) 


class OneStepLookAhead():

    # constructor
    def __init__(self, simulator, n_rollouts=100):
        self.simulator = simulator

    def action(self, obs):
        # run each action n_rollouts times, take the highest average
        pass

