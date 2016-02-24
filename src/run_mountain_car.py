"""
File to initialize training.
Contains settings, network definition for Chainer.
Creates the simulator, replay memory, DQN learner, and passes these to the agent framework for training.
"""

import numpy as np

from simulators.mdp.MDPSimulator import MDPSimulator
from simulators.mdp import MountainCar 
from utils.Policies import RandomPolicy

mdp = MountainCar()
simulator = MDPSimulator(mdp)
policy = RandomPolicy(simulator.n_actions())

rtot = simulator.simulate(100, policy, verbose=True)


