import random 
import sys
from tiger import TigerPOMDP
from simulator import POMDPSimulator

#####################################################################
# This is a sample simulation loop for the DRL framework using POMDPs
#####################################################################

# initialize pomdp
pomdp = TigerPOMDP()

# initialize and pass the pomdp into simulator
sim = POMDPSimulator(pomdp) # state and initial belief automatically initialized

na = sim.n_actions() # number of actions-output layer size
ns = sim.n_states() # number of states-input layer size


steps = 100

for i in range(steps):
    # pick random action
    ai = random.randint(0,na)

    # progress simulation
    sim.act(ai)

    # get reward and next states
    r = sim.reward() # real valued reward
    s = sim.get_screenshot() # pomdp state, this is a belief

    print s, ai, r

    # check if reached terminal state
    if sim.episode_over():
        sim.reset_episode()
