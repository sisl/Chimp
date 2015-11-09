import numpy as np
import sys
from models.tiger import TigerPOMDP
from models.simulator import POMDPSimulator

#####################################################################
# This is a sample simulation loop for the DRL framework using POMDPs
#####################################################################

# initialize pomdp
pomdp = TigerPOMDP(seed=1)

# initialize and pass the pomdp into simulator
sim = POMDPSimulator(pomdp) # state and initial belief automatically initialized

sim.n_states # number of states-input layer size

opt = pomdp.optimal_policy()

steps = 100

rtot = 0.0

for i in range(steps):
    # get the initial state
    s = sim.get_screenshot()
    # pick random action
    #ai = np.random.randint(sim.n_actions)
    # pick optimal aciton
    ai = opt(s) 

    # progress simulation
    sim.act(ai)

    # get reward and next states
    r = sim.reward() # real valued reward
    sp = sim.get_screenshot() # pomdp state, this is a belief

    print "Action ", ai, " Reward: ", r, " Screen Shot: ", sp
    print "Current State: ", sim.current_state, " Current Belief: ", sim.current_belief.bnew, "\n"

    rtot += r

    # check if reached terminal state
    if sim.episode_over():
        sim.reset_episode()

print "Total reward: ", rtot
