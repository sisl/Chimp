import numpy as np
from models.rock_sample import RockSamplePOMDP
from models.simulator_momdp import MOMDPSimulator

from datetime import datetime
import numpy as np

#####################################################################
# This is a sample simulation loop for the DRL framework using POMDPs
#####################################################################

# initialize pomdp
#pomdp = RockSamplePOMDP(seed=np.random.randint(1,1000),h_conf=0.5)
pomdp = RockSamplePOMDP(seed=100, h_conf=0.15, d0=10000)

# initialize and pass the pomdp into simulator
sim = MOMDPSimulator(pomdp) # state and initial belief automatically initialized

sim.model_dims # number of states-input layer size

steps = 10000

rtot = 0.0

st = datetime.now()

for i in xrange(steps):
    # get the initial state
    s = sim.get_screenshot()
    # pick random action
    #ai = np.random.randint(sim.n_actions)
    ai = pomdp.heuristic_policy(s)
    #ai = 1
    
    # progress simulation
    sim.act(ai)
    # get reward and next states
    r = sim.reward() # real valued reward
    sp = sim.get_screenshot() # pomdp state, this is a belief
    #print "Time Step: ", i
    #print "Action ", ai, " Reward: ", r, " Current X State: ", sim.current_xstate, " Current Y State: ", sim.current_ystate, "\n"
    #print "Screen shot: ", sp
    rtot += r
    # check if reached terminal state
    if sim.episode_over():
        #print "\n\nResetting Episode\n\n"
        sim.reset_episode()

print "Runtime: ", datetime.now() - st
print "Total reward: ", rtot

