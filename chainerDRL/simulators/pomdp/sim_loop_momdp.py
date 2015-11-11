import numpy as np
from models.rock_sample import RockSamplePOMDP
from models.simulator_momdp import MOMDPSimulator

#####################################################################
# This is a sample simulation loop for the DRL framework using POMDPs
#####################################################################

# initialize pomdp
pomdp = RockSamplePOMDP(seed=1)

# initialize and pass the pomdp into simulator
sim = MOMDPSimulator(pomdp) # state and initial belief automatically initialized

sim.model_dims # number of states-input layer size

steps = 100

rtot = 0.0

for i in xrange(steps):
    # get the initial state
    s = sim.get_screenshot()
    # pick random action
    ai = np.random.randint(sim.n_actions)

    # progress simulation
    sim.act(ai)

    # get reward and next states
    r = sim.reward() # real valued reward
    sp = sim.get_screenshot() # pomdp state, this is a belief

    print "Time Step: ", i
    print "Action ", ai, " Reward: ", r, " Screen Shot: ", sp
    print "Current X State: ", sim.current_xstate, " Current Y State: ", sim.current_ystate, " Current Belief: ", sim.current_belief.bnew, "\n"

    rtot += r

    # check if reached terminal state
    if sim.episode_over():
        print "\n\nResetting Episode\n\n"
        sim.reset_episode()

print "Total reward: ", rtot

