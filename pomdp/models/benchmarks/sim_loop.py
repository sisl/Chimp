import random 
import sys
from tiger import TigerPOMDP
from simulator import POMDPSimulator

# initialize pomdp
pomdp = TigerPOMDP()

# pass the pomdp into simulator
sim = POMDPSimulator(pomdp)

na = sim.n_actions() # number of actions-output layer size
ns = sim.n_states() # number of states-input layer size

steps = 100

s = sim.get_screenshot()

for i in range(steps):
    ai = random.randint(0,na)

    sim.act(ai)

    r = sim.reward()
    s = sim.get_screenshot()

    print s, ai, r

    if sim.episode_over():
        sim.reset_episode()
