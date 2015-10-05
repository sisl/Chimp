import numpy as np
import sys
sys.path.append("../tools/")
from tiger import TigerPOMDP
from belief import DiscreteBelief

pomdp = TigerPOMDP()

# optimal policy for tiger
policy = pomdp.optimal_policy()

# initialized to unifrom
b = DiscreteBelief(pomdp.n_states())
ts = 100

# pointers to pre-allocated distributions
trans_dist = pomdp.create_transition_distribution()
obs_dist = pomdp.create_observation_distribution()

# initial state, belief
s = pomdp.initial_state()
b = pomdp.initial_belief()

# average rewards
rave = np.zeros(ts)

# simulation loop
for i in range(ts):
    # use the optimal policy from the problem def to generate actions
    a = policy(b)

    r = pomdp.reward(s, a)
    rave[i] = r

    trans_dist = pomdp.transition(s, a, dist = trans_dist)
    s = pomdp.sample_state(trans_dist)

    obs_dist = pomdp.observation(s, a, dist = obs_dist)
    o = pomdp.sample_observation(obs_dist)

    b.update(pomdp, a, o)

    # these are numpy arrays
    bold = b.old_belief()
    bnew = b.new_belief()

    exp = (bold, a, r, bnew)

    print exp

print "Average reward: ", rave.mean()
