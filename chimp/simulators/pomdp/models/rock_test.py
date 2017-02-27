from rock_sample import RockSamplePOMDP

pomdp = RockSamplePOMDP()

x = pomdp.initial_fully_obs_state()
y = pomdp.initial_partially_obs_state()

tdx = pomdp.create_fully_obs_transition_distribution()
tdy = pomdp.create_partially_obs_transition_distribution()
od = pomdp.create_observation_distribution()

for a in range(pomdp.n_actions()):
    print "Action ", x, y, a
    tdx = pomdp.fully_obs_transition(x, y, a, tdx)
    tdy = pomdp.partially_obs_transition(x, y, a, tdy)
    od = pomdp.observation(x, y, a, od)
    x = pomdp.sample_fully_obs_state(tdx)
    y = pomdp.sample_partially_obs_state(tdy)
    o = pomdp.sample_observation(od)
    print "Observation ", x, y, o

b = pomdp.initial_belief()

x = (1,1)
a = 6

od = pomdp.observation(x, y, a, od)
o = pomdp.sample_observation(od)

b.update(pomdp, x, a, o)

