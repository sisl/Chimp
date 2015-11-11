import numpy as np

#################################################################
# This file implements a pomdp simulator using the interface
# defined in the README
#################################################################

class MOMDPSimulator():

    # constructor
    def __init__(self, pomdp, robs=False):
        self.pomdp = pomdp
        self.current_xstate = pomdp.initial_fully_obs_state() # fully observable vars
        self.current_ystate = pomdp.initial_partially_obs_state() # partially observable vars
        self.current_action = None
        self.current_observation = None
        self.current_belief = pomdp.initial_belief()
        self.current_reward = 0.0

        self.robs = robs # returns observation or belief

        self.tdx = pomdp.create_fully_obs_transition_distribution() # fully obs distribution
        self.tdy = pomdp.create_partially_obs_transition_distribution() # partially obs distribution

        self.od = pomdp.create_observation_distribution()

        self.n_actions = self.pomdp.n_actions()
        self.n_xstates = self.pomdp.n_states()
        self.n_ystates = self.pomdp.n_states()

        self.x_b_state = np.zeros(pomdp.belief_shape[0] + pomdp.fully_obs_shape[0])

        if not robs:
            self.model_dims = (pomdp.belief_shape[0]+pomdp.fully_obs_shape[0], 1)
        else:
            self.model_dims = pomdp.observation_shape

    # progress single step in simulation
    def act(self, ai):
        pomdp = self.pomdp
        x = self.current_xstate
        y = self.current_ystate
        b = self.current_belief
        tdx = self.tdx
        tdy = self.tdy
        odist = self.odist

        a = pomdp.index2action(ai)

        r = pomdp.reward(x, y, a)
        
        tdx = pomdp.fully_obs_transition(x, y, a, tdx)
        tdy = pomdp.partially_obs_transition(x, y, a, tdy) 
        x = pomdp.sample_fully_obs_state(tdx)
        y = pomdp.sample_partially_obs_state(tdy)

        odist = pomdp.observation(x, y, a, odist)
        o = pomdp.sample_observation(odist)

        b.update(pomdp, x, a, o)

        self.current_reward = r
        self.current_xstate = x
        self.current_ystate = y
        self.current_observation = o

    # returns the current simulator belief
    def get_screenshot(self):
        if self.robs:
            return self.current_observation
        else:
            sc = self.x_b_state
            b = self.current_belief.new_belief()
            nx = self.n_xstates
            ny = self.n_ystates
            for i in xrange(nx):
                sc[i] = self.current_xstate[i]
            for i in xrange(ny):
                sc[i+nx] = b[i]
            return sc

    # returns the current reward
    def reward(self):
        return self.current_reward

    # check if reached terminal states
    def episode_over(self):
        return self.pomdp.isterminal(self.current_state)

    def reset_episode(self):
        pomdp = self.pomdp
        self.current_state = pomdp.initial_state()
        self.current_belief = pomdp.initial_belief()

