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

        self.n_actions = pomdp.n_actions()
        self.n_xstates = pomdp.n_xstates()
        self.n_ystates = pomdp.n_ystates()

        self.x_b_state = np.zeros(pomdp.xdims + pomdp.n_ystates())

        self.x_o_state = np.zeros(pomdp.xdims + pomdp.odims)

        if not robs:
            self.model_dims = (pomdp.xdims+pomdp.n_ystates(), 1)
        else:
            self.model_dims = (pomdp.xdims+pomdp.odims, 1) 

    # progress single step in simulation
    def act(self, ai):
        pomdp = self.pomdp
        x = self.current_xstate
        y = self.current_ystate
        b = self.current_belief
        tdx = self.tdx
        tdy = self.tdy
        od = self.od

        a = pomdp.index2action(ai)

        r = pomdp.reward(x, y, a)
        
        tdx = pomdp.fully_obs_transition(x, y, a, tdx)
        tdy = pomdp.partially_obs_transition(x, y, a, tdy) 
        x = pomdp.sample_fully_obs_state(tdx)
        y = pomdp.sample_partially_obs_state(tdy)

        od = pomdp.observation(x, y, a, od)
        o = pomdp.sample_observation(od)

        b.update(pomdp, x, a, o)

        self.current_reward = r
        self.current_xstate = x
        self.current_ystate = y
        self.current_observation = o

    # returns the current simulator belief
    def get_screenshot(self):
        if self.robs:
            sc = self.x_o_state
            nx = self.pomdp.xdims
            no = self.pomdp.odims
            for i in xrange(nx):
                sc[i] = self.current_xstate[i]
            sc[nx] = self.current_observation
            return sc 
        else:
            sc = self.x_b_state
            b = self.current_belief.new_belief()
            nx = self.pomdp.xdims
            nb = self.n_ystates
            for i in xrange(nx):
                sc[i] = self.current_xstate[i]
            for i in xrange(nb):
                sc[i+nx] = b[i]
            return sc

    # returns the current reward
    def reward(self):
        return self.current_reward

    # check if reached terminal states
    def episode_over(self):
        return self.pomdp.isterminal(self.current_xstate, self.current_ystate)

    def reset_episode(self):
        pomdp = self.pomdp
        self.current_xstate = pomdp.initial_fully_obs_state()
        self.current_ystate = pomdp.initial_partially_obs_state()
        self.current_belief = pomdp.initial_belief()

