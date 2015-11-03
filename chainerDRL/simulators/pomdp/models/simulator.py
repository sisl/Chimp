import numpy as np

#################################################################
# This file implements a pomdp simulator using the interface
# defined in the README
#################################################################

class POMDPSimulator():

    # constructor
    def __init__(self, pomdp, robs=False):
        self.pomdp = pomdp
        self.current_state = pomdp.initial_state()
        self.current_action = None
        self.current_observation = None
        self.current_belief = pomdp.initial_belief()
        self.current_reward = 0.0

        self.robs = robs # returns observation or belief

        self.tdist = pomdp.create_transition_distribution()
        self.odist = pomdp.create_observation_distribution()

        self.n_actions = self.pomdp.n_actions()
        self.n_states = self.pomdp.n_states()

        if not robs:
            self.model_dims = self.pomdp.belief_shape
        else:
            self.model_dims = self.pomdp.observation_shape

    # returns the number of actions
    #def n_actions(self):
    #    return self.pomdp.n_actions()

    # returns the number of states
    #def n_states(self):
    #    return self.pomdp.n_states()

    # progress single step in simulation
    def act(self, ai):
        pomdp = self.pomdp
        s = self.current_state
        b = self.current_belief
        tdist = self.tdist
        odist = self.odist

        a = pomdp.index2action(ai)

        r = pomdp.reward(s, a)
        
        tdist = pomdp.transition(s, a, dist = tdist)
        s = pomdp.sample_state(tdist)

        odist = pomdp.observation(s, a, dist = odist)
        o = pomdp.sample_observation(odist)

        b.update(pomdp, a, o)

        self.current_reward = r
        self.current_state = s

    # returns the current simulator belief
    def get_screenshot(self):
        if self.robs:
            return self.current_observation
        else:
            return self.current_belief.new_belief()

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

