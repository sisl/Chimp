'''
POMDP simulator
'''

import numpy as np

class POMDPSimulator():

    # constructor
    def __init__(self, pomdp, settings, robs=False):
        self.pomdp = pomdp
        self.current_state = pomdp.initial_state()
        self.current_action = None
        self.current_observation = None
        self.current_belief = pomdp.initial_belief()
        self.current_reward = 0.0

        self.robs = robs # returns observation or belief
        self.rseed = settings["seed"] # random seed

        self.tdist = pomdp.create_transition_distribution()
        self.odist = pomdp.create_observation_distribution()

        self.n_actions = self.pomdp.n_actions()
        self.n_states = self.pomdp.n_states()
        self.model_dims = settings['model_dims']

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
