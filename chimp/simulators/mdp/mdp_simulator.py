import numpy as np

#################################################################
# Implements the simulator class for MDPs 
#################################################################

class MDPSimulator():

    def __init__(self, model):
        """
        Implements the multi-agent simulator:
        This serves as a wrapper for MDP problem types 
        """

        self.model = model # problem instance

        # initalize 
        self.current_state = model.initial_state()
        self.last_action = 0
        self.last_reward = 0.0

        self.model_dims = model.state_shape

        self.n_actions = model.n_actions

    def act(self, action):
        """
        Transitions the model forward by moving
        """
        mdp = self.model

        self.last_reward = mdp.reward(self.current_state, action)
        self.current_state = mdp.transition(self.current_state, action)
        if self.episode_over():
            self.last_reward += mdp.reward(self.current_state, action)

    def reward(self):
        return self.last_reward

    def get_screenshot(self):
        return self.current_state 

    def episode_over(self):
        return self.model.isterminal(self.current_state)

    def reset_episode(self):
        self.current_state = self.model.initial_state()
        self.last_reward = 0.0

    def n_actions(self):
        return self.model.n_actions
