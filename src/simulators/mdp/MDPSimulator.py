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
        self.current_reward = 0.0

        self.model_dims = model.state_shape


    def act(self, action):
        """
        Transitions the model forward by moving
        """
        mdp = self.model
        s = self.current_state

        self.current_reward = mdp.reward(s, a)
        self.current_state = mdp.transition(s, action)

    def reward(self):
        return self.current_reward

    def observe(self):
        return self.current_state 

    def episode_over(self, state):
        return self.model.isterminal(state)

    def reset_episode(self):
        self.current_state = self.model.initial_state()
        self.current_reward = 0.0
         

    def simulate(self, nsteps, controller, verbose=False):
        mdp = self.model

        # re-initialize the model
        model.initialize()

        rtot = 0.0
        # run the simulation
        for i in xrange(nsteps):
            r = self.reward()
            state = self.observe()
            rtot += r
            if self.episode_over(state):
                if verbose:
                    print "Reached terminal state: ", s
                break
            a = controller.action(state)
            act.act(a)
            if verbose:
                print "Timestep: ", i
                print "Reward: ", r
                print "State: ", s
                print "Action: ", a
        return rtot

