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
        self.current_reward = 0.0

        self.model_dims = model.state_shape

        self.n_actions = model.n_actions

    def act(self, action):
        """
        Transitions the model forward by moving
        """
        mdp = self.model

        self.current_reward = mdp.reward(self.current_state, action)
        self.next_state = mdp.transition(self.current_state, action)

    def reward(self):
        return self.current_reward

    def get_screenshot(self):
        return self.current_state 

    def episode_over(self):
        return self.model.isterminal(self.current_state)

    def reset_episode(self):
        self.current_state = self.model.initial_state()
        self.current_reward = 0.0

    def n_actions(self):
        return self.model.n_actions
         
    def simulate(self, nsteps, policy, verbose=False):
        mdp = self.model

        # re-initialize the model
        self.reset_episode()

        input_state = np.zeros((1,)+self.model_dims, dtype=np.float32)

        x_trace = np.zeros(nsteps)
        v_trace = np.zeros(nsteps)

        rtot = 0.0
        # run the simulation
        for i in xrange(nsteps):
            r = self.reward()
            #state = self.observe()
            state = self.get_screenshot()
            x_trace[i] = state[0]
            v_trace[i] = state[1]
            rtot += r
            if self.episode_over():
                if verbose:
                    print "Terminal reward: ", r
                    print "Reached terminal state: ", state
                break
            input_state[0,:] = state
            a = policy.action((input_state, None))
            self.act(a)
            if verbose:
                print "Timestep: ", i
                print "Reward: ", r
                print "State: ", self.current_state
                print "Action: ", a
                print "Next State: ", self.next_state
        return rtot, x_trace, v_trace

