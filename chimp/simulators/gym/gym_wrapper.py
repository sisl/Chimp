class GymWrapper():

    def __init__(self, env):

        self.env = env
        self.last_reward = 0.0
        self.current_state = None
        self.terminal_flag = False
        self.n_actions = env.action_space.n
        self.model_dims = env.observation_space.shape

    def act(self, action):
        """
        Transitions to the next state and computes the reward
        """
        state, reward, done, info = self.env.step(action)
        self.last_reward = reward
        self.current_state = state
        self.terminal_flag = done
    def reward(self):
        return self.last_reward

    def get_screenshot(self):
        return self.current_state

    def episode_over(self):
        """
        Checks if the car reached the top of the mountain
        """
        return self.terminal_flag

    def reset_episode(self):
        self.current_state = self.env.reset()

    def simulate(self, nsteps):
        """
        Runs a simulation using the provided DQN policy for nsteps
        """

        self.reset_episode()

        rtot = 0.0
        # run the simulation
        for i in xrange(nsteps):
            self.env.render()
            state = self.get_screenshot()
            a = self.env.action_space.sample()
            self.act(a)
            r = self.reward()
            rtot += r
            if self.episode_over():
                break
        return rtot
        
