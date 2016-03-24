import os
import numpy as np
from copy import deepcopy
import pickle
from timeit import default_timer as timer

from chimp.utils.policies import RandomPolicy
from chimp.utils.policies import DQNPolicy

class DQNAgent(object):

    def __init__(self, learner, memory, simulator, settings, dqn_policy=None, rollout_policy=None):

        """
        The learning agent is responsible for communicating and moving
        data between the three modules: Learner, Simulator, Memory
        Inputs:
        - learner: containes the neural network and the optimizer to train it
        - memory: expereince replay memory that can be minibatch sampled
        - simulator: simulates the environemnt
        - settings: hyper parameters for training
        - rollout_policy: rollout policy, random by default
        """

        self.learner = learner
        self.memory = memory
        self.simulator = simulator # for populating the experience replay
        self.evaluator = deepcopy(simulator) # for evaluation

        self.dqn_policy = dqn_policy
        if dqn_policy is None:
            self.dqn_policy = DQNPolicy(learner)

        self.rollout_policy = rollout_policy
        if rollout_policy is None:
            self.rollout_policy = RandomPolicy(simulator.n_actions)

        self.set_params(settings)

        self.n_epochs = self.iterations / float(memory.memory_size)
        self.iteration = []
        self.loss = []
        self.q_ave = []
        self.eval_iteration = []
        self.r_eval = []
        self.r_per_episode_eval = [] 

    def policy(self, obs, epsilon):
        """
        e-greedy policy with customazible rollout
        """
        if self.random_state.rand() < epsilon:
            return self.rollout_policy.action(obs) 
        else:
            return self.dqn_policy.action(obs)


    def save(self,obj,name):
        ''' function to save a file as pickle '''
        # TODO: don't you need to close the I/O stream?
        pickle.dump(obj, open(name, "wb"))

    def load(self,name):
        ''' function to load a pickle file '''
        return pickle.load(open(name, "rb"))


    def train(self, verbose=True):
        """
        Trains the network
        """
        learner = self.learner
        memory = self.memory
        simulator = self.simulator

        if self.viz:
            simulator.init_viz_display()

        # run initial exploration and populate the experience replay
        self.populate_memory(self.initial_exploration) 

        # add initial observation to observatin history
        iobs = simulator.get_screenshot().copy()
        self.initial_obs(iobs)

        iteration = 0 # keeps track of all training iterations, ignores evaluation
        run_time = 0.0
        start_time = timer() # mark the global beginning of training
        last_print = timer()

        while iteration < self.iterations: # for the set number of iterations

            # perform a single simulator step
            self.step()
            # minibatch update for DQN
            if iteration % self.learn_freq == 0:
                loss, qvals = self.batch_update()
                self.iteration.append(iteration)
                self.loss.append(loss)
                self.q_ave.append(np.mean(qvals))

            if iteration % self.print_every == 0 and verbose:
                print "Iteration: %d, Loss: %.3f, Average Q-Values: %.2f, Time since print: %.2f, Total runtime: %.2f, epsilon: %.2f" % (iteration, loss, np.mean(qvals), timer() - last_print, timer() - start_time, self.epsilon)
                last_print = timer()
            
            if iteration % self.save_every == 0:
                # saving the net, the training history, and the learner itself
                learner.save_net('%s/net_%d.p' % (self.save_dir,int(iteration)))
                np.savetxt('%s/training_history.csv' % self.save_dir, np.asarray([self.iteration, self.loss, self.q_ave]).T)

            if iteration % self.eval_every == 0: # evaluation
                sim_r, sim_r_per_episode, sim_time = self.simulate(self.eval_iterations, self.eval_epsilon)
                self.eval_iteration.append(iteration)
                self.r_eval.append(sim_r)
                self.r_per_episode_eval.append(sim_r_per_episode)

                if verbose:
                    print "Evaluation, total reward: %.2f, Reward per episode: %.2f" % (sim_r, sim_r_per_episode)

                np.savetxt('%s/evaluation_history.csv' % self.save_dir, np.asarray([self.eval_iteration, self.r_eval, self.r_per_episode_eval]).T)

            if iteration % self.target_net_update == 0:
                learner.copy_net_to_target_net()

            self.epsilon -= self.epsilon_decay
            self.epsilon = 0.1 if self.epsilon < 0.1 else self.epsilon

            iteration += 1

        memory.close()

        learner.save_net('%s/net_%d.p' % (self.save_dir,int(iteration)))
        np.savetxt('%s/training_history.csv' % self.save_dir, np.asarray([self.iteration, self.loss, self.q_ave]).T)

        run_time = timer() - start_time
        print('Overall training + evaluation time: '+ str(run_time))



    def step(self):
        """
        Performs a single step with the DQN and updates the replay memory
        """
        loss = 0.0

        simulator = self.simulator

        obs = simulator.get_screenshot().copy()
        a = self.policy((self.ohist, self.ahist), self.epsilon)
        simulator.act(a)
        r = simulator.reward()

        term = False
        obsp = None
        if simulator.episode_over():
            term = True
            obsp = obs.copy()
            simulator.reset_episode()
            iobs = simulator.get_screenshot().copy()
            self.empty_history()
            self.initial_obs(iobs)
        else:
            obsp = simulator.get_screenshot().copy()
            self.update_history(obsp, a)

        if self.viz: # move the image to the screen / shut down the game if display is closed
            simulator.refresh_viz_display()

        self.memory.store_tuple(obs, a, r, obsp, term)


    def batch_update(self):
        """
        Performs a mini-batch update on the DQN
        """
        ohist, ahist, rhist, ophist, term = self.memory.minibatch()
        # take the last as our action and reward
        a = ahist[:,-1] 
        r = rhist[:,-1]
        t = term[:,-1]
        oahist = None
        # TODO: this indexing is a hack to deal with single sample history
        # Using the first history entry of the minibatch (there is only one) - could do this with reshape as well
        if self.ahist_size == 0 or self.ohist_size == 1:
            oahist = (ohist[:,0], None)
            oaphist = (ophist[:,0], None)
        else:
            oahist = (ohist, ahist[:self.ahist_size])
            oaphist = (ophist, ahist[1:self.ahist_size])
        loss, qvals = self.learner.update(oahist, a, r, oaphist, t)
        return loss, qvals


    #################################################################
    ################### Some Utility Functions ######################
    #################################################################

    def simulate(self, nsteps, epsilon, viz=False):
        """
        Simulates the DQN policy
        """
        simulator = self.evaluator # use a different simulator to prevent breaks 
        simulator.reset_episode()
        # add initial observation to observation history
        iobs = simulator.get_screenshot().copy()
        self.initial_eval_obs(iobs)

        if self.viz:
            simulator.init_viz_display()

        rtot = 0.0
        r_per_episode = 0.0
        episode_count = 0
        start_sim = timer()
        for i in xrange(nsteps):
            # generate reward and step the simulator
            ohist, ahist = self.eval_ohist, self.eval_ahist
            a = self.policy((ohist, ahist), epsilon)

            simulator.act(a)
            r = simulator.reward()
            rtot += r
            if simulator.episode_over():
                simulator.reset_episode()
                iobs = simulator.get_screenshot().copy()
                self.empty_eval_history()
                self.initial_eval_obs(iobs)
                episode_count += 1
                r_per_episode = rtot
            else:
                obsp = simulator.get_screenshot().copy()
                self.update_eval_history(obsp, a)
 
            if self.viz: # move the image to the screen / shut down the game if display is closed
                simulator.refresh_viz_display()

        print r_per_episode, episode_count
        if episode_count > 0:
            r_per_episode /= episode_count
        else:
            r_per_episode = rtot
        runtime = timer() - start_sim
        return rtot, r_per_episode, runtime


    def populate_memory(self, nsamples):
        # TODO: do we need to copy obs and obsp?
        memory = self.memory
        simulator = self.simulator

        simulator.reset_episode()
        for i in xrange(nsamples):
            # generate o, a, r, o' tuples
            obs = simulator.get_screenshot().copy() 
            a = self.rollout_policy.action(obs)
            simulator.act(a)
            r = simulator.reward()
            obsp = simulator.get_screenshot().copy() 
            term = False
            if simulator.episode_over():
                term = True
                simulator.reset_episode() # reset
            # store the tuples
            memory.store_tuple(obs, a, r, obsp, term)
        simulator.reset_episode()


    def plot_loss(self):
        try:
            from matplotlib import pyplot
        except ImportError:
            "Can not plot loss, matplotlib required"
        pyplot.plot(self.loss[1:])
        pyplot.xlabel("Iteration")
        pyplot.ylabel("Loss")
        pyplot.show()

    def plot_eval_reward(self):
        try:
            from matplotlib import pyplot
        except ImportError:
            "Can not plot loss, matplotlib required"
        pyplot.plot(self.eval_every * np.arange(len(self.r_eval)), self.r_eval)
        pyplot.xlabel("Reward")
        pyplot.ylabel("Loss")
        pyplot.show()



    def set_params(self, settings):
        # set up the setting parameters
        self.random_state = np.random.RandomState(settings.get('seed_agent', None)) # change to a new random seed

        self.batch_size = settings.get('batch_size', 32) 
        self.n_frames = settings.get('n_frames', 1)
        self.iterations = settings.get('iterations', 1000000)

        self.epsilon = settings.get('epsilon', 1.0) # exploration
        self.epsilon_decay = settings.get('epsilon_decay', 0.00001) # decay in 
        self.eval_epsilon = settings.get('eval_epsilon', 0.0) # exploration during evaluation
        self.initial_exploration = settings.get('initial_exploration', 10000) # of iterations during initial exploration

        self.viz = settings.get('viz', False) # whether to visualize the state/observation, False when not supported by simulator

        self.eval_iterations = settings.get('eval_iterations', 500)
        self.eval_every = settings.get('eval_every', 5000)
        self.print_every = settings.get('print_every', 5000)
        self.save_dir = settings.get('save_dir', '.')
        self.save_every = settings.get('save_every', 5000)
        
        self.learn_freq = settings.get('learn_freq', 1) # how frequently to do back prop on a minibatch
        self.target_net_update = settings.get('target_net_update', 5000)

        self.ohist_size, self.ahist_size, self.rhist_size = settings.get('history_sizes', (1,0,0))
        self.ahist_size = 1 if self.ahist_size == 0 else self.ahist_size
        self.ohist_size = 1 if self.ohist_size == 0 else self.ohist_size

        self.ohist = np.zeros((self.ohist_size,) + self.simulator.model_dims, dtype=np.float32)
        self.ahist = np.zeros(self.ahist_size, dtype=np.int32)
        self.rev_ohist = np.zeros((self.ohist_size,) + self.simulator.model_dims, dtype=np.float32)
        self.rev_ahist = np.zeros(self.ahist_size, dtype=np.int32)

        self.eval_ohist = np.zeros((self.ohist_size,) + self.simulator.model_dims, dtype=np.float32)
        self.eval_ahist = np.zeros(self.ahist_size, dtype=np.int32)
        self.rev_eval_ohist = np.zeros((self.ohist_size,) + self.simulator.model_dims, dtype=np.float32)
        self.rev_eval_ahist = np.zeros(self.ahist_size, dtype=np.int32)

    #################################################################
    ################# History utility functions #####################
    #################################################################
    """
    These are messy, and could be optimized
    """

    def update_history(self, obs, a):
        # roll the histories forward and replace the first entry
        # keep a reversed history so we can easily roll though it
        self.rev_ohist = np.roll(self.rev_ohist, 1, axis=0)
        self.rev_ahist = np.roll(self.rev_ahist, 1, axis=0)
        self.rev_ahist[0] = a
        self.rev_ohist[0] = obs

        # reverse to get history in [s0, s1, s2,...,sn] format
        self.ohist = np.flipud(self.rev_ohist)
        self.ahist = np.flipud(self.rev_ahist)


    def update_eval_history(self, obs, a):
        # roll the histories forward and replace the first entry
        self.rev_eval_ohist = np.roll(self.rev_eval_ohist, 1, axis=0)
        self.rev_eval_ahist = np.roll(self.rev_eval_ahist, 1, axis=0)
        self.rev_eval_ahist[0] = a
        self.rev_eval_ohist[0] = obs

        self.eval_ohist = np.flipud(self.rev_eval_ohist)
        self.eval_ahist = np.flipud(self.rev_eval_ahist)

    def initial_obs(self, obs):
        self.rev_ohist[0] = obs
        self.ohist[-1] = obs

    def initial_eval_obs(self, obs):
        self.rev_eval_ohist[0] = obs
        self.eval_ohist[-1] = obs


    def empty_history(self):
        self.ohist.fill(self.memory._emptyfloat)
        self.ahist.fill(self.memory._emptyint)
        self.rev_ohist.fill(self.memory._emptyfloat)
        self.rev_ahist.fill(self.memory._emptyint)

    def empty_eval_history(self):
        self.eval_ohist.fill(self.memory._emptyfloat)
        self.eval_ahist.fill(self.memory._emptyint)
        self.rev_eval_ohist.fill(self.memory._emptyfloat)
        self.rev_eval_ahist.fill(self.memory._emptyint)

