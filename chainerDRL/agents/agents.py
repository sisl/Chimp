'''
Agent framework.
It is set up with the settings,
manipulates learner, replay memory and the simulator to do training and evaluation, 
record, and save results.
'''

import os
import numpy as np
from copy import deepcopy
import pickle
import matplotlib.pyplot as plt
from timeit import default_timer as timer

class DQNAgent(object):

    def __init__(self, settings):

        self.random_state = np.random.RandomState(settings['seed_agent'])
        self.batch_size = settings['batch_size']
        self.n_frames = settings['n_frames']
        self.epsilon = settings['epsilon'] # exploration
        self.epsilon_decay = settings['epsilon_decay'] # RMSprop parameter
        self.eval_epsilon = settings['eval_epsilon'] # exploration during evaluation
        self.viz = settings['viz'] # whether to visualize the state/observation, False when not supported by simulator
        self.initial_exploration = settings['initial_exploration'] # of iterations during initial exploration
        self.iterations = settings['iterations']
        self.eval_iterations = settings['eval_iterations']
        self.eval_every = settings['eval_every']
        self.print_every = settings['print_every']
        self.save_dir = settings['save_dir']
        self.save_every = settings['save_every']
        self.learn_freq = settings['learn_freq']

    def save(self,obj,name):
        ''' function to save a file as pickle '''
        pickle.dump(obj, open(name, "wb"))

    def load(self,name):
        ''' function to load a pickle file '''
        return pickle.load(open(name, "rb"))

    def policy(self, learner, simulator, s, ahist, epsilon = 0):
        ''' e-greedy policy '''
        if self.random_state.rand() < epsilon:
            opt_a = self.random_state.randint(0, simulator.n_actions)
        else:
            opt_a = learner.policy(s, ahist)
        return opt_a

    def get_state(self, simulator):
        ''' update current state with a new observation '''

        self.frame = simulator.get_screenshot().reshape(simulator.model_dims)

        # drop the oldest frame, add the newest one
        ind = [i+1 for i in xrange(self.n_frames - 1)]
        tmp = []
        for i in ind:
            tmp.append(self.state[0][i])
        tmp.append(self.frame)

        # recreate the array
        self.state = np.asanyarray(tmp, dtype=np.float32).reshape(1, self.n_frames, 
            simulator.model_dims[0], simulator.model_dims[1])

        return self.state.copy()

    def get_ahist(self, a):
        ''' update current action history with a new action '''

        # drop the oldest action, add the newest one
        ind = [i+1 for i in xrange(self.n_frames - 1)]
        tmp = []
        for i in ind:
            tmp.append(self.ahist[0][i])
        tmp.append(a)

        # recreate the array
        self.ahist = np.asanyarray(tmp, dtype=np.float32).reshape(1, self.n_frames)

        return self.ahist.copy()

    def reset_episode(self, simulator, initial=False):
        ''' resets an episode and sets up the necessary initial state and action history arrays '''

        if not initial: # if the simulator has been launched, reset it; otherwise only set up arrays
            simulator.reset_episode()

        self.ahist = -1*np.ones((1, self.n_frames), dtype=np.float32) # used by get_ahist() function
        self.ahist0 = -1*np.ones((1, self.n_frames), dtype=np.float32) # initialize action history

        self.state = -1*np.ones((1, self.n_frames, simulator.model_dims[0], simulator.model_dims[1]), dtype=np.float32) # used by get_state() function
        self.s0 = self.get_state(simulator) # get s0 state in a new episode - before running perceive()

    #@profile
    def train(self, learner, memory, simulator):
        ''' wrapper around the training process '''

        if not os.path.exists(self.save_dir):
            print("Creating '%s' directory to store training results..." % self.save_dir)
            os.makedirs(self.save_dir)

        if learner.clip_reward or learner.reward_rescale:
            print("Rewards are clipped/rescaled in training, but not in evaluation")

        if self.viz:
            simulator.init_viz_display()

        print("Running initial exploration for " + str(self.initial_exploration) + " screen transitions...")

        self.reset_episode(simulator, initial=True) # initialize the necessary arrays
        self.iteration = 0 # keeps track of all training iterations, ignores evaluation
        self.episode = 0 # keeps track of all training episodes, ignores evaluation

        # local trackers of reward, loss, and avg. Q-values during each training/validation run
        total_reward = 0.0
        total_loss = 0.0
        total_qval_avg = 0.0

        local_episode_counter = 0 # local episode counter during each training/validation run
        local_iteration_counter = 0 # local iteration counter during each training/validation run
        
        global_start = timer() # mark the global beginning of training
        end_evaluation = 0.0 # used to track when the last evaluation has ended

        while self.iteration < self.iterations: # for the set number of iterations

            self.iteration += 1

            if self.iteration % self.print_every == 0:
                print("Episode: " + str(self.episode) + ", " + "Transitions experienced: " +  str(self.iteration))

            if self.iteration <= self.initial_exploration: # during initial exploration
                reward, loss, qval_avg, episode = self.perceive(learner, memory, simulator, train=True, initial_exporation=True)
                self.episode += episode

            if self.iteration == self.initial_exploration:
                # when initial exploration has ended, mark the end of the episode and reset it
                print("Initial exploration over. Learning begins...")
                self.reset_episode(simulator)
                self.episode += 1
                end_exploration = timer()

            if self.iteration >= self.initial_exploration: 
                # from the moment initial exploration has ended 
                # greater or equal sign is here to make sure we save the initial net

                # 'episode' is a 1/0 flag on whether we have reset the episode during this iteration
                reward, loss, qval_avg, episode = self.perceive(learner, memory, simulator, train=True, initial_exporation=False)
                
                self.episode += episode # global episode counter

                total_reward += reward
                total_loss += loss
                total_qval_avg += qval_avg
                local_episode_counter += episode
                local_iteration_counter += 1

                if self.iteration % self.save_every == 0:

                    global_end = timer()
                    learner.overall_time = global_end - global_start

                    print('Saving %s/net_%d.p' % (self.save_dir,int(self.iteration)))
                    print('Overall training + evaluation time since the start of training: '+ str(learner.overall_time))

                    # saving the net, the training history, and the learner itself
                    learner.save_net('%s/net_%d.p' % (self.save_dir,int(self.iteration)))
                    learner.save_training_history(self.save_dir)
                    self.save(learner,'%s/learner_final.p' % self.save_dir)

                if self.iteration % self.eval_every == 0: # evaluation

                    end_training = timer()

                    # count an episode, marking the end of the last unfinished training episode
                    self.reset_episode(simulator)
                    local_episode_counter += 1
                    self.episode += 1

                    # take averages for the loss and Q-values, compute training time
                    total_loss /= local_iteration_counter
                    total_qval_avg /= local_iteration_counter
                    total_train_time = end_training - max(end_exploration,end_evaluation)

                    # keep track of training history
                    learner.train_rewards.append(total_reward)
                    learner.train_losses.append(total_loss)
                    learner.train_qval_avgs.append(total_qval_avg)
                    learner.train_episodes.append(local_episode_counter)
                    learner.train_times.append(total_train_time)

                    # reset local tracking variables
                    total_reward = 0
                    total_loss = 0
                    total_qval_avg = 0
                    local_episode_counter = 0
                    local_iteration_counter = 0

                    for i in xrange(self.eval_iterations):

                        reward, loss, qval_avg, episode = self.perceive(learner, memory, simulator, train=False, initial_exporation=False)
                        
                        total_reward += reward
                        total_loss += loss
                        total_qval_avg += qval_avg
                        local_episode_counter += episode
                        local_iteration_counter += 1

                    end_evaluation = timer()

                    self.reset_episode(simulator)
                    local_episode_counter += 1
                    # note, we do not add the evaluation episodes to the global episode counter

                    total_loss /= local_iteration_counter
                    total_qval_avg /= local_iteration_counter
                    total_eval_time = (end_evaluation - end_training)

                    print('episode %d, epoch %.2f, iteration %d, avg. loss %.3f, eval. cumulative reward %.2f, avg. Q-value %.2f, training time %.2f, evaluation time %.2f, train epsilon %.5f' % (
                        local_episode_counter,self.iteration/float(memory.memory_size),
                        self.iteration,total_loss,total_reward,total_qval_avg,total_train_time,total_eval_time,self.epsilon))

                    learner.val_rewards.append(total_reward)
                    learner.val_losses.append(total_loss)
                    learner.val_qval_avgs.append(total_qval_avg)
                    learner.val_episodes.append(local_episode_counter)
                    learner.val_times.append(total_eval_time)

                    total_reward = 0
                    total_loss = 0
                    total_qval_avg = 0
                    local_episode_counter = 0
                    local_iteration_counter = 0

        global_end = timer()
        learner.overall_time = global_end - global_start
        print('Overall training + evaluation time: '+ str(learner.overall_time))
        print('Saving results...')
        self.save(learner,'%s/learner_final.p' % self.save_dir)
        print('Done')


<<<<<<< HEAD
    #@profile
=======
>>>>>>> ffc8433a2bf0819b64511c281371b3a9ab18844e
    def perceive(self, learner, memory, simulator, train=True, initial_exporation=False, custom_policy=None):
        ''' 
        one iteration in training or evaluation mode - assumes s0 and ahist0 have already been retrieved 
        returns reward, loss, avg. Q-value, episode flag if simulator has been reset, i.e., has ended
        '''

        loss = 0
        qval_avg = 0
        episode = 0

        # get an action depending on a situation
        if train and initial_exporation:
            self.a = self.policy(learner, simulator, self.s0, self.ahist0, epsilon = 1)
        elif train and not initial_exporation:
            self.a = self.policy(learner, simulator, self.s0, self.ahist0, epsilon = self.epsilon)
        elif not train and custom_policy: # evaluation with a custom policy
            self.a = custom_policy(self.s0[-1].squeeze())
        elif not train and not custom_policy: # evaluation with a learned policy
            self.a = self.policy(learner, simulator, self.s0, self.ahist0, epsilon = self.eval_epsilon)

        simulator.act(self.a);
        self.reward = np.zeros((1),dtype=np.float32) + float(simulator.reward())
        self.s1 = self.get_state(simulator)
        self.ahist1 = self.get_ahist(self.a)

        if self.viz: # move the image to the screen / shut down the game if display is closed
            simulator.refresh_viz_display()

        if train: # in training mode - during or after initial exploration

            memory.store_tuple(self.s0,self.ahist0,self.a,self.reward,self.s1,self.ahist1,simulator.episode_over())

            if not initial_exporation and self.iteration % self.learn_freq == 0: # completed initial exploration, and time to update
                
                self.s0_mb,self.ahist0_mb,self.a_mb,self.reward_mb,self.s1_mb,self.ahist1_mb,self.end_mb = memory.minibatch(self.batch_size)

                batch_approx_q_all, batch_loss, batch_qval_avg = learner.gradUpdate(
                    self.s0_mb,self.ahist0_mb,self.a_mb,self.reward_mb,self.s1_mb,self.ahist1_mb,self.end_mb)

                loss = batch_loss
                qval_avg = batch_qval_avg

                if self.iteration % learner.target_net_update == 0:
                    learner.copy_net_to_target_net()

                self.epsilon -= self.epsilon_decay
                if self.epsilon < 0.1:
                    self.epsilon = 0.1

        elif not train and not custom_policy: # evaluation of the learned policy

            batch_approx_q_all, batch_loss, batch_qval_avg = learner.forwardLoss(   # evaluation on one observation
                self.s0,self.ahist0,self.a,self.reward,self.s1,self.ahist1,simulator.episode_over())

            loss = batch_loss
            qval_avg = batch_qval_avg

        self.s0 = self.s1   # s1 now is s0 during the next turn
        self.ahist0 = self.ahist1

        if simulator.episode_over():
            episode += 1
            self.reset_episode(simulator)

        return self.reward, loss, qval_avg, episode


    def evaluate(self, learner, simulator, eval_iterations=5000, custom_policy=None):
        '''
        evaluate policy, once all training has finished: 
        returns cumulative reward, avg. loss, avg. Q-value, # of episodes, 
        evaluation time, paths, action histories and reward histories
        '''

        if self.viz:
            simulator.init_viz_display()

        self.reset_episode(simulator)

        total_reward = 0
        total_loss = 0
        total_qval_avg = 0
        local_episode_counter = 0
        episode = 0

        start = timer()

        # lists to record movement through space
        paths = []
        path = []
        reward_histories = []
        rewards = []
        action_histories = []
        actions = []

        #counter = 0

        for i in xrange(eval_iterations):

            if episode > 0:
                paths.append(path)
                path = []
                reward_histories.append(rewards)
                rewards = []
                action_histories.append(actions)
                actions = []
                #counter = 0

            path.append(self.s0)

            reward, loss, qval_avg, episode = self.perceive(learner, None, simulator, False, False, custom_policy)
            
            actions.append(self.a)
            rewards.append(float(self.reward)) # * (learner.discount**counter)
            total_reward += reward
            total_loss += loss
            total_qval_avg += qval_avg
            local_episode_counter += episode

            #counter += 1

        paths.append(path)
        reward_histories.append(rewards)
        action_histories.append(actions)

        end = timer()

        total_loss /= eval_iterations
        total_qval_avg /= eval_iterations

        self.reset_episode(simulator)
        local_episode_counter += 1

        return total_reward, total_loss, total_qval_avg, local_episode_counter, end - start, paths, action_histories, reward_histories

