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
import pygame
import matplotlib.pyplot as plt

from timeit import default_timer as timer
# from time import clock


class DQNAgent(object):

    def __init__(self, settings):

        # general settings
        self.batch_size = settings['batch_size']
        self.n_frames = settings['n_frames']
        self.epsilon = settings['epsilon']
        self.epsilon_decay = settings['epsilon_decay']
        self.eval_epsilon = settings['eval_epsilon']
        self.viz = settings['viz']

        # unit of measurement - number of transitions
        self.initial_exploration = settings['initial_exploration']
        self.iterations = settings['iterations']
        self.eval_iterations = settings['eval_iterations']
        self.eval_every = settings['eval_every']
        self.print_every = settings['print_every']
        self.save_dir = settings['save_dir']
        self.save_every = settings['save_every']

    # helper function to save net (or any object)
    def save(self,obj,name):
        pickle.dump(obj, open(name, "wb"))

    # helper function to load net (or any object)
    def load(self,name):
        return pickle.load(open(name, "rb"))

    # helper function to load a pre-trained net
    def load_net(self,learner,name):
        net = self.load(name)
        learner.load_net(net)

    # masking function for learner policy - for e-greedy simulator action selection
    def policy(self, learner, simulator, s, epsilon = 0):
        if np.random.rand() < epsilon:
            opt_a = np.random.randint(0, simulator.n_actions)
        else:
            opt_a = learner.policy(s)
        return opt_a

    # get state
    def get_state(self, simulator):
        # get screenshot
        self.frame = simulator.get_screenshot().reshape(simulator.model_dims)

        # add screenshot to the state to the 
        ind = [i+1 for i in xrange(self.n_frames - 1)]
        tmp = []
        for i in ind:
            tmp.append(self.state[0][i])
        tmp.append(self.frame)
        self.state = np.asanyarray(tmp, dtype=np.float32).reshape(1, self.n_frames, 
            simulator.model_dims[0], simulator.model_dims[1])
        return self.state.copy()

    # launch training process
    def train(self, learner, memory, simulator):

        # create "nets" directory to save training output
        if not os.path.exists(self.save_dir):
            print("Creating '%s' directory to store training results..." % self.save_dir)
            os.makedirs(self.save_dir)

        if learner.clip_reward:
            print("Rewards are clipped in training, but not in evaluation")

        # initialize visualization display
        if self.viz:
            simulator.init_viz_display()

        print("Running initial exploration for " + str(self.initial_exploration) + " screen transitions...")

        # setting initial state that will last trough one episode
        self.state = np.zeros((1, self.n_frames, simulator.model_dims[0], simulator.model_dims[1]), dtype=np.float32)
        self.s0 = self.get_state(simulator)

        self.iteration = 0

        total_reward = 0.0
        total_loss = 0.0
        total_qval_avg = 0.0
        episode_counter = 0
        local_counter = 0
        end_evaluation = 0.0

        global_start = timer()

        while self.iteration < self.iterations:

            if self.iteration == self.initial_exploration:
                print("Initial exploration over. Learning begins...")

            if self.iteration % self.print_every == 0:
                print("Episode: " + str(episode_counter) + ", " + 
                    "Transitions experienced: " +  str(self.iteration))

            if self.iteration < self.initial_exploration:

                reward, loss, qval_avg, episode = self.perceive(learner, memory, simulator, True, True)

                episode_counter += episode
                end_exploration = timer()

            else: # exploration ended

                reward, loss, qval_avg, episode = self.perceive(learner, memory, simulator, True, False)

                total_reward += reward
                total_loss += loss
                total_qval_avg += qval_avg
                
                episode_counter += episode
                local_counter += 1

                if self.iteration % self.save_every == 0:
                    print('Saving %snet_%d.p' % (self.save_dir,int(self.iteration)))
                    self.save(learner.net,'%snet_%d.p' % (self.save_dir,int(self.iteration)))

                if self.iteration % self.eval_every == 0:

                    end_training = timer()

                    total_reward /= local_counter
                    total_loss /= local_counter
                    total_qval_avg /= local_counter
                    total_train_time = end_training - max(end_exploration,end_evaluation)

                    learner.train_rewards.append(total_reward)
                    learner.train_losses.append(total_loss)
                    learner.train_qval_avgs.append(total_qval_avg)
                    learner.train_times.append(total_train_time)

                    total_reward = 0
                    total_loss = 0
                    total_qval_avg = 0
                    local_counter = 0

                    simulator.reset_episode()
                    self.state = np.zeros((1, self.n_frames, simulator.model_dims[0], simulator.model_dims[1]), dtype=np.float32)
                    self.s0 = self.get_state(simulator)

                    for i in xrange(self.eval_iterations):

                        reward, loss, qval_avg, episode = self.perceive(learner,memory,simulator,False, False)
                        
                        total_reward += reward
                        total_loss += loss
                        total_qval_avg += qval_avg
                        local_counter += 1

                    end_evaluation = timer()

                    total_reward /= local_counter
                    total_loss /= local_counter
                    total_qval_avg /= local_counter
                    total_eval_time = (end_evaluation - end_training)

                    print('episode %d, epoch %.2f, iteration %d, avg. loss %.3f, avg. reward %.2f, avg. Q-value %.2f, training time %.2f, evaluation time %.2f, train epsilon %.5f' % (
                        episode_counter,self.iteration/float(memory.memory_size),
                        self.iteration,total_loss,total_reward,
                        total_qval_avg,total_train_time,total_eval_time,self.epsilon))

                    learner.val_rewards.append(total_reward)
                    learner.val_losses.append(total_loss)
                    learner.val_qval_avgs.append(total_qval_avg)
                    learner.val_times.append(total_eval_time)

                    total_reward = 0
                    total_loss = 0
                    total_qval_avg = 0
                    local_counter = 0

                    simulator.reset_episode()
                    self.state = np.zeros((1, self.n_frames, simulator.model_dims[0], simulator.model_dims[1]), dtype=np.float32)
                    self.s0 = self.get_state(simulator)

            self.iteration += 1

        global_end = timer()
        learner.overall_time = global_end - global_start
        print('Overall training + evaluation time: '+ str(learner.overall_time))
        self.save(learner,'./%s/learner_final.p' % self.save_dir)


    # one iteration in training or evaluation mode
    def perceive(self, learner, memory, simulator, train=True, initial_exporation=False, custom_policy=None):

        loss = 0
        qval_avg = 0
        episode = 0

        # INTERACTING WITH THE SIMULATOR AND STORING THE EXPERIENCE
        # getting observation and forming the state
        if train and initial_exporation:
            self.a = self.policy(learner, simulator, self.s0, epsilon = 1)
        elif train and not initial_exporation:
            self.a = self.policy(learner, simulator, self.s0, epsilon = self.epsilon)
        elif not train and custom_policy:
            self.a = custom_policy(self.s0.squeeze())
        elif not train and not custom_policy:
            self.a = self.policy(learner, simulator, self.s0, epsilon = self.eval_epsilon)

        simulator.act(self.a);
        self.reward = float(simulator.reward());
        self.s1 = self.get_state(simulator)

        # move the image to the screen / shut down the game if display is closed
        if self.viz:
            simulator.refresh_viz_display()

        if train:

            memory.store_tuple(self.s0,self.a,self.reward,self.s1,simulator.episode_over())

            if not initial_exporation:
                
                # sample a minibatch
                self.s0_mb,self.a_mb,self.reward_mb,self.s1_mb,self.end_mb = memory.minibatch(self.batch_size)

                # run an update self.iteration on one mini-batch
                batch_approx_q_all, batch_loss, batch_qval_avg = learner.gradUpdate(
                    self.s0_mb,self.a_mb,self.reward_mb,self.s1_mb,self.end_mb)

                loss = batch_loss
                qval_avg = batch_qval_avg

                # copy a net every fixed number of steps
                if self.iteration % learner.target_net_update == 0:
                    learner.net_to_target_net()

                # discount the randomization constant
                self.epsilon -= self.epsilon_decay
                if self.epsilon < 0.1:
                    self.epsilon = 0.1

        elif not train and not custom_policy: # DQN evaluation

            # calculate loss and other metrics on the observed tuple
            batch_approx_q_all, batch_loss, batch_qval_avg = learner.forwardLoss(
                self.s0,self.a,self.reward,self.s1,simulator.episode_over())

            loss = batch_loss
            qval_avg = batch_qval_avg

        self.s0 = self.s1 # s1 now is s0 during next turn

        if simulator.episode_over():
            episode += 1

            simulator.reset_episode()
            # setting initial state that will last trough one episode
            self.state = np.zeros((1, self.n_frames, simulator.model_dims[0], simulator.model_dims[1]), dtype=np.float32)
            self.s0 = self.get_state(simulator)

        return self.reward, loss, qval_avg, episode


    # function to evaluate own and custom policies
    def evaluate(self, learner, simulator, eval_iterations=5000, custom_policy=None):

        if self.viz:
            simulator.init_viz_display()

        self.state = np.zeros((1, self.n_frames, simulator.model_dims[0], simulator.model_dims[1]), dtype=np.float32)
        self.s0 = self.get_state(simulator)

        total_reward = 0
        total_loss = 0
        total_qval_avg = 0

        episode_counter = 0

        start = timer()

        for i in xrange(eval_iterations):

            reward, loss, qval_avg, episode = self.perceive(learner, None, simulator, False, False, custom_policy)
            
            total_reward += reward
            total_loss += loss
            total_qval_avg += qval_avg

            episode_counter += episode

        end = timer()

        total_reward /= eval_iterations
        total_loss /= eval_iterations
        total_qval_avg /= eval_iterations

        simulator.reset_episode()

        return total_reward, total_loss, total_qval_avg, episode_counter,  end - start

