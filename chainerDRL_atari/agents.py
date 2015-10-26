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

class Agent(object):

    def __init__(self, settings):

        # general settings
        self.batch_size = settings['batch_size']
        self.n_frames = settings['n_frames']
        self.epsilon = settings['epsilon']
        self.viz = settings['viz']
        
        # unit of measurement - number of episodes
        self.n_episodes = settings['n_episodes']
        self.eval_every = settings['eval_every']
        self.eval_episodes = settings['eval_episodes']

        # unit of measurement - number of transitions (memory.counter)
        self.initial_exploration = settings['initial_exploration']
        self.print_every = settings['print_every']
        self.save_dir = settings['save_dir']
        self.save_every = settings['save_every']


    # helper function to save net (or any object)
    def save(self,obj,name):
        pickle.dump(obj, open(name, "wb"))

    # helper function to load net (or any object)
    def load(self,name):
        return pickle.load(open(name, "rb"))

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
        self.frame = simulator.get_screenshot()

        # add screenshot to the state to the 
        ind = [i+1 for i in range(self.n_frames - 1)]
        tmp = []
        for i in ind:
            tmp.append(self.state[0][i])
        tmp.append(self.frame)
        self.state = np.asanyarray(tmp, dtype=np.float32).reshape(1, self.n_frames, 
            simulator.screen_dims_new[0], simulator.screen_dims_new[1])
        return self.state.copy()

    # launch training process
    def train(self, learner, memory, simulator):

        # create "nets" directory to save training output
        if not os.path.exists(self.save_dir):
            print("Creating '%s' directory to store training results..." % self.save_dir)
            os.makedirs(self.save_dir)

        if learner.clip_reward:
            print("Rewards are clipped in training, but not in evaluation")

        memory.episode_counter = 0
        local_counter = 0
        total_transition = 0
        total_reward = 0
        total_loss = 0
        total_qval_avg = 0

        # initialize visualization display
        if self.viz:
            simulator.init_viz_display()

        print("Running initial exploration for " + str(self.initial_exploration) + " screen transitions...")

        while memory.episode_counter < self.n_episodes:

            episode_reward, loss, qval_avg, transition_counter = self.episode(learner,memory,simulator,True)
            total_reward += episode_reward
            total_loss += loss
            total_qval_avg += qval_avg
            total_transition += transition_counter
            memory.episode_counter += 1
            local_counter += 1

            # print("Episode #" + str(memory.episode_counter))

            if local_counter >= self.eval_every and memory.counter >= self.initial_exploration:

                total_reward /= local_counter
                total_loss /= local_counter
                total_qval_avg /= local_counter
                total_transition /= local_counter

                learner.train_rewards.append(total_reward)
                learner.train_losses.append(total_loss)
                learner.train_qval_avgs.append(total_qval_avg)

                total_reward = 0
                total_loss = 0
                total_qval_avg = 0
                local_counter = 0

                self.save(learner.net,'./%s/net_%d.p' % (self.save_dir,int(memory.counter/memory.memory_size)))

                for i in range(self.eval_episodes):

                    episode_reward, loss, qval_avg, transition_counter = self.episode(learner,memory,simulator,False)
                    total_reward += episode_reward
                    total_loss += loss
                    total_qval_avg += qval_avg
                    total_transition += transition_counter
                    local_counter += 1

                total_reward /= local_counter
                total_loss /= local_counter
                total_qval_avg /= local_counter
                total_transition /= local_counter

                print('episode %d, epoch %.2f, iteration %d, loss %.3f, reward %.2f, avg. Q-value %.2f, epsilon %.5f, avg. # of transitions %d' % (
                    memory.episode_counter,memory.counter/float(memory.memory_size),memory.counter,total_loss,total_reward,total_qval_avg,self.epsilon, total_transition))

                learner.val_rewards.append(total_reward)
                learner.val_losses.append(total_loss)
                learner.val_qval_avgs.append(total_qval_avg)

                total_reward = 0
                total_loss = 0
                total_qval_avg = 0
                total_transition = 0
                local_counter = 0

        self.save(learner,'./%s/agent_final.p' % self.save_dir)


    # play one game, in training or evaluation mode
    def episode(self, learner, memory, simulator, train=True):

        episode_reward = 0
        approx_q_all = 0 
        loss = 0 
        qval_avg = 0

        transition_counter = 0

        # setting initial state that will last trough one episode
        self.state = np.zeros((1, self.n_frames, simulator.screen_dims_new[0], simulator.screen_dims_new[1]), dtype=np.float32)
        self.s0 = self.get_state(simulator)

        while not simulator.episode_over():

            transition_counter += 1

            if memory.counter == self.initial_exploration:
                print("Initial exploration over. Learning begins...")

            if memory.counter % self.print_every == 0:
                print("Episode: " + str(memory.episode_counter) + ", " + 
                    "Transitions experienced: " +  str(memory.counter))

            # INTERACTING WITH THE SIMULATOR AND STORING THE EXPERIENCE
            # getting observation and forming the state
            if train and memory.counter < self.initial_exploration:
                self.a = np.random.randint(simulator.n_actions)
            elif train and memory.counter >= self.initial_exploration:
                self.a = self.policy(learner, simulator, self.s0, epsilon = self.epsilon)
            else:
                self.a = self.policy(learner, simulator, self.s0, epsilon = 0.01)

            self.reward = float(simulator.act(self.a));
            episode_reward += self.reward;

            self.s1 = self.get_state(simulator)

            # move the image to the screen / shut down the game if display is closed
            if self.viz:
                simulator.refresh_viz_display()

            if train:

                # storing only tuples observed during training
                memory.store_tuple(self.s0,self.a,self.reward,self.s1,simulator.episode_over())

                if memory.counter >= self.initial_exploration:
                    
                    # sample a minibatch
                    self.s0_l,self.a_l,self.reward_l,self.s1_l,self.end_l = memory.minibatch(self.batch_size)

                    # run an update iteration on one mini-batch
                    batch_approx_q_all, batch_loss, batch_qval_avg = learner.gradUpdate(
                        self.s0_l,self.a_l,self.reward_l,self.s1_l,self.end_l)

                    loss += batch_loss
                    qval_avg += batch_qval_avg

                    # copy a net every fixed number of steps
                    if memory.counter % learner.target_net_update == 0:
                        learner.target_net = deepcopy(learner.net)

                    # discount the randomization constant
                    self.epsilon -= 1.0/10**6
                    if self.epsilon < 0.1:
                        self.epsilon = 0.1

            else: # evaluation

                # calculate loss and other metrics on the observed tuple
                batch_approx_q_all, batch_loss, batch_qval_avg = learner.forwardLoss(
                    self.s0,self.a,self.reward,self.s1,simulator.episode_over())
                loss += batch_loss
                qval_avg += batch_qval_avg

            self.s0 = self.s1 # s1 now is s0 during next turn


        # restart episode
        simulator.reset_episode()

        loss /= transition_counter
        qval_avg /= transition_counter

        return episode_reward, loss, qval_avg, transition_counter
