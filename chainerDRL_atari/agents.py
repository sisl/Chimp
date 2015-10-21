import os
import numpy as np
import scipy.misc as spm
import chainer
import chainer.functions as F
from chainer import optimizers
from copy import deepcopy
import math
import random

import pygame

from ale_python_interface import ALEInterface

import pickle # used to save the nets

class Agent(object):

    def __init__(self, settings):

        self.batch_size = settings['batch_size']
        self.n_episodes = settings['n_episodes']
        self.n_frames = settings['n_frames']
        self.actions = settings['actions']
        self.n_actions = self.actions.size
        self.epsilon = settings['epsilon']
        self.initial_exploration = settings['initial_exploration']
        self.eval_every = settings['eval_every']
        self.eval_episodes = settings['eval_episodes']

        self.print_every = settings['print_every']
        self.save_dir = settings['save_dir']
        self.save_every = settings['save_every']

        self.screen_dims = settings['screen_dims']
        self.screen_dims_new = settings['screen_dims_new']
        self.display_dims = settings['display_dims']
        self.pad = settings['pad']
        self.screen_data = np.zeros(self.screen_dims[0]*self.screen_dims[1],dtype=np.uint32)
        self.state = np.zeros((1,self.n_frames, self.screen_dims_new[0], self.screen_dims_new[1]), dtype=np.float32)

        self.viz = settings['viz']

    # helper function to save net (or any object)
    def save(self,obj,name):
        pickle.dump(obj, open(name, "wb"))

    # helper function to load net (or any object)
    def load(self,name):
        return pickle.load(open(name, "rb"))

    # masking function for learner policy - for e-greedy simulator action selection
    def policy(self, learner, s, epsilon = 0):
        if np.random.rand() < epsilon:
            opt_a = np.random.randint(0, self.n_actions)
        else:
            opt_a = learner.policy(s)
        return opt_a

    # get cropped screen image
    def get_screen(self, ale):
        ale.getScreenRGB(self.screen_data)
        if self.viz:
            self.screen.blit(pygame.transform.scale2x(self.game_surface),(0,0))
            pygame.display.flip()
        tmp = np.bitwise_and(self.screen_data.reshape([self.screen_dims[0], self.screen_dims[1]]), 0b0001111)  # Get Intensity from the observation
        # frame = (spm.imresize(tmp, (84, 110)))[:, 110-84-8:110-8]  # Scaling
        frame = (spm.imresize(tmp[:, self.screen_dims[1]-self.screen_dims[0]-self.pad:self.screen_dims[1]-self.pad], 
            (self.screen_dims_new[0], self.screen_dims_new[1])))  # Scaling
        # frame = (spm.imresize(tmp[:, 210-160-15:210-15], (84, 84)))
        return frame

    # get state
    def update_state(self, ale):
        frame = self.get_screen(ale)
        ind = [i+1 for i in range(self.n_frames - 1)]
        temp = []
        for i in ind:
            temp.append(self.state[0][i])
        temp.append(frame)
        self.state = np.asanyarray(temp, dtype=np.float32).reshape(1, self.n_frames, self.screen_dims_new[0], self.screen_dims_new[1])
        return self.state.copy()

    # launch training process
    def train(self, learner, memory, ale):

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

        print("Running initial exploration for " + str(self.initial_exploration) + " screen transitions...")

        if self.viz:
            pygame.init()
            self.screen = pygame.display.set_mode((self.display_dims[0],self.display_dims[1]))
            pygame.display.set_caption("Arcade Learning Environment Random Agent Display")

            self.game_surface = pygame.Surface((self.screen_dims[0],self.screen_dims[1]))
            self.screen_data = np.frombuffer(self.game_surface.get_buffer(),dtype=np.uint32)

        while memory.episode_counter < self.n_episodes:

            episode_reward, loss, qval_avg, transition_counter = self.episode(learner,memory,ale,True)
            total_reward += episode_reward
            total_loss += loss
            total_qval_avg += qval_avg
            total_transition += transition_counter
            memory.episode_counter += 1
            local_counter += 1

            print("Episode #" + str(memory.episode_counter))

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
                    episode_reward, loss, qval_avg, transition_counter = self.episode(learner,memory,ale,False)
                    total_reward += episode_reward
                    total_loss += loss
                    total_qval_avg += qval_avg
                    total_transition += transition_counter
                    local_counter += 1

                total_reward /= local_counter
                total_loss /= local_counter
                total_qval_avg /= local_counter
                total_transition /= local_counter

                print('epoch %.2f, iteration %d, loss %.3f, reward %.2f, avg. Q-value %.2f, epsilon %.5f, avg. # of transitions %d' % (
                    memory.counter/float(memory.memory_size),memory.counter,total_loss,total_reward,total_qval_avg,self.epsilon, total_transition))

                learner.val_rewards.append(total_reward)
                learner.val_losses.append(total_loss)
                learner.val_qval_avgs.append(total_qval_avg)

                total_reward = 0
                total_loss = 0
                total_qval_avg = 0
                total_transition = 0
                local_counter = 0

        self.save(learner,'./%s/DQN_final.p' % self.save_dir)


    # play one game, in training or evaluation mode
    def episode(self, learner, memory, ale, train=True):

        episode_reward = 0
        approx_q_all = 0 
        loss = 0 
        qval_avg = 0

        transition_counter = 0

        self.state = np.zeros((1, self.n_frames, self.screen_dims_new[0], self.screen_dims_new[1]), dtype=np.float32)
        self.s0 = self.update_state(ale)

        while not ale.game_over():

            transition_counter += 1

            if memory.counter == self.initial_exploration:
                print("Initial exploration over. Learning begins...")

            if memory.counter % 500 == 0:
                print(str(memory.counter) + " transitions experienced")

            # INTERACTING WITH THE SIMULATOR AND STORING THE EXPERIENCE
            # getting observation and forming the state
            if train and memory.counter < self.initial_exploration:
                self.a = np.random.randint(self.n_actions)
            elif train and memory.counter >= self.initial_exploration:
                self.a = self.policy(learner, self.s0, epsilon = self.epsilon)
            else:
                self.a = self.policy(learner, self.s0, epsilon = 0)

            self.reward = float(ale.act(self.actions[self.a]));
            episode_reward += self.reward;

            self.s1 = self.update_state(ale)

            if train:

                # storing only tuples observed during training
                memory.storeTuple(self.s0,self.a,self.reward,self.s1,ale.game_over())

                if memory.counter >= self.initial_exploration:
                    
                    self.s0_l,self.a_l,self.reward_l,self.s1_l,self.end_l = memory.minibatch(self.batch_size)
                    batch_approx_q_all, batch_loss, batch_qval_avg = learner.gradUpdate(self.s0_l,self.a_l,self.reward_l,self.s1_l,self.end_l) # run an update iteration on one mini-batch
                    loss += batch_loss
                    qval_avg += batch_qval_avg

                    if memory.counter % learner.target_net_update == 0:
                        learner.target_net = deepcopy(learner.net)

                    self.epsilon -= 1.0/10**6
                    if self.epsilon < 0.1:
                        self.epsilon = 0.1

            else:

                batch_approx_q_all, batch_loss, batch_qval_avg = learner.forwardLoss(self.s0,self.a,self.reward,self.s1,ale.game_over())
                loss += batch_loss
                qval_avg += batch_qval_avg

            self.s0 = self.s1.copy() # s1 now is s0 during next turn

            if self.viz:
                for event in pygame.event.get():
                    if event.type == pygame.QUIT:
                        exit()

        ale.reset_game()

        loss /= transition_counter
        qval_avg /= transition_counter

        return episode_reward, loss, qval_avg, transition_counter
