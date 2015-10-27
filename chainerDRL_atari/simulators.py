''' Implements generic definition of the environment simulator.
Here we use Arcade Learning Environment to for Atari games simulation.

This file would have to be rewritten, depending on the simulator in use.
All simulators should provide the following functions:
__init__, get_screenshot, act, game_over, reset_game
'''

import numpy as np
import scipy.misc as spm

from ale_python_interface import ALEInterface
import matplotlib.pyplot as plt

import pygame

class Atari(object):

    def __init__(self, settings):

        # initialize arcade learning environment
        # using python interface to Arcade Learning Environment
        # https://github.com/bbitmaster/ale_python_interface/wiki
        self.ale = ALEInterface()

        # set # of frames to skip, random seed and the ROM to load
        self.ale.setInt("frame_skip",settings["frame_skip"])
        self.ale.setInt("random_seed",settings["seed"])
        self.ale.loadROM(settings["rom"])

        # has to be defined for visualization purposes
        self.title = "ALE Simulator: " + str(settings["rom"])

        # the vector of possible actions and their count
        self.actions = self.ale.getLegalActionSet()
        self.n_actions = self.actions.size

        # the original dimensions of the observation (width/height)
        self.screen_dims = self.ale.getScreenDims()
        print("Original screen width/height: " + str(self.screen_dims[0]) + "/" + str(self.screen_dims[1]))

        # visualize the just the part of image that an agent sees vs. the full display 
        self.viz_cropped = settings['viz_cropped']

        # cropped dimensions of the observation
        self.screen_dims_new = settings['screen_dims_new']
        print("Modified screen width/height: " + str(self.screen_dims_new[0]) + "/" + str(self.screen_dims_new[1]))

        # size of the visualization display
        if self.viz_cropped:
            self.display_dims = (int(self.screen_dims_new[0]*2), int(self.screen_dims_new[1]*2))
        else:
            self.display_dims = (int(self.screen_dims[0]*2), int(self.screen_dims[1]*2))

        # padding during cropping
        self.pad = settings['pad']

        # allocating memory for generated screenshots - needs to be of a particular type
        # !!!! accepts dims in (height/width format)
        self.screen_data = np.empty((self.screen_dims[1],self.screen_dims[0]),dtype=np.uint8)


    # get cropped screenshot
    def get_screenshot(self):

        # load screen image into self.screen_data
        self.ale.getScreenGrayscale(self.screen_data)

        # cut out a square and downsize it
        # frame = (spm.imresize(self.screen_data,(110, 84),interp='nearest'))[110-84-8:110-8,:]
        self.tmp = self.screen_data[(self.screen_dims[1]-self.screen_dims[0]-self.pad):(self.screen_dims[1]-self.pad),:]
        
        # Scaling + going from (height/width) to (width/height) - may be dropped for square images
        self.frame = spm.imresize(self.tmp,self.screen_dims_new[::-1], interp='nearest').T  
        
        return self.frame

    # function to transition the simulator from s to s' using provided action
    # !!! returns the observed reward
    # the action that is provided is in form of an index
    # simulator deals with translating the index into an actual action
    def act(self,action_index):
        return self.ale.act(self.actions[action_index])

    # function that return a bool indicator as to whether the game is still running
    def episode_over(self):
        return self.ale.game_over()

    # function to reset the game that ended
    def reset_episode(self):
        self.ale.reset_game()

    # initialize display that will show visualization
    def init_viz_display(self):
        pygame.init()
        self.screen = pygame.display.set_mode(self.display_dims)
        if self.title:
            pygame.display.set_caption(self.title)

    # refresh display: 
    # if display shut down - shut down the game
    # else move current simulator's frame into display
    def refresh_viz_display(self):
        
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                exit()

        if self.viz_cropped:
            self.surface = pygame.surfarray.make_surface(self.frame)
        else:
            self.surface = pygame.surfarray.make_surface(self.screen_data.T)

        self.screen.blit(pygame.transform.scale2x(self.surface),(0,0))
        pygame.display.flip()
