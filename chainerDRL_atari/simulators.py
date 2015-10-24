''' Implements generic definition of the environment simulator.
Here we use Arcade Learning Environment to for Atari games simulation.

This file would have to be rewritten, depending on the simulator in use.
All simulators should provide the following functions:
__init__, get_screenshot, act, game_over, reset_game

The details of the functions are covered below.
'''

import numpy as np
import scipy.misc as spm

from ale_python_interface import ALEInterface
import matplotlib.pyplot as plt

class Atari(object):

    def __init__(self, settings):

        # initialize arcade learning environment
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

        # the original dimensions of the observation
        self.screen_dims = self.ale.getScreenDims()
        print("Original screen width/height: " + str(self.screen_dims[0]) + "/" + str(self.screen_dims[1]))

        # cropped dimensions of the observation
        self.screen_dims_new = settings['screen_dims_new']
        print("Modified screen width/height: " + str(self.screen_dims_new[0]) + "/" + str(self.screen_dims_new[1]))

        # padding during cropping
        self.pad = settings['pad']

        # allocating memory for generated screenshots - needs to be of a particular type
        self.screen_data = np.empty((self.screen_dims[1],self.screen_dims[0]),dtype=np.uint8)


    # get cropped screenshot
    def get_screenshot(self):

        # load screen image into self.screen_data
        self.ale.getScreenGrayscale(self.screen_data)

        # cut out a square and downsize it
        # frame = (spm.imresize(self.screen_data,(110, 84),interp='nearest'))[110-84-8:110-8,:]
        tmp = self.screen_data[(self.screen_dims[1]-self.screen_dims[0]-self.pad):(self.screen_dims[1]-self.pad),:]
        frame = spm.imresize(tmp,self.screen_dims_new[::-1], interp='nearest')  # Scaling
        
        return frame.T

    # function to transition the simulator from s to s' using provided action
    # returns the observed reward
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
