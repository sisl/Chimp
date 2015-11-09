from ale_python_interface import ALEInterface
import pygame

import numpy as np
import scipy.misc as spm


class AtariSimulator(object):

    def __init__(self, settings):

        '''Initiate Arcade Learning Environment (ALE) using Python interface
        https://github.com/bbitmaster/ale_python_interface/wiki

        - Set number of frames to be skipped, random seed, ROM and title for display.
        - Retrieve a set of legal actions and their number.
        - Retrieve dimensions of the original screen (width/height), and set the dimensions
        of the cropped screen, together with the padding used to crop the screen rectangle.
        - Set dimensions of the pygame display that will show visualization of the simulation.
        (May be cropped --- showing what the learner sees, or not --- showing full Atari screen)
        - Allocate memory for generated grayscale screenshots. Accepts dims in (height/width) format
        '''

        self.ale = ALEInterface()
        self.ale.setInt("frame_skip",settings["frame_skip"])
        self.ale.setInt("random_seed",settings["seed_simulator"])
        self.ale.loadROM(settings["rom_dir"] + '/' + settings["rom"])

        self.title = "ALE Simulator: " + str(settings["rom"])
        self.actions = self.ale.getLegalActionSet()
        self.n_actions = self.actions.size

        self.screen_dims = self.ale.getScreenDims()
        self.model_dims = settings['model_dims']
        self.pad = settings['pad']

        print("Original screen width/height: " + str(self.screen_dims[0]) + "/" + str(self.screen_dims[1]))
        print("Cropped screen width/height: " + str(self.model_dims[0]) + "/" + str(self.model_dims[1]))

        self.viz_cropped = settings['viz_cropped']
        if self.viz_cropped:
            self.display_dims = (int(self.model_dims[0]*2), int(self.model_dims[1]*2))
        else:
            self.display_dims = (int(self.screen_dims[0]*2), int(self.screen_dims[1]*2))

        # preallocate an array to accept ALE screen data (height/width) !
        self.screen_data = np.empty((self.screen_dims[1],self.screen_dims[0]),dtype=np.uint8)


    def get_screenshot(self):
        '''returns a cropped snapshot of the simulator
        - store grayscale values in a preallocated array
        - cut out a square from the rectangle, using provided padding value
        - downsample to the desired size and transpose from (height/width) to (width/height)
        '''

        self.ale.getScreenGrayscale(self.screen_data)
        self.tmp = self.screen_data[(self.screen_dims[1]-self.screen_dims[0]-self.pad):(self.screen_dims[1]-self.pad),:]
        self.frame = spm.imresize(self.tmp,self.model_dims[::-1], interp='nearest').T

        return self.frame


    def act(self,action_index):
        '''function to transition the simulator from s to s' using provided action
        the action that is provided is in form of an index
        simulator deals with translating the index into an actual action'''

        self.last_reward = self.ale.act(self.actions[action_index])


    def reward(self):
        '''return reward - has to be called after the "act" function'''

        return self.last_reward


    def episode_over(self):
        '''return a boolean indicator on whether the game is still running'''

        return self.ale.game_over()
        

    def reset_episode(self):
        '''reset the game that ended'''

        self.ale.reset_game()


    def init_viz_display(self):
        '''initialize display that will show visualization'''

        pygame.init()
        self.screen = pygame.display.set_mode(self.display_dims)
        if self.title:
            pygame.display.set_caption(self.title)


    def refresh_viz_display(self):
        '''if display is shut down, shut the game down
        else move the current simulator's frame (cropped or not cropped) into the pygame display,
        after expanding it 2x along x and y dimensions'''

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                exit()

        if self.viz_cropped:
            self.surface = pygame.surfarray.make_surface(self.frame) # has already been transposed
        else:
            self.surface = pygame.surfarray.make_surface(self.screen_data.T)

        self.screen.blit(pygame.transform.scale2x(self.surface),(0,0))
        pygame.display.flip()

