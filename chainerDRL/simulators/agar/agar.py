#!/usr/bin/env python

from __future__ import (absolute_import, division,
                        print_function, unicode_literals)

from selenium import webdriver
from selenium.webdriver.common.action_chains import ActionChains
from selenium.webdriver.common.by import By
import json
import random
import time
import scipy.misc as spm
import numpy as np
import pygame

class AgarIODriver(object):

    CANVAS_ID = 'canvas'
    PLAY_BUTTON_CLASS = "btn-play-guest"
    CONTINUE_BUTTON_ID = "statsContinue"

    WINDOW_SIZE_X = 200
    WINDOW_SIZE_Y = 300

    PROXY = '127.0.0.1:8080'

    GET_CANVAS_PIXELS_JAVASCRIPT = """
        var canvas = document.getElementById('canvas');
        var canvasWidth = canvas.width;
        var canvasHeight = canvas.height;
        var ctx = canvas.getContext('2d');
        var imageData = ctx.getImageData(0, 0, canvasWidth, canvasHeight);
        return '[' + String(Array.prototype.slice.call(imageData.data)) + ']';
        """

    GET_SCORE_JAVASCRIPT = "return document.getElementById('score').innerHTML"

    EPISODE_OVER_JAVASRIPT = """
        var continueBttn = document.getElementById("statsContinue");
        var bttnIsVisible = (continueBttn.offsetParent != null);
        return bttnIsVisible;
        """

    RESET_EPISODE_JAVASCRIPT = "setNick('SISLlaboratory');"

    def __init__(self, settings):

        self.driver = self._get_chrome_webdriver()
        self.driver.set_window_size(self.WINDOW_SIZE_X, self.WINDOW_SIZE_Y)
        self.driver.get('http://agar.io')
        self.canvas_element = self.driver.find_element(By.ID, self.CANVAS_ID)
        self.canvas_width = self.canvas_element.size['width']
        self.canvas_height = self.canvas_element.size['height']

        self.play_button = self.driver.find_element_by_class_name(self.PLAY_BUTTON_CLASS)
        self.continue_button = self.driver.find_element_by_id(self.CONTINUE_BUTTON_ID)

        self.screen_dims = (self.canvas_width,self.canvas_height)
        self.model_dims = settings['model_dims']
        self.pad = settings['pad']

        self.viz_cropped = settings['viz_cropped']
        if self.viz_cropped:
            self.display_dims = (int(self.model_dims[0]*2), int(self.model_dims[1]*2))
        else:
            self.display_dims = (int(self.screen_dims[0]*2), int(self.screen_dims[1]*2))

        self.title = 'agar.io'

        print("%s x %s" % (self.canvas_width, self.canvas_height))

        self.score = 1024
        self.actions = [
            (0,0),
            (0,self.screen_dims[1]-1),
            (0,self.screen_dims[1]/2),
            (self.screen_dims[1]-1,0),
            (self.screen_dims[1]/2,0),
            (self.screen_dims[1]/2,self.screen_dims[1]/2),
            (self.screen_dims[1]-1,self.screen_dims[1]/2),
            (self.screen_dims[1]/2,self.screen_dims[1]-1),
            (self.screen_dims[0]-1,self.screen_dims[1]-1)
            ]
        self.n_actions = len(self.actions)

        self.driver.execute_script(self.RESET_EPISODE_JAVASCRIPT)

    def get_canvas_pixels(self):
        return self.driver.execute_script(self.GET_CANVAS_PIXELS_JAVASCRIPT)

    def get_screenshot(self):
        self.tmp = np.array(json.loads(self.get_canvas_pixels())).reshape((self.screen_dims[1],self.screen_dims[0],4))
        self.tmp_screen = self.tmp[:,:,0]*0.299 + self.tmp[:,:,1]*0.587 + self.tmp[:,:,2]*0.114
        self.frame = spm.imresize(self.tmp_screen,self.model_dims, interp='nearest').T
        return self.frame

    def get_score(self):
        return float(self.driver.execute_script(self.GET_SCORE_JAVASCRIPT))

    def act(self,action_index):
        action = self.actions[action_index]
        ActionChains(self.driver).move_to_element_with_offset(self.canvas_element, action[0], action[1]).perform()
        self.last_reward = self.get_score() - self.score
        # print(self.last_reward)
        self.score += self.last_reward

    def reward(self):
        return self.last_reward

    def episode_over(self):
        return self.driver.execute_script(self.EPISODE_OVER_JAVASRIPT)

    def reset_episode(self):
        if self.episode_over:
            ActionChains(self.driver).click(self.continue_button).perform()
        self.driver.execute_script(self.RESET_EPISODE_JAVASCRIPT)

    def play_game(self):
        self.reset_episode()
        while True:
            loop_start = time.time()
            score, pixels, game_over = self._get_game_state()
            loop_duration = time.time() - loop_start
            print("Score %s, Game over: %s (Duration: %f)" % (score, game_over, loop_duration))

            if game_over:
                ActionChains(self.driver).click(self.continue_button).perform()
                return

            x_offset = random.choice([0, self.canvas_width-1])
            y_offset = random.choice([0, self.canvas_height-1])
            ActionChains(self.driver).move_to_element_with_offset(self.canvas_element, x_offset, y_offset).perform()
            
    def _get_game_state(self):
        pixelJson = self.get_canvas_pixels()
        pixels = json.loads(pixelJson)
        score = self.get_score()
        game_over = self.episode_over()
        return score, pixels, game_over
    
    def init_viz_display(self):
        pygame.init()
        self.screen = pygame.display.set_mode(self.display_dims)
        if self.title:
            pygame.display.set_caption(self.title)

    def refresh_viz_display(self):
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                exit()

        if self.viz_cropped:
            self.surface = pygame.surfarray.make_surface(self.frame) # has already been transposed
        else:
            self.surface = pygame.surfarray.make_surface(self.tmp_screen.T) # has already been transposed
        self.screen.blit(pygame.transform.scale2x(self.surface),(0,0))
        pygame.display.flip()

    @classmethod
    def _get_chrome_webdriver(cls):
        chrome_options = webdriver.ChromeOptions()
        chrome_options.add_argument('--proxy-server=%s' % cls.PROXY)
        return webdriver.Chrome(chrome_options=chrome_options)

    @classmethod
    def _get_firefox_webdriver(cls):
        proxy = webdriver.Proxy(raw={
            "httpProxy":cls.PROXY,
            "ftpProxy":cls.PROXY,
            "sslProxy":cls.PROXY,
            "noProxy":None,
            "proxyType":"MANUAL",
            "class":"org.openqa.selenium.Proxy",
            "autodetect":False
        })
        return webdriver.Firefox(proxy=proxy)

if __name__ == "__main__":
    agar = AgarIODriver()
    while True:
        agar.play_game()
