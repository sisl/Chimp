import numpy as np

#################################################################
# Implements the simulator class for MDPs 
#################################################################

class MountainCar():

    def __init__(self,
                 term_r = 10.0,
                 nonterm_r = -1.0,
                 discount = 0.95):

        self.actions = np.array([-1.0, 0.0, 1.0])
        self.n_actions = 3

        self.state_shape = (1,2) # x and v

        self.term_r = term_r
        self.nonterm_r = nonterm_r

        self.current_state = np.zeros(2, dtype=np.float32)

        self.vmin, self.vmax = (-0.07, 0.07)
        self.xmin, self.xmax = (-1.2, 0.6)
        

    def transition(self, s, a):
        """
        Returns a next state, given a state and an action
        """
        sp = self.current_state
        #sp = np.zeros(2, dtype=np.float32)
        #sp[1] = s[1] + 0.001 * self.actions[a] - 0.0025 * np.cos(3 * s[0])
        sp[1] = s[1] + 1.0 * self.actions[a] - 0.0025 * np.cos(3 * s[0])
        sp[1] = self.vclip(sp[1])
        sp[0] = self.xclip(s[0] + sp[1])

        return sp

    
    def reward(self, s, a):
        """
        Rewarded for reaching goal state, penalized for all other states
        """
        if s[0] >= self.xmax:
            return self.term_r
        else:
            return self.nonterm_r

    def isterminal(self, s):
        if s[0] >= self.xmax:
            return True
        return False

    def n_actions(self):
        return self.n_actions

    def initial_state(self):
        xi = np.random.uniform(self.xmin/2.0, self.xmax/2.0)
        vi = 0.0
        return np.array([xi, vi])


    #################################################################  
    ########################## UTILITIES ############################
    #################################################################  

    def clip(self, val, lo, hi):
        return min(hi, max(val, lo))

    def vclip(self, val):
        return self.clip(val, self.vmin, self.vmax)

    def xclip(self, val):
        return self.clip(val, self.xmin, self.xmax)
