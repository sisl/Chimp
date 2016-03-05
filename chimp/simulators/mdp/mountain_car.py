import numpy as np

#################################################################
# Implements the mountain car MDP 
#################################################################

class MountainCar():

    def __init__(self,
                 term_r = 10.0,
                 nonterm_r = -1.0,
                 height_reward = True,
                 discrete = False,
                 discount = 0.95):

        self.actions = np.array([-1.0, 0.0, 1.0])
        self.n_actions = 3

        self.state_shape = (1,2) # x and v

        self.term_r = term_r
        self.nonterm_r = nonterm_r

        self.current_state = np.zeros(2, dtype=np.float32)

        self.vmin, self.vmax = (-0.07, 0.07)
        self.xmin, self.xmax = (-1.2, 0.6)

        self.height_reward = height_reward

        self.discrete = discrete
        self.xgrid = 10
        self.vgrid = 10
        self.discrete_x = np.linspace(self.xmin, self.xmax, self.xgrid)
        self.discrete_v = np.linspace(self.vmin, self.vmax, self.vgrid)

        self.sp_discrete = np.zeros(2, dtype=np.float32) 
        

    def transition(self, s, a):
        """
        Returns a next state, given a state and an action
        """
        sp = self.current_state
        #sp = np.zeros(2, dtype=np.float32)
        sp[1] = s[1] + 0.001 * self.actions[a] - 0.0025 * np.cos(3 * s[0])
        sp[1] = self.vclip(sp[1])
        sp[0] = self.xclip(s[0] + sp[1])

        if self.discrete:
            self.sp_discrete[0] = self.find_nearest(self.discrete_x, sp[0])
            self.sp_discrete[1] = self.find_nearest(self.discrete_v, sp[1])
            return self.sp_discrete

        return sp

    
    def reward(self, s, a):
        """
        Rewarded for reaching goal state, penalized for all other states
        """
        r = s[0] if (self.height_reward and s[0] > 0.0) else 0
        if s[0] >= self.xmax:
            r += self.term_r
        else:
            r += self.nonterm_r
        return r


    def isterminal(self, s):
        if s[0] >= self.xmax:
            return True
        return False

    def initial_state(self):
        xi = np.random.uniform(self.xmin, self.xmax*0.9)
        vi = 0.0
        return np.array([xi, vi], dtype=np.float32)


    #################################################################  
    ########################## UTILITIES ############################
    #################################################################  

    def clip(self, val, lo, hi):
        return min(hi, max(val, lo))

    def vclip(self, val):
        return self.clip(val, self.vmin, self.vmax)

    def xclip(self, val):
        return self.clip(val, self.xmin, self.xmax)

    def find_nearest(self, vals, target):
        idx = (np.abs(vals - target)).argmin()
        return vals[target]
