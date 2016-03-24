import numpy as np

#################################################################
# Implements the simulator class for pole cart MDP
#################################################################

class CartPole():

    def __init__(self):
        self.actions = np.array([-1,1])
        self.n_actions = 2

        self.state_shape = (1,4) # x, xdot, theta, thetadot

        self.gravity = 9.8
        self.mass_cart = 1.0
        self.mass_pole = 0.3
        self.total_mass = self.mass_cart + self.mass_pole
        self.length = 0.7
        self.polemass_length = self.mass_pole * self.length
        self.force_mag = 10.0
        self.tau = 0.02

        self.term_deg = 0.2094384


    def transition(self, s, a):
        if self.isterminal(s):
            return s.copy()
        x, xdot, theta, thetadot = s[0], s[1], s[2], s[3]

        sint = np.sin(theta)
        cost = np.cos(theta)

        force = self.actions[a] * self.force_mag

        temp = (force + self.polemass_length * thetadot**2 * sint) / self.total_mass
        thetaacc = (self.gravity * sint - cost * temp) / (self.length * (4.0/3.0 - self.mass_pole * cost**2 /
            self.total_mass))
        xacc = temp - self.polemass_length * thetaacc * cost / self.total_mass

        sp = np.zeros(4, dtype=np.float32)
        sp[0] = x + self.tau * xdot
        sp[1] = xdot + self.tau * xacc
        sp[2] = theta + self.tau * thetadot
        sp[3] = thetadot + self.tau * thetaacc

        return sp

    def reward(self, s, a):
        r = 0.0
        if self.isterminal(s):
            r = -1.0
        return r
        

    def isterminal(self, s):
        if (s[0] < -2.4 or s[0] > 2.4 or s[2] < -self.term_deg or s[2] > self.term_deg):
            return True
        return False


    def initial_state(self):
        s = np.zeros(4, dtype=np.float32)
        s[0] = 2.2 * np.random.rand() - 1.1
        s[1], s[2], s[3] = 0.0, 0.0, 0.0
        return s

