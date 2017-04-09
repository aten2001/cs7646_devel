"""
Template for implementing QLearner  (c) 2015 Tucker Balch
"""

import numpy as np
import random as rand

class QLearner(object):

    def __init__(self, \
        num_states=100, \
        num_actions = 4, \
        alpha = 0.2, \
        gamma = 0.9, \
        rar = 0.5, \
        radr = 0.99, \
        dyna = 0, \
        verbose = False):

        self.num_states = num_states
        self.num_actions = num_actions
        self.actions_possible = np.array(range(self.num_actions))
        self.alpha = alpha
        self.gamma = gamma
        self.rar = rar
        self.radr = radr
        self.dyna = dyna
        # initialize Q[] with uniform random values between -1.0 and 1.0
        self.Q = np.random.uniform(-1, 1, size=(num_states, num_actions))

        # init T, TC & Rewards matrix if dyna is set.
        if self.dyna > 0:
            # pre lecture
            self.Tc = 0.00001 * np.ones((num_states, num_actions, num_states))
            # standard formula
            self.T = self.Tc / self.Tc.sum(axis=2, keepdims=True)
            # intialize R
            self.R = -1.0 * np.ones((num_states, num_actions))

        self.verbose = verbose
        self.num_actions = num_actions
        self.s = 0
        self.a = 0

    def querysetstate(self, s):
        """
        @summary: Update the state without updating the Q-table
        @param s: The new state
        @returns: The selected action
        """
        self.s = s
        prand = np.random.random()

        if prand < self.rar:
            action = rand.randint(0, self.num_actions-1)
        else:
            action = np.argmax(self.Q[s, :])

        self.rar = self.rar * self.radr

        if self.verbose: print "s =", s,"a =",action
        return action

    def query(self,s_prime,r):
        """
        @summary: Update the Q table and return an action
        @param s_prime: The new state
        @param r: The ne state
        @returns: The selected action
        """

        # QTable Update
        self.Q[self.s, self.a] =(((1 - self.alpha) * self.Q[self.s, self.a]) +
                                self.alpha * (r + self.gamma * self.Q[s_prime, np.argmax(self.Q[s_prime, :])]))

        if self.dyna > 0:
            # keep track of t_c.
            self.Tc[self.s, self.a, s_prime] += 1

            # UPdate T matrix as we go through iterations
            self.T[self.s, self.a, :] = self.Tc[self.s, self.a, :] / self.Tc[self.s, self.a, :].sum()

            # UPdate R
            self.R[self.s, self.a] = (1 - self.alpha) * self.R[self.s, self.a] + \
                self.alpha * r

            # upadate Q based on dyna
            self._genDynaQData()


        self.s = s_prime
        self.a = self.querysetstate(self.s)
        action = self.a

        # default code
        #action = rand.randint(0, self.num_actions-1)
        if self.verbose: print "s =", s_prime,"a =",action,"r =",r
        return action

    def _genDynaQData(self):
        s_dyna = np.random.randint(0, self.num_states, self.dyna)
        a_dyna = np.random.randint(0, self.num_actions, self.dyna)
        for i in range(self.dyna):
            s = s_dyna[i]
            a = a_dyna[i]
            # Infer new state based on experience
            s_prime = np.argmax(np.random.multinomial(1, self.T[s, a, :]))
            # compute r based on new data
            r = self.R[s, a]
            #UPdate Q
            self.Q[s, a] = (1 - self.alpha) * self.Q[s, a] + \
                            self.alpha * (r + self.gamma * np.max(self.Q[s_prime,:]))




if __name__=="__main__":
    print "Remember Q from Star Trek? Well, this isn't him"
