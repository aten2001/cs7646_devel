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
        self.alpha = alpha
        self.gamma = gamma
        self.rar = rar
        self.radr = radr
        self.dyna = dyna
        self.verbose = verbose

        self.Q = np.random.uniform(-1, 1, size=(num_states, num_actions))

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

        self.a = action

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
        #action = rand.randint(0, self.num_actions-1)

        self.Q[self.s, self.a] =(((1 - self.alpha) * self.Q[self.s, self.a]) +
                                self.alpha * (r + self.gamma * self.Q[s_prime, np.argmax(self.Q[s_prime, :])]))
        self.s = s_prime
        self.a = self.querysetstate(self.s)

        if self.verbose: print "s =", s_prime,"a =", self.a,"r =",r
        return self.a

if __name__=="__main__":
    print "Remember Q from Star Trek? Well, this isn't him"

    #learner = QLearner()
    #print learner.Q

