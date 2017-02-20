"""
A simple wrapper for bagLearner that makes use of random tree learner, linRegLearner.  template code taken from LinRegLearner.py (c) 2015 Tucker Balch
"""

import numpy as np
from random import randint
import RTLearner as rtl

class BagLearner(object):

    def __init__(self, learner, bags, kwargs=None, boost=False, verbose=False):
        self.learner = learner
        if learner == rtl.RTLearner:
            if kwargs['leaf_size'] <= 0:
                raise ValueError('leaf_size: must be > 0')
            else:
                self.kwargs = kwargs['leaf_size']
        else:
            self.kwargs = kwargs
        
        self.bags  = bags
        self.boost = boost # WIP
        

        
            
    def author(self):
        return 'raghavendra6' 

    def addEvidence(self, Xtrain, Ytrain):
        self.Xtrain = Xtrain
        self.Ytrain = Ytrain
    
    def get_rand_index(self, high, size):
        for i in xrange(size):
            rand_indexes = np.random.randint(0,high, size)
        return rand_indexes
       
        
    def query(self,Xtest):
        #  in sample split is at 0.6
        bag_size = self.Xtrain.shape[0]
        results = []
        # initiliaze the dataset
        Xtrain = np.zeros((bag_size, self.Xtrain.shape[1]), dtype='float')
        Ytrain = np.zeros((bag_size, ), dtype='float')
        
        for i in range(self.bags):
            if self.kwargs:
                learner = self.learner(self.kwargs)
            else:
                learner = self.learner()
            rand_indexes = self.get_rand_index(self.Xtrain.shape[0], bag_size)
            j = 0
            for i in rand_indexes:
                Xtrain[j] = self.Xtrain[i, :]
                Ytrain[j] = self.Ytrain[i]
                
                j += 1
            # recurse and add evidence        
            learner.addEvidence(Xtrain, Ytrain)
            results.append(learner.query(Xtest))
        # mean of results coming out of each bag
        result = sum(results)/len(results)
        return [float(i) for i in result]
        
        
if __name__=="__main__":
    print "the secret clue is 'zzyzx'"
