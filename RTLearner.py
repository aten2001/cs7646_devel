"""
A simple wrapper for random tree learner.  template code taken from LinRegLearner.py (c) 2015 Tucker Balch
"""


import numpy as np
from random import randint


class RTLearner(object):

    def __init__(self, leaf_size, verbose=False):
        self.leaf_size = leaf_size
        self.verbose = verbose
        self.tree = np.array([])
    def author(self):
        return 'raghavendra6'
    
    def split_index(self, xtrain, num_instances):
        class_index = randint(0, xtrain.shape[1] - 1)
        s1 = randint(0, num_instances - 1)
        s2 = randint(0, num_instances - 1)
        # mean of random instances
        split_val = (xtrain[s1][class_index] + xtrain[s2][class_index]) / 2
        # segregate all the data to left of mean
        left_index  = [i for i in xrange(xtrain.shape[0])
                            if xtrain[i][class_index] <= split_val]
        right_index = [i for i in xrange(xtrain.shape[0])
                            if xtrain[i][class_index] > split_val]
        return left_index, right_index, class_index, split_val
        
   
    def build_tree(self, x_train, y_train):
        num_instances = x_train.shape[0]
        
        if num_instances == 0:
            print 'Null Data'
            return np.array([-1, -1, -1, -1])
        if num_instances <= self.leaf_size:
            # If there's only one instance, take the mean of the labels
            return np.array([-1, np.mean(y_train), -1, -1])
        
        values = np.unique(y_train)
        if len(values) == 1:
            # If all instances have the same label, return that label
            return np.array([-1, y_train[0], -1, -1])
        # derive the split randomly, usual DT uses gini index to get the split
        left_i, right_i, class_index, split_val = self.split_index(x_train, num_instances)
        
        while len(left_i) < 1 or len(right_i) < 1:
            left_i, right_i, class_index, split_val = \
                self.split_index(x_train, num_instances)
                
        left_x_train  = np.array([x_train[i] for i in left_i])
        left_y_train  = np.array([y_train[i] for i in left_i])
        right_x_train = np.array([x_train[i] for i in right_i])
        right_y_train = np.array([y_train[i] for i in right_i])
        
        # recurse the tree to leaf  node
        ltree = self.build_tree(left_x_train, left_y_train)
        rtree = self.build_tree(right_x_train, right_y_train)
        if len(ltree.shape) == 1:
            num_left_instances = 2
        else:
            num_left_instances = ltree.shape[0] + 1
        root = [class_index, split_val, 1, num_left_instances]
        return np.vstack((root, np.vstack((ltree, rtree))))
        
    def addEvidence(self, Xtrain, Ytrain):
        self.tree = self.build_tree(Xtrain, Ytrain)

    def queryTree(self, instance, row=0):
        feature_index = int(self.tree[row][0])
        if feature_index == -1:
            return self.tree[row][1]
        if instance[feature_index] <= self.tree[row][1]:
            return self.queryTree(instance, row + int(self.tree[row][2]))
        else:
            return self.queryTree(instance, row + int(self.tree[row][3]))

    def query(self, Xtest):
        result = []
        for instance in Xtest:
            result.append(self.queryTree(instance))
        return np.array(result)