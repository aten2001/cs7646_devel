"""
template for generating data to fool learners (c) 2016 Tucker Balch
"""

import numpy as np
import math

# this function should return a dataset (X and Y) that will work
# better for linear regresstion than random trees
def best4LinReg(seed=5):
    np.random.seed(seed)
    N = 1000 # number of instances
    X1 = np.linspace(1, 10**4, num=N)  + np.random.normal(0,10**2,N)
    X2 = np.linspace(1, 10**2, num=N)  + np.random.normal(0,10**2,N)
    Y  = np.linspace(1, 10**5, num=N)  + np.random.normal(0,10**2,N)
    X = np.column_stack((X1, X2))
    return X, Y

def best4RT(seed=5):
    np.random.seed(seed)
    N = 1000 # number of instances
    data = np.random.randn(N,2)
    # add noise uniformly in data
    for row in range(50,1000,50):
        data[row:(row+50),0] = data[row:(row+50),0] + np.random.randint(0,1000)
        data[row:(row+50),1] = data[row:(row+50),1] - np.random.randint(0,1000)

    X1 = data[:,0]
    X2 = data[:,1]
    # Derive Y to be a monotonically increasing response, since RT is a non parametric learner
    Y = np.linspace(1, 10**4, num=N)  + np.random.normal(0,10**2,N)
    X = np.column_stack((X1, X2))
    return X, Y

def author():
    return 'raghavendra6'

if __name__=="__main__":
    print "they call me Tim."
