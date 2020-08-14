from __future__ import print_function, division
from builtins import range

import numpy as np 
import pandas as pd 
import os

# so scripts from other folders can import this file
dir_path = os.path.abspath(os.path.dirname(os.path.realpath(__file__)))

def get_data():
    df = pd.read_csv(dir_path +'/ecommerce_data.csv')
    data = df.values    
    
    X = data[:, :-1]
    Y = data[:, -1].astype(np.int32)

    # normalize
    X[:, 1] = (X[:, 1] - X[:, 1].mean()) / X[:, 1].std()
    X[:, 2] = (X[:, 2] - X[:, 2].mean()) / X[:, 2].std()

    # one-hot encode the categorical data
    # create a new matrix X2 with the correct number of columns
    N, D = X.shape
    X2 = np.zeros((N, D+3))
    X2[:, 0:(D-1)] = X[:, 0:(D-1)]  # non-categorical

    # one-hot
    for n in range(N):
        t = int(X[n, D-1])
        X2[n, D-1+t] = 1

    # one-hot (method2)
    # Z = np.zeros((N, 4))
    # Z[np.arange(N), X[:, D-1].astype(np.int32)] = 1
    # X2[:, -4:] = Z

    # assert(np.abs(X2[:, -4]-Z).sum() < 10e-10)

    return X2, Y

def get_binary_data():
    # return only the data from the first 2 classes
    X, Y = get_data()
    X2 = X[Y <= 1]
    Y2 = Y[Y <= 1]
    return X2, Y2



    


