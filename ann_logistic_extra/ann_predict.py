from __future__ import print_function, division
from builtins import range
# Note: you may need to update your version of future
# sudo pip install -U future

'''寫出一個forward propagation 架構的Neuron Network，還沒加入weight udate的功能'''

import numpy as np 
from process import get_data

X, Y = get_data()

# randomly initialize weights
M = 5
D = X.shape[1]
K = len(set(Y))

W1 = np.random.randn(D, M)
b1 = np.zeros(M)
W2 = np.random.randn(M, K)
b2 = np.zeros(K)

# make predictions
def softmax(a):
    expA = np.exp(a)
    return expA / expA.sum(axis=1, keepdims=True)

def forward(X, W1, b1, W2, b2):
    Z = np.tanh(X.dot(W1) + b1)
    return softmax(Z.dot(W2) + b2)

P_Y_given_X = forward(X, W1, b1, W2, b2)
print("P_Y_given_X.shape: ", P_Y_given_X.shape)
predictions = np.argmax(P_Y_given_X, axis=1)

# calculate the accuracy
def classifacatioin_rate(Y, P):
    return np.mean(Y == P)

print("Score: ", classifacatioin_rate(Y, predictions))

