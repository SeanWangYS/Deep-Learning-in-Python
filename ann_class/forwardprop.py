# 這邊做的事情是，讓py3語法的code可以在py2的環境下run
# original Env in tutorial : Python2.7
from __future__ import print_function, division       # 可以讓py3的code 在py2底下運行
from builtins import range                            # 可以讓range語法在py2底下運行

import numpy as np
import matplotlib.pyplot as plt

'''寫出一個forward propagation 架構的Neuron Network，還沒加入weight udate的功能'''
# create the data
Nclass = 500
X1 = np.random.randn(Nclass, 2) + np.array([0, 2])
X2 = np.random.randn(Nclass, 2) + np.array([2, 2])
X3 = np.random.randn(Nclass, 2) + np.array([-2, -2])
X = np.vstack([X1, X2, X3])

# target
Y = np.array([0]*Nclass + [1]*Nclass + [2]*Nclass)

plt.scatter(X[:,0], X[:,1], c=Y, s=100, alpha=0.5)
plt.show()

# initialize weights 
D = 2                           # dimensionality of input
M = 3                           # hidden layer size
K = 3                           # number of classes
W1 = np.random.randn(D, M)
b1 = np.random.randn(M)
W2 = np.random.randn(M, K)
b2 = np.random.randn(K)

def sigmoid(a):                   # 無論input是vextor ro matrix，sigmoid作用的對象，是每一個inner element
    return 1 / (1 + np.exp(-a))

def forword(X, W1, b1, W2, b2):
    Z = sigmoid(X.dot(W1) + b1)
    A = Z.dot(W2) + b2
    expA = np.exp(A)           
    Y = expA / expA.sum(axis=1, keepdims=True)   # softmax 作用的對象是個別針對每個row(有N個)的K-dim
    return Y                                     # (N, K)

def classification_rate(Y, P):                    # Y= target(N,) , P= predicted lebel(N,)
    n_correct = 0
    n_total = 0
    for i in range(len(Y)):
        n_total+=1
        if Y[i] == P[i]:                          # input are vector 比較對象是每一個sample的預測結果，是element對element的比較
            n_correct+=1
    return float(n_correct) / n_total


P_Y_given_X = forword(X, W1, b1, W2, b2)
P = np.argmax(P_Y_given_X, axis=1)                # 把softmax後的 matrix (prediction)轉成 vector，類似用threshold將每個預測結果判斷成對應的 category

# verify we chose the correct axis
assert(len(P) == len(Y))

print ("Classification rate for rondomly cohesen weight:", classification_rate(Y, P))


