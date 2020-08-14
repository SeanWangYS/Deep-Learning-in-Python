from __future__ import print_function, division
from builtins import range

import numpy as np
import matplotlib.pyplot as plt
''' NN架構，並完成model training'''

'''步驟:
    Step0. 準備raw data
        0.0. 確認X , y shape
        0.1. y(targer)，從vector 轉成(one hot encoding)適合NN training 的shape

    Step1. model
        1.1 建立forward()
        1.2 視情況建立predict()，其實就是forward()的output，轉成適合做error rate計算的 shape

    Step2. cost function
        2.1. 建立 cost() ，就是 negtive log-likelihood or cross-entropy or SSE
        2.2  建立 classification_rate() 得出準確率
        PS. 這裡的cost function 視問題選擇相應的，以coding角度看
        cost funciton的選擇與下面training 過程無關

    Step3. solver
        3.1 weight initialization
        3.2 確認NN架構中所有layer的 unit 個數，預先變數指派
        3.3 需要決定的training 參數: learning_rate, epoch
        3.4 預先建立變數costs 紀錄每次Epoch，cost value的增減
        3.5 equations of weight updating 
'''

def forward(X, W1, b1, W2, b2):
    Z = 1 / (1 + np.exp(-X.dot(W1) - b1))        # (N, M)
    A = Z.dot(W2) + b2
    expA = np.exp(A)
    Y = expA / expA.sum(axis=1, keepdims=True)   # (N, K)
    return Y, Z  #  output, hidden


# determine the classification rate
# num correct / num total
def classification_rate(Y, P): # Y= target(N,) , P= predicted lebel(N,)
    n_correct = 0
    n_total = 0
    for i in range(len(Y)):
        n_total += 1
        if Y[i] == P[i]:
            n_correct += 1
    return float(n_correct) / n_total


def cost(T, Y): # T=target(N,K), Y=output(N, K)
    tot = T * np.log(Y)
    return tot.sum()


def derivative_w2(Z, T, Y): # hidden, target, output
    N, K = T.shape
    M = Z.shape[1]

    # # slow 
    # ret1 = np.zeros((M, K))
    # for n in range(N):
    #     for m in range(M):
    #         for k in range(K):
    #             ret1[m, k] += (T[n,k] - Y[n,k]) * Z[n,m]

    # # a bit faster - let's not loop over m
    # ret2 = np.zeros((M, K))
    # for n in range(N):
    #     for k in range(K):
    #         ret2[:, k] += (T[n,k] - Y[n,k]) * Z[n,:]
    # assert(np.abs(ret1 - ret2).sum() < 10e-10)

    # # even faster  - let's not loop over k either
    # ret3 = np.zeros((M, K))
    # for n in xrange(N): # slow way first
    #     ret3 += np.outer( Z[n], T[n] - Y[n] )
    # assert(np.abs(ret1 - ret3).sum() < 0.00001)

    # fastest - let's not loop over anything
    ret4 = Z.T.dot(T - Y) # (M,K)
    return ret4


def derivative_b2(T, Y): # target, output
    return (T - Y).sum(axis=0) # (M,)


def derivative_w1(X, Z, W2, T, Y): # X, hidden, W2, target, output 
    N, D = X.shape
    M, K = W2.shape

    # # slow
    # ret1 = np.zeros((D, M))
    # for n in range(N):
    #     for k in range(K):
    #         for d in range(D):
    #             for m in range(M):
    #                 ret1[d ,m ] += (T[n,k] - Y[n,k])*W2[m,k]*Z[n,m]*(1 - Z[n,m])*X[n,d]
    # return ret1
    return X.T.dot((T-Y).dot(W2.T) * Z* (1-Z))


def derivative_b1(Z, W2, T, Y):  # hidden, W2, T, output   
    return ((T -Y).dot(W2.T) * Z * (1-Z)).sum(axis=0) # (K,)


def main():
    # create the data
    Nclass = 500
    D = 2                           # dimensionality of input
    M = 3                           # hidden layer size
    K = 3                           # number of classes

    X1 = np.random.randn(Nclass, D) + np.array([0, -2])
    X2 = np.random.randn(Nclass, D) + np.array([2, 2])
    X3 = np.random.randn(Nclass, D) + np.array([-2, 2])
    X = np.vstack([X1, X2, X3])                           # (N, 2)
    
    # traget
    Y = np.array([0]*Nclass + [1]*Nclass + [2]*Nclass)    # (N, )
    N = len(Y)  # 1500

    # traget轉 適合做gradient descdent的型別
    T =  np.zeros((N, K))        
    for i in range(N):             
        T[i, Y[i]] = 1             # one hot encoding for Target (turn the target into a indicate variable)
    

    # let's see what it looks like
    plt.scatter(X[:,0], X[:,1], c=Y, s=100, alpha=0.5)
    plt.show()


    # randomly initialize weights
    W1 = np.random.randn(D, M)
    b1 = np.random.randn(M)
    W2 = np.random.randn(M, K)
    b2 = np.random.randn(K)

    learning_rate = 10e-5
    costs = []
    for epoch in range(1000):
        output, hidden = forward(X, W1, b1, W2, b2)
        if epoch % 100 == 0:
            c = cost(T, output) 
            P = np.argmax(output, axis=1)
            r = classification_rate(Y, P) # input: target and prediction 
            print("cost:", c,  "classification_rate:", r)
            costs.append(c)

        # this is gradient ASCENT, not DESCENT
        # going to do gradient ascent, it's just backward of gradient descent !!!!!!!!!!!!(是這樣子的喔)
        W2 += learning_rate * derivative_w2(hidden, T, output)
        b2 += learning_rate * derivative_b2(T, output)
        W1 += learning_rate * derivative_w1(X, hidden, W2, T, output)
        b1 += learning_rate * derivative_b1(hidden, W2, T, output)

    plt.plot(costs)
    plt.show()

if __name__ == '__main__':
    main()


