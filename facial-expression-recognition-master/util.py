import os
import numpy as np
import pandas as pd
from sklearn.utils import shuffle

def init_weight_and_bias(M1, M2):
    W = np.random.randn(M1, M2) / np.sqrt(M1 + M2)
    b = np.zeros(M2)
    return W.astype(np.float32), b.astype(np.float32)


def init_filter(shape, poolsz): #for Convolutional Neuron Network
    w = np.random.randn(*shape) / np.sqrt(np.prod(shape[1:]) + shape[0]*np.prod(shape[2:] / np.prod(poolsz)))
    return w.astype(np.float32)


def relu(x):
    return x * (x > 0)


def sigmode(x):
    return 1 / (1 + np.exp(-x))


def softmax(A):
    expA = np.exp(A)
    return expA / expA.sum(axis=1, keepdims=True)


def sigmoid_cost(T, Y):
    return -(T*np.log(Y) + (1-T)*np.log(1-Y)).sum()


def cost(T, Y):
    return -(T*np.log(Y)).sum()


def cost2(T, Y):                         # 功能跟cost()一樣，比較fancy的寫法，還不知道是怎麼寫的
    # same as cost(), just uses the targets to index Y
    # instead of multiplying by a large indicator matrix with mostly 0s
    N = len(T)
    return -np.log(Y[np.arange(N), T]).sum()


def error_rate(target, prediction):      # target, prediction ( vector type)
    return np.mean(target!=prediction)


def y2indicator(y):                        # vector => matrix    (truns a N by one vector of target into a N by K Matrix)
    N = len(y)
    K = len(set(y))
    ind = np.zeros((N, K))
    for i in range(N):
        ind[i, y[i]] = 1
    return ind


def getData(balance_ones=True, Ntest=1000):
    # images are 48x48 = 2304 size vectors
    Y = []
    X = []
    first = True
    for line in open(os.getcwd() + '//facial-expression-recognition-master//fer2013.csv'):
    # for line in open(os.getcwd() + '//fer2013.csv'):  # 在jupyter 測試時用這個路徑
        if first:
            first = False
        else:
            row = line.split(',')
            Y.append(int(row[0]))
            X.append([int(p) for p in row[1].split()])

    X, Y = np.array(X) / 255.0, np.array(Y)

    # shuffle and split
    X, Y = shuffle(X, Y)
    Xtrain, Ytrain = X[:-Ntest], Y[:-Ntest]
    Xvalid, Yvalid = X[-Ntest:], Y[-Ntest:]

    # 這個人臉情緒辨識資料集有個特性，label=1的數量只有其他類別的1/10，所以要做下面的動作
    if balance_ones:
        # balance the 1 class
        X0, Y0 = Xtrain[Ytrain!=1, :], Ytrain[Ytrain!=1]
        X1 = Xtrain[Ytrain==1, :]
        X1 = np.repeat(X1, 9, axis=0)
        Xtrain = np.vstack([X0, X1])
        Ytrain = np.concatenate((Y0, [1]*len(X1)))

    return Xtrain, Ytrain, Xvalid, Yvalid

def getImageData():
    Xtrain, Ytrain, Xvalid, Yvalid = getData()
    N, D = Xtrain.shape
    d = int(np.sqrt(D))
    Xtrain = Xtrain.reshape(N, 1, d, d)
    Xvalid = Xvalid.reshape(N, 1, d, d)
    return Xtrain, Ytrain, Xvalid, Yvalid





def getBinaryData():  # 這個method刻意不處裡 imbalance 的問題
    Y = []
    X = []
    first = True
    for line in open(os.getcwd() + '//facial-expression-recognition-master//fer2013.csv'):
        if first:
            first = False
        else:
            row = line.split(',')
            y = int(row[0])
            if y == 0 or y == 1:
                Y.append(y)
                X.append([int(p) for p in row[1].split()])
    return np.array(X) / 255.0, np.array(Y)


    