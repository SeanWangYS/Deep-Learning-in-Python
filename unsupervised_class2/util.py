import numpy as np 
import pandas as pd 
import os 

from sklearn.utils import shuffle

def getKaggleMNIST():
    # MNIST data:
    # column 0 is labels
    # column 1-785 is data, with values 0 .. 255
    # total size of CSV: (42000, 1, 28, 28)
    # train = pd.read_csv(os.getcwd() + '\\large_files\\train.csv').values.astype(np.float32)
    train = pd.read_csv('..\\large_files\\train.csv').values.astype(np.float32) # for .ipynb file
    train = shuffle(train)

    Xtrain = train[:-1000, 1:] / 255
    Ytrain = train[:-1000, 0].astype(np.int32)

    Xtest = train[-1000:, 1:] / 255
    Ytest = train[-1000:, 0].astype(np.int32)
    return Xtrain, Ytrain, Xtest, Ytest

def relu(x):
    return x * (x > 0)

def error_rate(p, t):
    return np.mean(p != t)

def init_weight(shape):
    # 如果你有個函式擁有固定的參數，你可以將一個Tuple傳入，只要在傳入時加上*，則Tuple中每個元素會自動指定給各個參數
    w = np.random.randn(*shape) / np.sqrt(sum(shape))
    return w.astype(np.float32)


def main():
    w = init_weight((3,3,1))
    print(type(w))
    print(w)

if __name__ == "__main__":
    main()

