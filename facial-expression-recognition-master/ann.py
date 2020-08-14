import numpy as np
import matplotlib.pyplot as plt

from util import getData, softmax, cost, y2indicator, error_rate, relu, cost2
from sklearn.utils import shuffle

'''
Classification model

這篇運用 numpy ，寫了一個ANN 架構
用上的技術有: L2 regularization
只有一層hidden layer
'''


'''use traditional ANN on a facial expression recognition problem'''
'''Facial Expression Recognition in Code (ANN Softmax)'''


class ANN(object):
    def __init__(self, M):
        self.M = M


    def fit(self, X, Y, Xvalid, Yvalid, learning_rate=1e-6, reg=1e-6, epochs=10000, show_fig=False):
        Tvalid = y2indicator(Yvalid)

        N, D = X.shape
        K = len(set(Y) | set(Yvalid))
        T = y2indicator(Y)

        self.W1 = np.random.randn(D, self.M) / np.sqrt(D)
        self.b1 = np.zeros(self.M)
        self.W2 = np.random.randn(self.M,K) / np.sqrt(self.M)
        self.b2 = np.zeros(K)

        costs =[]
        best_validation_error = 1
        for i in range(epochs):
            # forward propagation and cost calculation
            pY, Z = self.forward(X)

            # Gradient Descent step
            ''' 在玩這個資料集的時候，首度引入L2概念， 注意L2原本就寫在 loss function內，
            整個loss funciton作微分後，||W||變成一次方的型態，如下方運算式'''
            pY_T = pY - T  # 先設成變數，這樣之後計算才會快阿
            self.W2 -= learning_rate*(Z.T.dot(pY_T) + reg*self.W2)
            self.b2 -= learning_rate*(pY_T.sum(axis=0) + reg*self.b2)
            # dZ = pY_T.dot(self.W2.T) * (Z > 0) # relu
            dZ = pY_T.dot(self.W2.T) * (1 - Z*Z) # tanh
            self.W1 -= learning_rate*(X.T.dot(dZ) + reg*self.W1)
            self.b1 -= learning_rate*(dZ.sum(axis=0) + reg*self.b1)

            if i%20 == 0:
                pYvalid, _ = self.forward(Xvalid)
                c = cost2(Yvalid, pYvalid)
                # c = cost(Tvalid, pYvalid)
                costs.append(c)
                e = error_rate(Yvalid, np.argmax(pYvalid, axis=1))
                print("i:", i, "cost:", c, "error:", e)
                if e < best_validation_error:
                    best_validation_error = e
        print('best_validation_error:', best_validation_error)

        if show_fig:
            plt.plot(costs)
            plt.show()


    def forward(self, X):
        # Z = relu(X.dot(self.W1) + self.b1)
        Z = np.tanh(X.dot(self.W1) + self.b1)
        return softmax(Z.dot(self.W2) + self.b2), Z


    def predict(self, X):
        pY, _ = self.forward(X)
        return np.argmax(pY, axis=1)


    def score(self, X, Y):
        prediction = self.predict(X)
        return 1 - error_rate(Y, prediction)


def main():
    Xtrain, Ytrain, Xvalid, Yvalid = getData()
    
    model = ANN(200)
    model.fit(Xtrain, Ytrain, Xvalid, Yvalid, reg=0, show_fig=True)

    print(model.score(Xvalid, Yvalid))
 


if __name__ == '__main__':
    main()

