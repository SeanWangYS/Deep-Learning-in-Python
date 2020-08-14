import numpy as np
import matplotlib.pyplot as plt
from sklearn.utils import shuffle
from util import getBinaryData, sigmode, sigmoid_cost, error_rate, relu


'''
binary classification model

這篇運用 numpy ，寫了一個ANN 架構
用上的技術有: L2 regularization
只有一層hidden layer
'''


'''use a ANN + sigmoid(Binary) on a facial expression recognition problem'''
'''Facial Expression Recognition in Code(Binary / Sigmoid)'''


class ANN(object):
    def __init__(self, M):
        self.M = M  # hidden layeer dimension

    def fit(self, X, Y, learning_rate=5*10e-8, reg=1, epochs=3000, show_fig=False):
        X, Y = shuffle(X, Y)
        Xvalid, Yvalid = X[-1000:], Y[-1000:]
        X, Y = X[:-1000], Y[:-1000]

        N, D = X.shape
        self.W1 = np.random.randn(D, self.M) / np.sqrt(D + self.M)
        self.b1 = np.zeros(self.M)
        self.W2 = np.random.randn(self.M) / np.sqrt(self.M)
        self.b2 = 0
        
        costs= []
        best_validation_error = 1
        for i in range(epochs):
            # fordward propagation
            pY, Z  = self.forward(X)

            # gradient descent step
            self.W2 -= learning_rate*(Z.T.dot(pY-Y) + reg*self.W2)    # 這裡是prediction - target (pY-Y)，代表用gradient descent 前面是減號
            self.b2 -= learning_rate*((pY-Y).sum() + reg*self.b2)

            dZ = np.outer((pY-Y), self.W2) * (Z > 0)   # when activation is relu
            # dZ = np.outer((pY-Y), self.W2) * (1 - Z*Z)  # when activation is tanh
            self.W1 -= learning_rate*(X.T.dot(dZ) + reg*self.W1)
            self.b1 -= learning_rate*(dZ.sum(axis=0) + reg*self.b1)

            if i % 20==0:
                pYvalid, _ = self.forward(Xvalid)
                c = sigmoid_cost(Yvalid, pYvalid)
                costs.append(c)
                e = error_rate(Yvalid, np.round(pYvalid))
                print("i:", i, "cost:", c, "error:", e)
                if e < best_validation_error:
                    best_validation_error = e
        print('best_validation_error:', best_validation_error)

        if show_fig:
            plt.plot(costs)
            plt.show()


    def forward(self, X):
        Z = relu(X.dot(self.W1 + self.b1))    # when activation is relu
        # Z = np.tanh(X.dot(self.W1) + self.b1)   # when activation is tanh
        return sigmode(Z.dot(self.W2) + self.b2), Z


    def predict(self, X):
        pY, _ = self.forward(X)
        return np.round(pY)


    def score(self, X):
        prediction, _ = self.forward(X)
        return 1 - error_rate(Y, prediction)
        


def main():
    X, Y = getBinaryData()

    X0 = X[Y==0,:]
    X1 = X[Y==1,:]
    X1 = np.repeat(X1, 9, axis=0)
    X = np.vstack([X0, X1])
    Y = np.array([0]*len(X0) + [1]*len(X1))

    model = ANN(100)
    model.fit(X, Y, show_fig=True)

    print(model.score(Xvalid, Yvalid))


if __name__ == '__main__':
    main()

