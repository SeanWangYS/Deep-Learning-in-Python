import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from util import getData, softmax, cost, y2indicator, error_rate
from sklearn.utils import shuffle

'''
Classification model

這篇運用 numpy ，寫了一個 Logistic Regression
用上的技術有: L2 regularization
'''


'''use logistic regression + softmax (multi-classes) on a facial expression recognition problem'''
'''Facial Expression Recognition in Code (Logistic Regression Softmax)'''

class LogisticModel(object):
    def __init__(self):
        pass

    def fit(self, X, Y, Xvalid, Yvalid, learning_rate=1e-6, reg=0., epochs=10000, show_fig=False):
        Tvalid = y2indicator(Yvalid)
        
        N, D = X.shape
        K = len(set(Y) | set(Yvalid))
        T = y2indicator(Y)
        
        self.W1 = np.random.randn(D, K) / np.sqrt(D)
        self.b1 = np.zeros(K)

        costs =[]
        best_validation_error = 1
        for i in range(epochs):
            # forward propagation and cost calculation
            pY = self.forward(X)

            # Gradient Descent step
            self.W1 -= learning_rate*(X.T.dot(pY -T) + reg*self.W1)
            self.b1 -= learning_rate*((pY-T).sum(axis=0) + reg*self.b1)
            
            if i%20 == 0:
                pYvalid = self.forward(Xvalid)
                c = cost(Tvalid, pYvalid)
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
        A = X.dot(self.W1) + self.b1
        return softmax(A)


    def predict(self, X):
        pY = self.forward(X)
        return np.argmax(pY, axis=1)

    def score(self, X, Y):
        prediction = self.predict(X)
        return 1 - error_rate(Y, prediction)



def main():
    Xtrain, Ytrain, Xvalid, Yvalid = getData()
    
    model = LogisticModel()
    model.fit(Xtrain, Ytrain, Xvalid, Yvalid, show_fig=True)

    print(model.score(Xvalid, Yvalid))



if __name__ == '__main__':
    main()


