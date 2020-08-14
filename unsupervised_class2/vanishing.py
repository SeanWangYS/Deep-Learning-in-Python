import numpy as np
import theano
import theano.tensor as T
import matplotlib.pyplot as plt

from sklearn.utils import shuffle
from autoencoder import momentum_updates
from util import relu, error_rate, getKaggleMNIST, init_weight

class HiddenLayer(object):
    def __init__(self, D, M):
        W = init_weight((D,M))
        b = np.zeros(M, dtype=np.float32)
        self.W = theano.shared(W)
        self.b = theano.shared(b)
        self.params = [self.W, self.b]

    def forward(self, X):
        # we want to use the sigmoid so we can observe
        # the vanishing gradient!
        return T.nnet.sigmoid(X.dot(self.W) + self.b)



class ANN(object):
    def __init__(self, hidden_layer_size):
        self.hidden_layer_size = hidden_layer_size

    def fit(self, X, Y, learning_rate=0.01, mu=0.99, epoches=30, batch_sz=100):
        learning_rate = np.float32(learning_rate)
        mu = np.float32(mu)

        N, D = X.shape
        K = len(set(Y))
        self.hidden_layers = []
        mi = D
        for mo in self.hidden_layer_size:
            h = HiddenLayer(mi, mo)
            self.hidden_layers.append(h)
            mi = mo

        # initialize logistic regression layer
        W = init_weight((mo, K))
        b = np.zeros(K, dtype=np.float32)
        self.W = theano.shared(W)
        self.b = theano.shared(b)

        self.params = [self.W, self.b]
        self.allWs = []
        for h in self.hidden_layers:
            self.params += h.params
            self.allWs.append(h.W)
        self.allWs.append(self.W)

        X_in = T.matrix("X_in", dtype='float32')
        targets = T.ivector('Targets')
        pY = self.forward(X_in)
        
        prediction = self.predict(X_in)
        cost = -T.mean(T.log(pY[T.arange(pY.shape[0]), targets]) )
        updates = momentum_updates(cost, self.params, mu, learning_rate)
        train_op = theano.function(
            inputs=[X_in, targets], 
            updates=updates, 
            outputs=[cost, prediction]
        )

        n_batches = N // batch_sz
        costs = []
        lastWs = [W.get_value() for W in self.allWs]
        W_changes = []
        print("supervised training...")
        for i in range(epoches):
            print("epoch:", i)
            X, Y = shuffle(X, Y)
            for j in range(n_batches):
                Xbatch = X[j*batch_sz:(j*batch_sz + batch_sz)]
                Ybatch = Y[j*batch_sz:(j*batch_sz + batch_sz)]
                c, p = train_op(Xbatch, Ybatch)
                if j % 100 == 0:
                    print("j / n_batches:", j, "/", n_batches, "cost:", c, "error:", error_rate(p, Ybatch))
                costs.append(c)

                # log changes in all Ws
                W_change = [ np.abs(W.get_value() - lastW).mean() for W, lastW in zip(self.allWs, lastWs)]
                W_changes.append(W_change)  # 結構會是->  [ [W_h1, W_h2, W_h3], [W_h1, W_h2, W_h3], [W_h1, W_h2, W_h3], .....]
                lastWs = [W.get_value() for W in self.allWs]
                

        W_changes = np.array(W_changes)
        print(W_changes.shape)
        plt.subplot(2, 1, 1)
        for i in range(W_changes.shape[1]):
            plt.plot(W_changes[:,i], label='layer %s'% i)
        plt.legend()

        plt.subplot(2, 1, 2)
        plt.plot(costs)
        plt.show()


    def forward(self, X):
        current = X
        for ae in self.hidden_layers:
            current = ae.forward(current)
        # logistic layer
        Y = T.nnet.softmax(current.dot(self.W) + self.b)
        return Y

    def predict(self, X):
        return T.argmax(self.forward(X), axis=1)
        
        
def main():
    Xtrain, Ytrain, Xtest, Ytest = getKaggleMNIST()
    dnn = ANN([1000, 750, 500])
    dnn.fit(Xtrain, Ytrain)


if __name__ == "__main__":
    main() 