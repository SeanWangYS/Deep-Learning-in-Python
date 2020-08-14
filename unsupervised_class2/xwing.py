import numpy as np
import theano 
theano.config.floatX = 'float32'
import theano.tensor as T 
import matplotlib.pyplot as plt 

from sklearn.utils import shuffle
from util import relu, error_rate, getKaggleMNIST, init_weight


import time

''' Deep Auto-encoder 做視覺化，看來主要目的就是做視覺化'''

# class Layer : 單層layer的參數定義與初始化 + 單層layer的forward propagation
class Layer(object):
    def __init__(self, m1, m2):
        W = init_weight((m1, m2))
        bi = np.zeros(m2, dtype=np.float32)
        bo = np.zeros(m1, dtype=np.float32)
        self.W = theano.shared(W)
        self.bi = theano.shared(bi)
        self.bo = theano.shared(bo)
        self.params = [self.W, self.bi, self.bo]

    def forward(self, X):
        return T.nnet.sigmoid(X.dot(self.W) + self.bi)

    def forwardT(self, X):
        return T.nnet.sigmoid(X.dot(self.W.T) + self.bo)

class DeepAutoEncoder(object):
    def __init__(self, hidden_layer_sizes):
        self.hidden_layer_sizes = hidden_layer_sizes

    def fit(self, X, learning_rate=0.5, mu=0.99, epochs=50, batch_sz=100, show_fig=False):
        N, D = X.shape
        n_batches = N // batch_sz

        mi = D
        self.layers = []
        self.params = []
        for mo in self.hidden_layer_sizes:
            layer = Layer(mi, mo)
            self.layers.append(layer)
            self.params += layer.params
            mi = mo

        X_in = T.matrix('X', dtype='float32')
        X_hat = self.forward(X_in)

        cost = -(X_in * T.log(X_hat) + (1 - X_in) * T.log(1 - X_hat)).mean()
        cost_op = theano.function(
            inputs=[X_in], 
            outputs=cost,
        )

        dparams = [theano.shared(p.get_value()*0) for p in self.params]
        grads = T.grad(cost, self.params)
        updates = [
            (p, p + mu*dp - learning_rate*g) for p, dp, g in zip(self.params, dparams, grads)
        ] + [
            (dp, mu*dp - learning_rate*g) for dp, g, in zip(dparams, grads)
        ]
        train_op = theano.function(
            inputs=[X_in],
            updates=updates,
        )

        costs = []
        for i in range(epochs):
            print('epoch:', i)
            X = shuffle(X)
            for j in range(n_batches):
                batch = X[j*batch_sz : (j*batch_sz + batch_sz)]
                train_op(batch)
                c = cost_op(batch)
                costs.append(c)
                if j % 100 == 0:
                    print('j / n_batches:', j, '/', n_batches, 'cost:', c)
        if show_fig:
            plt.plot(costs)
            plt.show()

    def forward(self, X):
        Z = X
        # 先跑模型前半結構
        for layer in self.layers:
            Z = layer.forward(Z) #....這時候已經跑到正中間hidden layer

        # 擷取NN結構正中間hidden layer 的實際值，作圖用
        self.map2center = theano.function(
            inputs=[X],
            outputs=Z,
        )
        # 再跑模型後半結構
        for i in range(len(self.layers)-1, -1, -1):
            Z = self.layers[i].forwardT(Z)

        # forward propagation output 回傳
        return Z

    
def main():
    Xtrain, Ytrain, Xtest, Ytest = getKaggleMNIST()
    dae = DeepAutoEncoder([500, 300, 2])
    start_time = time.time() # 時間戳記
    dae.fit(Xtrain, show_fig=True)
    end_time = time.time()   # 時間戳記
    mapping = dae.map2center(Xtrain)
    plt.scatter(mapping[:,0], mapping[:, 1], c=Ytrain, s=100, alpha=0.5)
    print(end_time - start_time)


if __name__ == '__main__':
    main()
