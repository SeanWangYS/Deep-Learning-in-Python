import numpy as np 
import theano 
import theano.tensor as T 
import matplotlib.pyplot as plt 

from util import init_weight, all_parity_pairs 
from sklearn.utils import shuffle 

'''
MLP 就是一般的 ANN
用傳統的ANN 來解 parity 資料集

(沒有用RNN喔)
'''

class HiddenLayer(object):
    def __init__(self, M1, M2, an_id):
        self.id = an_id
        self.M1 = M1
        self.M2 = M2
        W = init_weight(M1, M2)
        b = np.zeros(M2)
        self.W = theano.shared(W, 'W_%s' % self.id)
        self.b = theano.shared(b, 'b_%s' % self.id)
        self.params = [self.W, self.b]

    def forward(self, X):
        return T.nnet.relu(X.dot(self.W) + self.b)

class ANN(object):
    def __init__(self, hidden_layer_sizes):
        self.hidden_layer_sizes = hidden_layer_sizes

    def fit(self, X, Y, learning_rate=1e-2, mu=0.99, decay=0.999, reg=1e-12, eps=10e-10, epochs=400, batch_sz=20, print_period=1, show_fig=False):
        Y = Y.astype(np.int32)
        X = X.astype(np.float32)
        
        # initial hidden layers
        N, D = X.shape
        K = len(set(Y))
        self.hidden_layers = []
        M1 = D
        count = 0 
        for M2 in self.hidden_layer_sizes:
            h = HiddenLayer(M1, M2, count)
            self.hidden_layers.append(h)
            M1 = M2
            count += 1
        
        # initialize logistic regression layer
        W = init_weight(M1, K)
        b = np.zeros(K)
        self.W = theano.shared(W, 'W_logerg')
        self.b = theano.shared(b, 'b_logreg')

        # collect params for later use
        self.params = [self.W, self.b]
        for h in self.hidden_layers:
           self.params += h.params

        # for momentum
        dparams = [theano.shared(np.zeros(p.get_value().shape)) for p in self.params]

        # for rmsprop
        catch = [theano.shared(np.zeros(p.get_value().shape)) for p in self.params]

        # set up theano functions and variable
        # 1. model
        thX = T.matrix('X')
        thY = T.ivector('Y')
        pY = self.forward(thX)

        # 2. cost
        rcost = reg*T.sum([(p*p).sum() for p in self.params])
        cost = -T.mean(T.log(pY[T.arange(thY.shape[0]), thY])) + rcost
        
        # 3. solver
        grads = T.grad(cost, self.params)
        # RMSpromp
        # updates = [
        #     (p, p + mu*dp - learning_rate*g / T.sqrt(c + eps)) for p, dp, g, c in zip(self.params, dparams, grads, catch)
        # ] + [
        #     (dp, mu*dp - learning_rate*g / T.sqrt(c + eps)) for dp, g, c in zip(dparams, grads, catch)
        # ] + [
        #     (c, c*decay + (np.float32(1.0)-decay)*g*g) for c, g in zip(catch, grads)
        # ]

        # momentum only
        updates = [
            (p, p + mu*dp - learning_rate*g) for p, dp, g in zip(self.params, dparams, grads)
        ] + [
            (dp, mu*dp - learning_rate*g) for dp, g in zip(dparams, grads)
        ]

        prediction = self.predict(thX)
        # theano functions
        train_op = theano.function(
            inputs=[thX, thY],
            updates = updates, 
            outputs = [cost, prediction]
        )

        n_batches = N // batch_sz
        costs = []
        for i in range(epochs):
            X, Y = shuffle(X, Y)
            for j in range(n_batches):
                Xbatch = X[j*batch_sz: (j+1)*batch_sz]
                Ybatch = Y[j*batch_sz: (j+1)*batch_sz]

                c, p = train_op(Xbatch, Ybatch)

                if j % print_period == 0:
                    costs.append(c)
                    e = np.mean(Ybatch != p)
                    print("i:", i, "j:", j, "nb:", n_batches, "cost:", c, "error rate:", e)

        if show_fig:
            plt.plot(costs)
            plt.show()

    def forward(self, X):
        Z = X
        for h in  self.hidden_layers:
            Z = h.forward(Z)
        return T.nnet.softmax(Z.dot(self.W) + self.b)

    def predict(self, X):
        return T.argmax(self.forward(X), axis=1)


def wide():
    X, Y = all_parity_pairs(12)
    model = ANN([2048])
    model.fit(X, Y, learning_rate=1e-4, print_period=10, epochs=300, show_fig=True)

def deep():
    X, Y = all_parity_pairs(12)
    model = ANN([1024]*2)
    model.fit(X, Y, learning_rate=1e-3, print_period=10, epochs=100, show_fig=True)

if __name__ == '__main__':
    # wide()
    deep()

