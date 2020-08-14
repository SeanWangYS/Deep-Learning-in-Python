import numpy as np
import theano
import theano.tensor as T
import matplotlib.pyplot as plt
from theano.tensor.shared_randomstreams import RandomStreams
from util import get_normalized_data
from sklearn.utils import shuffle

'''
classification model

這篇運用theano，寫了一個ANN的api
用上的技術有:  momentum, RMSProm, mini batch gradient descent, dropout
可建立多層hidden layer 的 class
'''
'''
note0212
還沒debug出來，一樣卡在型別問題，有個參數變數是np.float64，不知道藏在哪裡
後來把tutorial的範例copy過來再改成自己要的格式跟註解，就可以run了，還是不知道為什麼????機車
'''

class HiddenLayer(object):
    # HiddenLayer 物件，entity物件，建立單一hidden layer 的參數初始化後的物件
    def __init__(self, M1, M2, an_id):
        self.id = an_id
        self.M1 = M1
        self.M2 = M2
        W = np.random.randn(M1, M2) * np.sqrt(2.0 / M1)
        b = np.zeros(M2)
        self.W = theano.shared(W, 'W_%s' % self.id)
        self.b = theano.shared(b, 'b_%s' % self.id)
        self.params = [self.W, self.b]

    def forward(self, X):
        return T.nnet.relu(X.dot(self.W) + self.b)


class ANN(object):
    def __init__(self, hidden_layer_sizes, p_keep):
        self.hidden_layer_sizes = hidden_layer_sizes
        self.dropout_rates = p_keep       # this is a list of probability

    def fit(self, X, Y, Xvalid, Yvalid, learning_rate=1e-3, mu=0.99, decay=0.999, eps=1e-8, epochs=6, batch_sz=100, show_fig=False):
        # step1 get the data 
        X, Y =  shuffle(X, Y)
        X = X.astype(np.float32)
        Y = Y.astype(np.int32)
        Xvalid = Xvalid.astype(np.float32)
        Yvalid = Yvalid.astype(np.int32)

        self.rng = RandomStreams()

        # initialize hidden layers
        N, D = X.shape
        K = len(set(Y))
        self.hidden_layers = []
        M1 = D
        count = 0
        for M2 in self.hidden_layer_sizes:    # 這邊做出順序從第一層~ 最後一層 hidden layer 的HiddenLayer object
            h = HiddenLayer(M1, M2, count)
            self.hidden_layers.append(h)
            M1 = M2
            count += 1
        W = np.random.randn(M1, K) * np.sqrt(2.0 / M1)
        b = np.zeros(K)
        self.W = theano.shared(W, 'W_logreg')
        self.b = theano.shared(b, 'b_logreg')

        # collect params for later use
        self.params = [self.W, self.b]    # 先放最後一層output layer 進 params
        for h in self.hidden_layers:
            self.params += h.params    # 接著應該是照著 hidden lyer1 , hidden layer2, 的順序放進去params

        # step2. model
        # theano variabels
        thX = T.matrix('X')    # data input (matrix)
        thY = T.ivector('Y')   # target ( vector)
        pY_train = self.forward_train(thX)    # the function for dropout while training 

        # define theano's computation grath
        # step3. cost function
        cost = -T.mean(T.log(pY_train[T.arange(thY.shape[0]), thY]))

        # step4. solver
        # define theano's operations
        # for momentum, we need to create zero matrix for each layer

        # for momentum
        dparams = [theano.shared(np.zeros_like(p.get_value())) for p in self.params]

        # for rmsprop
        cache = [theano.shared(np.ones_like(p.get_value())) for p in self.params]

        grads = T.grad(cost, self.params)
        updates = [ 
            (c, decay*c + (np.float32(1.0)-decay)*g*g) for c, g in zip(cache, grads) 
        ] + [
            (p, p + mu*dp - learning_rate*g / T.sqrt(c + eps)) for p, c, dp, g in zip(self.params, cache, dparams, grads)
        ] + [
            (dp, mu*dp - learning_rate*g / T.sqrt(c + eps)) for c, dp, g in zip(cache, dparams, grads)
        ]

        train_op = theano.function(
            inputs=[thX, thY],
            updates=updates
        )

        # step5. validation part
        # for evaluation and prediction
        pY_predict = self.forward_predict(thX)
        cost_predict = -T.mean(T.log(pY_predict[T.arange(thY.shape[0]), thY]))
        prediction = self.predict(thX)
        cost_predict_op = theano.function(inputs=[thX, thY], outputs=[cost_predict, prediction])

        n_batches = N // batch_sz
        costs = []
        for i in range(epochs):
            X, Y = shuffle(X, Y)
            for j in range(n_batches):
                Xbatch = X[j*batch_sz:(j*batch_sz+batch_sz)]
                Ybatch = Y[j*batch_sz:(j*batch_sz+batch_sz)]

                train_op(Xbatch, Ybatch)

                if j % 50 == 0:
                    c, p = cost_predict_op(Xvalid, Yvalid)
                    costs.append(c)
                    e = error_rate(Yvalid, p)
                    print("i:", i, "j:", j, "nb:", n_batches, "cost:", c, "error rate:", e)
        
        if show_fig:
            plt.plot(costs)
            plt.show()

    def forward_train(self, X):
        Z = X
        for h, p in zip(self.hidden_layers, self.dropout_rates[:-1]):    # hidden_layers 本來就不包含最後output layer的資訊
            mask = self.rng.binomial(n=1, p=p, size=Z.shape)
            Z = mask * Z
            Z = h.forward(Z)
        mask = self.rng.binomial(n=1, p=self.dropout_rates[-1], size=Z.shape)
        Z = mask * Z
        return T.nnet.softmax(Z.dot(self.W) + self.b)

    def forward_predict(self, X):
        Z = X
        for h, p in zip(self.hidden_layers, self.dropout_rates[:-1]):
            Z = h.forward(p * Z)
        return T.nnet.softmax((self.dropout_rates[-1] * Z).dot(self.W) + self.b)

    def predict(self, X):
        pY = self.forward_predict(X)
        return T.argmax(pY, axis=1)


def error_rate(p, t):
    return np.mean(p != t)


def relu(a):
    return a * (a > 0)


def main():
    # step 1: get the data and define all the usual variables
    Xtrain, Xtest, Ytrain, Ytest = get_normalized_data()

    ann = ANN([500, 300], [0.8, 0.5, 0.5])
    ann.fit(Xtrain, Ytrain, Xtest, Ytest, show_fig=True)


if __name__ == '__main__':
    main()
