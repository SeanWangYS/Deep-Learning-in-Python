import numpy as np
import theano
theano.config.floatX = 'float32'
import theano.tensor as T
import matplotlib.pyplot as plt
from theano.tensor.shared_randomstreams import RandomStreams
from util import get_normalized_data
from sklearn.utils import shuffle
'''
這篇運用theano，寫了一個ANN的api
用上的技術有: L2, momentum, RMSProm, mini batch gradient descent
可建立多層hidden layer 的 class
'''

class HiddenLayer(object):
    # HiddenLayer 物件，entity物件，建立單一hidden layer 的參數初始化後的物件
    def __init__(self, M1, M2, an_id):
        self.id = an_id
        self.M1 = M1
        self.M2 = M2
        W, b = init_weight_and_bias(M1, M2)
        self.W = theano.shared(W, 'W_%s' % self.id)
        self.b = theano.shared(b, 'b_%s' % self.id)
        self.params = [self.W, self.b]    # keep track all parameter，因為 theano 的solver要自己寫，這邊設計了params來存放所有weights (在tensorflow，這個參數就非必要)

    def forward(self, X):
        # return relu(X.dot(self.W) + self.b)
        return T.nnet.relu(X.dot(self.W) + self.b)


class ANN(object):
    def __init__(self, hidden_layer_sizes):
        self.hidden_layer_sizes = hidden_layer_sizes

    def fit(self, X, Y, Xvalid, Yvalid, learning_rate=1e-3, mu=0.99, decay=0.999, reg=1e-3, eps=1e-8, epochs=10, batch_sz=100, show_fig=False):
        # # 特地轉換所有參數的型別到float32，要不然就會有error，但其他支theano 的 ANN class卻不用這樣做，不知道為什麼??
        learning_rate = np.float32(learning_rate)
        mu = np.float32(mu)
        decay = np.float32(decay)
        reg = np.float32(reg)
   

        # step1 get the data 
        X, Y =  shuffle(X, Y)
        X = X.astype(np.float32)  # for being avalibel for GPU
        Y = Y.astype(np.int32)
        Xvalid = Xvalid.astype(np.float32)    
        Yvalid = Yvalid.astype(np.int32)

        # initialize each layer and parameters of NN
        N, D = X.shape
        K = len(set(Y))
        self.hidden_layers = []  # 這個list用來放HiddenLyer物件
        M1 = D
        count = 0
        for M2 in self.hidden_layer_sizes:   # 建立ANN物件時輸入的參數，轉成在ANN物件內部建立實體HiddenLyer物件
            h = HiddenLayer(M1, M2, count)
            self.hidden_layers.append(h)
            M1 = M2
            count += 1

        # initialize logistic regression layer
        W, b = init_weight_and_bias(M1, K)   # 最後一層output物件 ( the last logist regression layer )
        self.W = theano.shared(W, 'W_logreg')    
        self.b = theano.shared(b, 'b_logreg')

        # collect all the parameters that we are going to use grediant descent
        self.params = [self.W, self.b]   # 先把最後一層output layer放進去
        for h in self.hidden_layers:
            self.params += h.params   # 應該是照著 hidden lyer1 , hidden layer2, 的順序放進去
            
        # for momentum, we need to create zero matrix for each layer
        dparams = [ theano.shared(np.zeros_like(p.get_value(), dtype=np.float32)) for p in self.params]
        
        # for rmsprop, we need create cache 
        cache = [ theano.shared(np.ones_like(p.get_value(), dtype=np.float32)) for p in self.params]
        
        # step2. model
        # theano variabels
        thX = T.fmatrix('X')    # data input (matrix)
        thY = T.ivector('Y')    # target ( vector)
        pY = self.forward(thX)  # forward的 output，出來的型別是matrix
        
        # step3. cost function
        # define theano's computation grath
        rcost = reg*T.sum([(p*p).sum() for p in self.params])
        cost = -T.mean(T.log(pY[T.arange(thY.shape[0]), thY])) + rcost    # thY there is a vector, and we do not need y2indicator in this case(算是特殊寫法)

        # step4. solver
        # define theano's operations
        grads = T.grad(cost, self.params)
        updates = [ 
            (c, decay*c + (np.float32(1.0)-decay)*g*g) for c, g in zip(cache, grads) 
        ] + [
            (p, p + mu*dp - learning_rate*g / T.sqrt(c + eps)) for p, c, dp, g in zip(self.params, cache, dparams, grads)
        ] + [
            (dp, mu*dp - learning_rate*g / T.sqrt(c + eps)) for c, dp, g in zip(cache, dparams, grads)
        ]
        
        # updates = [ 
        #     (c, decay*c + (np.float32(1.0)-decay)*T.grad(cost, Dp)*T.grad(cost, p)) for p, c in zip(self.params, cache) 
        # ] + [
        #     (p, p + mu*dp - learning_rate*T.grad(cost, p) / T.sqrt(c + eps)) for p, c, dp in zip(self.params, cache, dparams)
        # ] + [
        #     (dp, mu*dp - learning_rate*T.grad(cost, p) / T.sqrt(c + eps)) for p, c, dp in zip(self.params, cache, dparams)
        # ]

        train_op = theano.function(
            inputs=[thX, thY],
            updates=updates
        )
        
        # for evaluation and prediction
        prediction = self.th_prediction(thX)
        cost_prediction_op = theano.function(
            inputs=[thX, thY], 
            outputs=[cost, prediction]
        )
        
        n_batches = N // batch_sz
        costs = []
        for i in range(epochs):
            for j in range(n_batches):
                Xbatch = X[j*batch_sz:(j*batch_sz + batch_sz),]
                Ybatch = Y[j*batch_sz:(j*batch_sz + batch_sz),]
                
                train_op(Xbatch, Ybatch)
                
                if j % 20 == 0:
                    cost_valid, preds_valid = cost_prediction_op(Xvalid, Yvalid)
                    costs.append(cost_valid)
                    e = error_rate(Yvalid, preds_valid)
                    print( "i:", i, " j,", j, " nb:", n_batches, " cost:", cost_valid, " error_rate:", e)
       
        
        if show_fig:
            plt.plot(costs)
            plt.show()

    def forward(self, X):
        Z = X
        for h in self.hidden_layers:
            Z = h.forward(Z)
        return T.nnet.softmax(Z.dot(self.W) + self.b)
    
    def th_prediction(self, X):
        pY = self.forward(X)
        return T.argmax(pY, axis=1)


def error_rate(p, t):
    return np.mean(p != t)

def relu(a):
    return a * (a>0)

def init_weight_and_bias(M1, M2):
    W = np.random.randn(M1, M2) / np.sqrt(M1 + M2)
    b = np.zeros(M2)
    return W.astype(np.float32), b.astype(np.float32)

def main():
    Xtrain, Xvalid, Ytrain, Yvalid = get_normalized_data()

    model = ANN([300,100])
    model.fit(Xtrain, Ytrain, Xvalid, Yvalid, show_fig=True)



if __name__ == '__main__':
    main()
