import numpy as np
import theano
theano.config.floatX = 'float32'
# theano.config.warn_float64 = 'warn'
import theano.tensor as T 
import matplotlib.pyplot as plt

from sklearn.utils import shuffle
from util import relu, error_rate, getKaggleMNIST, init_weight

''' 
用 autoencoder 實踐 Greedy Layer-Wise pretraining
將上述效果功能加入DNN中，會使訓練效率大幅提升，cost 會下降得比較快
原理:
step1. 藉由Autoencoder 來 pre-training，將weights 調整到更好的初始值
step2. 訓練DNN
'''

def momentum_updates(cost, params, mu, learning_rate):
    
    mu = np.float32(mu)
    learning_rate = np.float32(learning_rate)
    # momentum changes
    dparams = [ theano.shared(np.zeros_like(p.get_value(), dtype=np.float32)) for p in params]

    # updates = []
    # grads = T.grad(cost, params)
    # for p, dp, g in zip(params, dparams, grads):
    #     dp_uptate = mu*dp - learning_rate*g
    #     p_update = p + dp_uptate

    #     updates.append((dp, dp_uptate))
    #     updates.append((p, p_update))

    grads = T.grad(cost, params)
    updates = [
        (p, p + mu*dp - learning_rate*g) for p, dp, g in zip(params, dparams, grads)
    ] + [
        (dp, mu*dp - learning_rate*g) for dp, g in zip( dparams, grads)
    ]
    return updates


class AutoEncoder(object):
    def __init__(self, M, an_id):
        self.M = M                 # neuron untis munber of hidden layer output
        self.id = an_id

    def fit(self, X, learning_rate=0.5, mu=0.99, epochs=1, batch_sz=100, show_fig=False):
        # cast to float
        mu = np.float32(mu)
        learning_rate = np.float32(learning_rate)
        X = X.astype(np.float32)

        N, D = X.shape
        n_batches = N // batch_sz


        # define shared (all weights in NN)
        W0 = init_weight((D, self.M))
        self.W = theano.shared(W0, 'W_%s' % self.id)
        self.bh = theano.shared(np.zeros(self.M, dtype=np.float32), 'bh_%s' % self.id)    # 重大發現阿，假如要固定型別 TensorType(float32, matrix)，也要記得將np.array物件的內部型別轉成np.float32
        self.bo = theano.shared(np.zeros(D, dtype=np.float32), 'bo_%s' % self.id)
        self.params = [self.W, self.bh, self.bo]    # keep tracking all parameters
        self.forward_params = [self.W, self.bh]

        self.dW = theano.shared(np.zeros(W0.shape, dtype=np.float32), 'dW_%s' % self.id)
        self.dbh = theano.shared(np.zeros(self.M, dtype=np.float32), 'dbh_%s' % self.id)
        self.dbo =  theano.shared(np.zeros(D, dtype=np.float32), 'dbo_%s' % self.id)
        self.dparams = [self.dW, self.dbh, self.dbo]
        self.forward_dparams = [self.dW, self.dbh]

        # define matrix (training data)
        X_in = T.matrix('X_%s' % self.id, dtype='float32')
        X_hat = self.forward_output(X_in)

        # attach it to the object so it can be used later
        # must be sigmoid because the output is also a sigmoid
        H = T.nnet.sigmoid(X_in.dot(self.W) + self.bh)    # define a hidden layer operation as a theano function
        # 取出中間hidden layer 的實際數值(作圖/ DNN訓練用)，這個 function 的輸入是 numpy.array ，輸出是 numpy.array
        self.hidden_op = theano.function(
            inputs=[X_in], 
            outputs=H,
        )

        # save this for later so we can call it to
        # create reconstructions of input        
        self.predict = theano.function(
            inputs=[X_in], 
            outputs=X_hat, 
        )

        # cost = ((X_in - X_hat) * (X_in - X_hat)).sum() / N  # squared error
        cost = -(X_in * T.log(X_hat) + (1 - X_in) * T.log(1 - X_hat)).sum() / N  # cross entropy
        cost_op = theano.function(
            inputs=[X_in], 
            outputs=cost
        )

        # updates = [
        #     (p, p + mu*dp - learning_rate*T.grad(cost, p)) for p, dp in zip(self.params, self.dparams)
        # ] + [
        #     (dp, mu*dp - learning_rate*T.grad(cost, p)) for p, dp, in zip(self.params, self.dparams)
        # ]

        updates = momentum_updates(cost, self.params, mu, learning_rate)
        train_op = theano.function(
            inputs=[X_in], 
            updates=updates,
        )

        costs = []
        print('training autoencoder: %s' % self.id)
        print('epochs to do:', epochs)
        for i in range(epochs):
            print('epoch:', i)
            X = shuffle(X)
            for j in range(n_batches):
                batch = X[j*batch_sz:(j*batch_sz + batch_sz), ]
                train_op(batch)
                the_cost = cost_op(batch)
                costs.append(the_cost)
                if j % 10 ==0:
                    print('j / n_batches', j, '/', n_batches, 'cost:', the_cost)

        if show_fig:
            plt.plot(costs)
            plt.show()

    def forward_hidden(self, X):
        # self.W / self.bh的型別是 theano.shared 耶
        # 這裡就不用利用theano.function 做計算了嗎????  沒錯，這裡是expression的一部分
        Z = T.nnet.sigmoid(X.dot(self.W) + self.bh)   
        return Z 

    def forward_output(self, X):
        Z = self.forward_hidden(X)
        return T.nnet.sigmoid(Z.dot(self.W.T) +  self.bo)
    

class DNN(object):
    def __init__(self, hidden_layer_sizes, UnsupervisedModel=AutoEncoder):
        self.hidden_layers = []
        
        # 建立 AutoEncoder 物件存到 hidden_layers 備著
        count = 0
        for M in hidden_layer_sizes:
            ae = UnsupervisedModel(M, count)
            self.hidden_layers.append(ae)
            count += 1

    def fit(self, X, Y, Xtest, Ytest, 
        pretrain=True, learning_rate=0.1, mu=0.99, reg=0.0, epochs=1, batch_sz=100):
        # cast to float32
        learning_rate = np.float32(learning_rate)
        mu = np.float32(mu)
        reg = np.float32(reg)

        # 選擇要不要 train AutoEncoder 物件
        pretrain_epochs = 2
        if not pretrain:
            pretrain_epochs = 0   # 假如這裡epoch=0 就只會initialize weights 但不做training

        # training AutoEncoder 物件
        current_input = X
        for ae in self.hidden_layers:
            ae.fit(current_input, epochs=pretrain_epochs)
            current_input = ae.hidden_op(current_input)

        # initialize logistic regression layer (最後一層)
        N = len(Y)
        K = len(set(Y))
        W0 = init_weight((self.hidden_layers[-1].M, K))
        self.W = theano.shared(W0, 'W_logreg')
        self.b = theano.shared(np.zeros(K, dtype=np.float32), 'b_logreg')
        self.params = [self.W, self.b]
        for ae in self.hidden_layers:
            # self.params.append(ae.forward_params)
            self.params += ae.forward_params

        self.dW = theano.shared(np.zeros(W0.shape, dtype=np.float32), 'dW_logreg')
        self.db = theano.shared(np.zeros(K, dtype=np.float32), 'db_logreg')
        self.dparams = [self.dW, self.db]
        for ae in self.hidden_layers:
            # self.dparams.append(ae.forward_dparams)
            self.dparams += ae.forward_dparams
            

        X_in = T.matrix('X_in', dtype='float32')
        targets = T.ivector('Targets')    # 注意，這邊要在建立vector存量同時，就宣告存量型別(ivector)，如果不宣告，預設是float32
        pY = self.forward(X_in)

        reg_cost = T.sum([(p*p).sum() for p in self.params])
        cost = -T.mean( T.log(pY[T.arange(pY.shape[0]), targets]) ) + reg*reg_cost
        updates = [
            (p, p + mu*dp -learning_rate*T.grad(cost, p)) for p, dp in zip(self.params, self.dparams)
        ] + [
            (dp, mu*dp - learning_rate*T.grad(cost, p)) for p, dp, in zip(self.params, self.dparams)
        ]    # ..............................包含之前已Autoencoder pretrain過的每一層都要training
        train_op = theano.function(     
            inputs=[X_in, targets], 
            updates=updates
        )

        prediction = self.predict(X_in)
        cost_predict_op = theano.function(
            inputs=[X_in, targets], 
            outputs=[cost, prediction]
        )

        n_batches = N // batch_sz
        costs = []
        print("supervised training...")
        for i in range(epochs):
            print("epoch:", i)
            X, Y = shuffle(X, Y)
            for j in range(n_batches):
                Xbatch = X[j*batch_sz:(j*batch_sz + batch_sz)]
                Ybatch = Y[j*batch_sz:(j*batch_sz + batch_sz)]
                train_op(Xbatch, Ybatch)
                the_cost, the_prediction = cost_predict_op(Xtest, Ytest)
                error = error_rate(the_prediction, Ytest)
                print("j / n_batches:", j, "/", n_batches, "cost:", the_cost, "error:", error)
                costs.append(the_cost)
        plt.plot(costs)
        plt.show()

    def forward(self, X):
        current_input = X
        for ae in self.hidden_layers:
            current_input = ae.forward_hidden(current_input)
        
        # logistic layer
        Y = T.nnet.softmax(current_input.dot(self.W) + self.b)
        return Y

    def predict(self, X):
        return T.argmax(self.forward(X), axis=1)


def main():
    Xtrain, Ytrain, Xtest, Ytest = getKaggleMNIST()
    dnn = DNN([1000, 750, 500])
    dnn.fit(Xtrain, Ytrain, Xtest, Ytest, pretrain=True, epochs=3)
    # vs
    # dnn = DNN([1000, 750, 500])
    # dnn.fit(Xtrain, Ytrain, Xtest, Ytest, pretrain=False, epochs=3)
    # note: try training the head only too! what does that mean?

    
def test_single_autoencoder():
    Xtrain, Ytrain, Xtest, Ytest = getKaggleMNIST()

    autoencoder  = AutoEncoder(300, 0)
    autoencoder.fit(Xtrain, learning_rate=0.5, epochs=3, show_fig=True) 
    # 假如這裡epoch=0 就只會initialize weights 但不做training

    done = False
    while not done:
        i = np.random.choice(len(Xtest))
        x = Xtest[i]  # 這裡取出來的x.shape = (784,), 是一個vector，但 input of predict() need to be a matrix (1, 784)
        y = autoencoder.predict([x])   # 這裡為什麼要寫 [x]? 這樣才可以把shape轉成 (1, 784) 作為input
        plt.subplot(1, 2, 1)
        plt.imshow(x.reshape(28, 28), cmap='gray')
        plt.title('Original')

        plt.subplot(1, 2, 2)
        plt.imshow(y.reshape(28, 28), cmap='gray')
        plt.title('Reconstructed')

        plt.show()
        
        ans = input("Generate another?")
        if ans and ans[0] in ('n' or 'N'):    # 第一次看到的寫法  in ('n' or 'N') ， 還不太懂
            done = True


if __name__ == '__main__':
    # test_single_autoencoder()
    main()

