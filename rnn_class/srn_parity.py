import numpy as np 
import theano 
import theano.tensor as T 
import matplotlib.pyplot as plt 

from sklearn.utils import shuffle 
from util import init_weight, all_parity_pairs, all_parity_pairs_with_sequence_labels

'''
第一個RNN 模型實例
訓練資料是程式產生的parity訓練集

學習情境:
T labels for sequence of length T*
剛好這裡的訓練集，T為定值
'''

class SimpleRNN(object):
    def __init__(self, M):
        self.M = M  # hidden layer size

    def fit(self, X, Y, learning_rate=10e-1, mu=0.99, reg=1.0, activation=T.tanh, epochs=100, show_fig=False):
        D = X[0].shape[1]     # X is of size N x T(n) x D 
        # print('D:', D)    # D = 1
        K = len(set(Y.flatten()))   #　Y is of size N x T(n)
        # print('K:', K)  # K = 2
        N = len(Y)
        M = self.M
        self.f = activation 

        # initialize weights
        Wx = init_weight(D, M)
        Wh = init_weight(M, M)
        bh = np.zeros(M)
        h0 = np.zeros(M)    # initialize hidden layer activation at t=0
        Wo = init_weight(M, K)
        bo = np.zeros(K)

        # make them theano shared
        self.Wx = theano.shared(Wx)
        self.Wh = theano.shared(Wh)
        self.bh = theano.shared(bh)
        self.h0 = theano.shared(h0)
        self.Wo = theano.shared(Wo)
        self.bo = theano.shared(bo)
        self.params = [self.Wx, self.Wh, self.bh, self.h0, self.Wo, self.bo]

        thX = T.fmatrix('X')    # will be a TxD matrix
        thY = T.ivector('Y')

        def recurrence(x_t, h_t1):    # 這邊f的用法，請教育叡
            # retruns h(t), y(t)
            h_t = self.f(x_t.dot(self.Wx) + h_t1.dot(self.Wh) + self.bh)  # h_t => M 維度的vector (M, )
            y_t = T.nnet.softmax(h_t.dot(self.Wo) + self.bo)              # y_t => K 維度的vector (K, )
            return h_t, y_t  

        ## 針對單一序列做一次循環，只走完一圈時步 (那應該可一次跑多個序列吧???)
        [h, y], _ = theano.scan(
            fn=recurrence,
            outputs_info=[self.h0, None],  # 原本的output=[h_t, y_t] ，只需要對 h_t 賦值，第二個就補上None
            sequences=thX,                # sequence shape=(12, 1)=(T, D) singla element in sequence=(1, ) 
            n_steps=thX.shape[0]          # time step = 12 ，走過一個完整的時步循環
        )
        # print('y.shape:', y.shape)
        py_x = y[:, 0, :]  # y.shape=(12, 1, 2) so we take first dimension and last dimension. py_x.shape = (T, K)
        # 所以 y.shape = (T, 1, K)， 但我以為會是(T, K) 
        # 看起來，上面的y_t 可能是(1, K)matrix ，所以最後的y才會變成(T, 1 ,K) 要再確認??
        prediction = T.argmax(py_x, axis=1)

        ## 走過一個完整時步循環後 計算的cost
        cost = -T.mean(T.log(py_x[T.arange(thY.shape[0]), thY])) 
        grads = T.grad(cost, self.params)
        dparams = [theano.shared(p.get_value()*0) for p in self.params]

        ## 走過一個完整的時步循環後，做的加權更新
        updates = [   
            (p, p + mu*dp - learning_rate*g) for p, dp, g in zip(self.params, dparams, grads)
        ] + [
            (dp, mu*dp - learning_rate*g) for dp, g in zip(dparams, grads)
        ]

        self.predict_op = theano.function(inputs=[thX], outputs=prediction)
        self.train_op = theano.function(
            inputs=[thX, thY], 
            updates=updates, 
            outputs=[cost, prediction, y],
        )

        costs = []
        for i in range(epochs):
            X, Y  = shuffle(X, Y)  # do one sequence at a time (batch_sz = 1)
            n_correct = 0
            cost = 0
            for j in range(N): # 每次只跑一筆sample喔，這裡不用mini-batch gradient descent
                c, p, rout = self.train_op(X[j], Y[j])     # X[j] 的形狀是 TxD 
                cost += c
                if p[-1] == Y[j, -1]:
                    n_correct += 1
            # print('shape y:', rout.shape)
            print('i:', i, 'cost:', cost, 'classification rate:', (float(n_correct)/N))
            costs.append(cost)
            if n_correct == N:
                break

        if show_fig:
            plt.plot(costs)
            plt.show()


def parity(B=12, learning_rate=1e-4, epochs=20):
    X, Y = all_parity_pairs_with_sequence_labels(B)

    rnn = SimpleRNN(20)
    rnn.fit(X, Y, learning_rate=learning_rate, epochs=epochs, activation=T.nnet.relu, show_fig=True)

        
if __name__ == "__main__":
    parity()