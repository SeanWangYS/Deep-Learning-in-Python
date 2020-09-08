import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

from util import get_normalized_data
from sklearn.utils import shuffle

import warnings 
warnings.filterwarnings('ignore')

'''
這篇運用tensorflow，寫了一個ANN的api
用上的技術有: L2, momentum, RMSProm, mini batch gradient descent
可建立多層hidden layer 的 class
'''

class HiddenLayer(object):
    # HiddenLayer object, entity class, it includes weight variabels of each layer
    def __init__(self, M1, M2, an_id):
        self.id = an_id
        self.M1 = M1
        self.M2 = M2
        W, b = init_weight_and_bias(M1, M2)
        self.W = tf.Variable(W.astype(np.float32))
        self.b = tf.Variable(b.astype(np.float32))
        self.params = [self.W, self.b]

    def forward(self, X):
        return tf.nn.relu(tf.matmul(X, self.W) + self.b)

class ANN(object):
    def __init__(self, hidden_layer_size):
        self.hidden_layer_size = hidden_layer_size

    def fit(self, X, Y, Xvalid, Yvalid, learning_rate=1e-4, mu=0.99, decay=0.999, reg=1e-3, epoch=10, batch_sz=100, show_fig=False):
        # step1. get data
        X, Y = shuffle(X, Y)
        X = X.astype(np.float32)
        Y = Y.astype(np.float32)
        Xvalid = Xvalid.astype(np.float32)
        Yvalid = Yvalid.astype(np.float32)

        # step1.1 initialize each layer and themselves parameters(with tf.Variable) of NN and keep them in a list
        N, D = X.shape
        M1 = D
        K = len(set(Y))
        self.hidden_layers = []  # for saving HiddenLayer object
        count = 0
        for M2 in self.hidden_layer_size:    # 這邊做出順序從第一層~ 最後一層 hidden layer 的HiddenLayer object
            h = HiddenLayer(M1, M2, count)
            self.hidden_layers.append(h)
            M1 = M2
            count += 1

        W, b = init_weight_and_bias(M1, K)   # 最後輸出層的 weight
        self.W = tf.Variable(W.astype(np.float32))
        self.b = tf.Variable(b.astype(np.float32))

        # collect all the parameters that we are going to use grediant descent
        self.params = [self.W, self.b]        # 先放最後一層output layer 進 params，在tf這個參數是為了L2計算才設計的，在theano則是為了寫所有的計算式(edge)
        for layer in self.hidden_layers:
            self.params += layer.params       # 接著應該是照著 hidden lyer1 , hidden layer2, 的順序放進去params

        # setp1.2 tf.placeholder
        inputs = tf.placeholder(tf.float32, shape=(None, D), name="inputs")
        labels = tf.placeholder(tf.int64, shape=(None,), name="inputs")

        # step2. model
        logits = self.forward(inputs)    # 最後不經過softmax喔，也不通過其他的activation function(因為tf 就是這樣要求的)

        # step3. cost function
        rcost = reg*sum([tf.nn.l2_loss(p) for p in self.params])
        cost = tf.reduce_mean(
            tf.nn.sparse_softmax_cross_entropy_with_logits(
                logits=logits,
                labels=labels
            )
        ) + rcost

        prediction_op = self.predict(inputs)

        # step4. solver
        train_op = tf.train.RMSPropOptimizer(learning_rate=learning_rate, momentum=mu, decay=decay).minimize(cost)
        # train_op = tf.train.MomentumOptimizer(lr, momentum=mu).minimize(cost)
        # train_op = tf.train.AdamOptimizer(lr).minimize(cost)     

        init = tf.global_variables_initializer()

        n_batches = N // batch_sz
        costs = []
        with tf.Session() as sess:
            sess.run(init)

            for i in range(epoch):
                for j in range(n_batches):
                    Xbatch = X[j*batch_sz:(j+1)*batch_sz,]
                    Ybatch = Y[j*batch_sz:(j+1)*batch_sz,]

                    sess.run(train_op, feed_dict={inputs:Xbatch, labels:Ybatch})
                    if j % 50 == 0:
                        cost_val = sess.run(cost, feed_dict={inputs: Xvalid, labels: Yvalid})
                        costs.append(cost_val)
                        preds = sess.run(prediction_op, feed_dict={inputs: Xvalid})
                        err = error_rate(Yvalid, preds)
                        print("i:", i, "j:", j, "nb:", n_batches, "cost:", cost_val, "error rate:", err)

        if show_fig:
            plt.plot(costs)
            plt.show()           

    def forward(self, X):
        Z = X
        for layer in self.hidden_layers:
            Z = layer.forward(Z)
        return tf.matmul(Z, self.W) + self.b

    def predict(self, X):
        act = self.forward(X)
        return tf.argmax(act, axis=1)



def init_weight_and_bias(M1, M2):
    W = np.random.randn(M1, M2) / np.sqrt(M1 + M2)
    b = np.zeros(M2)
    return W, b

def error_rate(targets, predictions):
    return np.mean(targets != predictions)

def main():
    Xtrain, Xvalid, Ytrain, Yvalid = get_normalized_data()
    model = ANN([300, 100])
    model.fit(Xtrain, Ytrain, Xvalid, Yvalid, show_fig=True)

if __name__ == '__main__':
    main()