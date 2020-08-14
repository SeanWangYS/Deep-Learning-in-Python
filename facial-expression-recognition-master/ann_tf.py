import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

from util import getData , getBinaryData, y2indicator, relu, error_rate, init_weight_and_bias
from sklearn.utils import shuffle

import warnings
warnings.filterwarnings('ignore')

'''
classification model

這篇運用 tensorflow，寫了一個ANN的api
用上的技術有: L2, momentum, RMSProm, mini batch gradient descent
可建立多層hidden layer
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
        
    def fit(self, X, Y, Xvalid, Yvalid, learning_rate=1e-3, mu=0.99, decay=0.999, reg=1e-3, epoches=10, batch_sz=100, show_fig=False):
        # step1. get data
        X, Y = shuffle(X, Y)
        X = X.astype(np.float32)
        Y = y2indicator(Y).astype(np.int32)
        Xvalid = Xvalid.astype(np.float32)
        Yvalid_vector = Yvalid.astype(np.int32)
        Yvalid = y2indicator(Yvalid).astype(np.int32)
        
        # step1.1 initialize each layer and parameters(with tf.Variable) of NN and keep them in a list
        N, D = X.shape
        M1 = D
        K = Y.shape[1] 
        self.hidden_layers = [] # for saving HiddenLayer object
        count = 0
        for M2 in self.hidden_layer_size:  # 這邊做出第一層~ 倒數第二層 hidden layer 的HiddenLayer object
            h = HiddenLayer(M1, M2, count)
            self.hidden_layers.append(h)
            M1 = M2
            count += 1

        W, b = init_weight_and_bias(M1, K)   # 最後輸出層的 weight
        self.W = tf.Variable(W.astype(np.float32))
        self.b = tf.Variable(b.astype(np.float32))
        
        # collect all the parameters that we are going to use grediant descent
        self.params = [self.W, self.b]
        for layer in self.hidden_layers:
            self.params += layer.params
        
        # step1.2 tf.palceholder
        tfX = tf.placeholder(tf.float32, shape=(None, D), name="X")
        tfT = tf.placeholder(tf.float32, shape=(None, K), name="T")
        
        # step2. model
        act = self.forward(tfX)   # 最後不經過softmax喔，也不通過其他的activation fun(因為tf 就是這樣要求的)
        
        # step3. cost function
        rcost = reg*sum([tf.nn.l2_loss(p) for p in self.params])
        cost = tf.reduce_mean(
            tf.nn.softmax_cross_entropy_with_logits_v2(
                logits=act,
                labels=tfT
            )
        ) + rcost
        
        prediction_op = self.predict(tfX)
        
        # step4. solver
        traiin_op = tf.train.RMSPropOptimizer(learning_rate=learning_rate, momentum=mu, decay=decay).minimize(cost)
        
        init = tf.global_variables_initializer()
        
        n_batches = N // batch_sz
        costs = []
        with tf.Session() as sess:
            sess.run(init)
            
            for i in range(epoches):
                for j in range(n_batches):
                    Xbatch = X[j*batch_sz:(j+1)*batch_sz,]
                    Ybatch = Y[j*batch_sz:(j+1)*batch_sz,]
                    
                    sess.run(traiin_op, feed_dict={tfX:Xbatch, tfT:Ybatch})
                    if j % 50 == 0:
                        cost_val = sess.run(cost, feed_dict={tfX: Xvalid, tfT: Yvalid})
                        costs.append(cost_val)
                        preds = sess.run(prediction_op, feed_dict={tfX: Xvalid})
                        err = error_rate(Yvalid_vector, preds)
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
        
        
def main():
    Xtrain, Ytrain, Xvalid, Yvalid = getData()
    model = ANN([2000, 1000, 500])
    model.fit(Xtrain, Ytrain, Xvalid, Yvalid, show_fig=False)

if __name__ == '__main__':
    main()