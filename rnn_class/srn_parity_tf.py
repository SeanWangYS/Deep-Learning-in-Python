import tensorflow as tf 
import numpy as np 
import matplotlib.pyplot as plt 

########## This works for TensorFlow v1.0 ##############
from tensorflow.contrib.rnn import static_rnn as get_rnn_output
from tensorflow.contrib.rnn import BasicRNNCell, GRUCell
########################################################

from sklearn.utils import shuffle 
from util import init_weight, all_parity_pairs_with_sequence_labels, all_parity_pairs

'''
RNN 模型實例
使用tensorflow 玩 parity資料集

過去我們習慣的資料shape = NxTxD  ，每一筆sample T(sequence lenght) 非定植
在TF中，tensorflow works with batches naturally
在TF中，會希望資料shape = TxNxD, (剛好這裡的訓練集，T為定值)
(tensor flow requires all your sequences to have the same length.)

所以在這支程式裡，從 all_parity_pairs_with_sequence_labels 取得的資料shape，需要做調整

學習情境:
T labels for sequence of length T*
剛好這裡的訓練集，T為定值

Note:
在此架構下，activation function 用 sigmoid 
1. 效果居然比 relu 好很多!!!!!
2. classification rate 會從 0.7 直接跳到 1.0
3. batch_sz 用 用full batch

根據以上觀察，感覺有點奇怪，是不是因為資料太簡單，讓他可以學到一個捷徑去fit 資料 pattern
'''

def x2sequence(x, T, D, batch_sz):
    # Permuting(排列) batch size and n_steps
    x = tf.transpose(x, (1, 0, 2))
    # Reshaping to (n_steps*batch_size, n_input)
    x = tf.reshape(x, (T*batch_sz, D))
    # Split to get a list of 'n_steps' tensors of shape (batch_size, n_input)  => T x batch_sz x D
    x = tf.split(x, T)
    print("type(x):", type(x))
    return x 

class SimpleRNN(object):
    def __init__(self, M):
        self.M = M 

    def fit(self, X, Y, batch_sz=20, learning_rate=0.1, mu=0.9, activation=tf.nn.sigmoid, epochs=100, show_fig=False):
        N, T, D = X.shape 
        K = len(set(Y.flatten()))
        M = self.M 
        self.f = activation

        # init_weights 
        # note: Wx, Wh, bh are all part of the RNN unit and will be created
        #       by BasicRNNCell        
        Wo = init_weight(M, K).astype(np.float32)
        bo = np.zeros(K, dtype=np.float32)

        # make them tf variables
        self.Wo = tf.Variable(Wo)
        self.bo = tf.Variable(bo)

        # tf Graph input 
        tfX = tf.placeholder(tf.float32, shape=(batch_sz, T, D), name='inputs')
        tfY = tf.placeholder(tf.int64, shape=(batch_sz, T), name='targets')

        ######################## 建模型架構  #############################
        # turn tfX into a sequence, e.g. T tensors all of size (batch_sz, D)
        sequenceX = x2sequence(tfX, T, D, batch_sz)   # input data，每次都會是一個batch_sz 的份量拉

        # create the simple rnn unit
        rnn_unit = BasicRNNCell(num_units=self.M, activation=self.f)

        # Get rnn cell output
        outputs, states = get_rnn_output(rnn_unit, sequenceX, dtype=tf.float32)

        # outputs are now of size (T, batch_sz, M)
        # so make it (batch_sz, T, M)
        outputs = tf.transpose(outputs, (1, 0, 2))
        outputs = tf.reshape(outputs, (T*batch_sz, M))   # 注意，刻意將output做成這樣的shape，targets 也要與之配合

        # Linear activation, using rnn inner loop last output
        logits = tf.matmul(outputs, self.Wo) + self.bo 

        #####################################################################

        predict_op = tf.argmax(logits, axis=1)
        targets = tf.reshape(tfY, (T*batch_sz,))     # targets 每次也都會是一個batch_sz的份量，跟input data 一樣
    
        cost_op = tf.reduce_mean(
            tf.nn.sparse_softmax_cross_entropy_with_logits(
                logits=logits,
                labels=targets
            )
        )
        train_op = tf.train.MomentumOptimizer(learning_rate, momentum=mu).minimize(cost_op)

        costs = []
        n_batches = N // batch_sz 

        init = tf.global_variables_initializer()
        with tf.Session() as session:
            session.run(init)
            for i in range(epochs):
                X, Y = shuffle(X, Y)
                n_correct = 0
                cost = 0
                for j in range(n_batches):
                    Xbatch = X[j*batch_sz:(j+1)*batch_sz]
                    Ybatch = Y[j*batch_sz:(j+1)*batch_sz]

                    _, c, p = session.run([train_op, cost_op, predict_op], feed_dict={tfX: Xbatch, tfY: Ybatch})
                    cost += c 
                    for b in range(batch_sz):
                        idx = (b + 1)*T - 1   # 只對每個sequence 的 last of prediction 做比較
                        n_correct += (p[idx] == Ybatch[b][-1])  # 這個技巧很帥
                if i % 10 == 0:
                    print("i:", i, "cost:", cost, "classification rate:", (float(n_correct)/N))
                if n_correct == N:
                    print("i:", i, "cost:", cost, "classification rate:", (float(n_correct)/N))
                    break
                costs.append(cost)

        if show_fig:
            plt.plot(costs)
            plt.show()

def parity(B=12, learning_rate=1., epochs=1000):    # 很少看到learning rate 用 1 
    X, Y = all_parity_pairs_with_sequence_labels(B)
    print('X.shape:', X.shape)
    print('Y.shape:', Y.shape)
    
    rnn = SimpleRNN(20)
    rnn.fit(X, Y, 
        batch_sz=len(Y),   #????用full batch???
        learning_rate=learning_rate, 
        epochs=epochs, 
        activation=tf.nn.sigmoid, 
        show_fig=True
    )

if __name__ == '__main__':
    parity()

