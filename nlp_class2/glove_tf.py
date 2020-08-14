import numpy as np 
import json 
import tensorflow as tf
import matplotlib.pyplot as plt

from datetime import datetime
from sklearn.utils import shuffle

import os
import sys

# sys.path.append(os.path.abspath('.'))    # os.path.abspath('.') 等同於 os.getcwd() 都是取得當前資料夾絕對路徑

# 因為想把這隻程式的輸出檔案存在本資料夾，因此這在這支程式，程式啟動的目錄位置改到nlp_calss2資料夾，
# 因此在nlp_class2啟動python時，要指定上一層的絕對路徑，才能import下面檔案
sys.path.append(os.path.abspath('..'))
from rnn_class.brown import get_sentences_with_word2idx_limit_vocab, get_sentences_with_word2idx
from nlp_class2.util import find_analogies

class Glove(object):
    def __init__(self, D, V, context_sz):
        self.D = D
        self.V = V
        self.context_sz = context_sz 

    def fit(self, sentences, cc_matrix=None, learning_rate=1e-4, reg=0.1, xmax=100, alpha=0.75, epochs=10):
        # build co-occurrence matrix
        # paper calls it X, so we will call it X, instead of calling
        # the training data X
        # TODO: would it be better to use a sparse matrix?
        t0 = datetime.now()
        V = self.V
        D = self.D
        
        if not os.path.exists(cc_matrix):
            X = np.zeros((V, V))
            N = len(sentences)
            print("number of sentences to process:", N)
            it = 0
            for sentence in sentences:
                it += 1
                if it % 10000 == 0:
                    print("process", it, "/", N)
                n = len(sentence)
                
                # loop through every tokens in the sentence 
                # 以sentence中第i 個字為基準，向左/向右 計算分數(context distance)
                for i in range(n):  
                    # i is not the word index!!!
                    # j is not the word index!!!
                    # i just points to which element of the sequence (sentence) we're looking at
                    wi = sentence[i]

                    # 對第i 個字而言，句子的邊界距離有多遠
                    start = max(0, i - self.context_sz)
                    end = min(n, i + self.context_sz)   

                    # we can either choose only one side as context, or both
                    # here we are doing both

                    # make sure "start" and "end" tokens are part of some context
                    # otherwise their f(X) will be 0 (denominator in bias update)
                    
                    # 如果 start token 還在 context 考慮的範圍內，那便要對 start token 做更新
                    if i - self.context_sz < 0:  #  if I minus contex size is less than zero that means at the start tokin is part of the context
                        points = 1.0 / (i + 1)
                        X[wi, 0] += points
                        X[0, wi] += points
                    # 如果 end token 還在 context 考慮的範圍內，那便要對 end token 做更新
                    if i + self.context_sz > n:  # end tokin is part of the context
                        points = 1.0 / (n - i)
                        X[wi, 1] += points
                        X[1, wi] += points

                    # left side
                    for j in range(start, i):
                        wj = sentence[j]
                        points  = 1.0 / (i - j) 
                        X[wi, wj] += points
                        X[wj, wi] += points

                    # right side
                    for j in range(i + 1, end):
                        wj = sentence[j]
                        points = 1.0 / (j - i)
                        X[wi, wj] += points
                        X[wj, wj] += points
            
            # save the cc matrix because it takes forever to create
            np.save(cc_matrix, X)
        else:
            X = np.load(cc_matrix)
        
        print("max in X:", X.max())


        # weighting (根據論文上的敘述，cost function 要再乘上一個權重)
        fX = np.zeros((V, V))
        fX[X < xmax] = (X[X < xmax] / float(xmax)) ** alpha
        fX[X >= xmax] = 1

        print("max in f(X):", fX.max())

        # target
        logX = np.log(X + 1)

        print('max in log(X):', logX.max())

        print('time to build co-occurrence matrix:', (datetime.now() - t0))

        # initialize weights
        W = np.random.randn(V, D) / np.sqrt(V + D)
        b = np.zeros(V)
        U = np.random.randn(V, D) / np.sqrt(V + D)
        c = np.zeros(V)
        mu = logX.mean()

        # initialize weights, inputs, targets placeholders
        tfW = tf.Variable(W.astype(np.float32))
        tfb = tf.Variable(b.reshape(V, 1).astype(np.float32))  # 這裡把b vector 改成2維矩陣，是因為之後計算delta比較方便
        tfU = tf.Variable(U.astype(np.float32))
        tfc = tf.Variable(c.reshape(1, V).astype(np.float32))
        tfLogX = tf.placeholder(tf.float32, shape=(V, V))
        tffX = tf.placeholder(tf.float32, shape=(V, V))

        delta = tf.matmul(tfW, tf.transpose(tfU)) + tfb + tfc + mu - tfLogX   # (prediction -  target)
        cost = tf.reduce_sum(tffX * delta * delta)
        regularized_cost = cost
        for param in (tfW, tfU):
            regularized_cost += reg*tf.reduce_sum(param*param)

        train_op = tf.train.MomentumOptimizer(
            learning_rate, 
            momentum=0.9
        ).minimize(regularized_cost)
        # train_op = tf.train.AdamOptimizer(1e-3).minimize(regularized_cost)
        init = tf.global_variables_initializer()
        session = tf.InteractiveSession()
        session.run(init)

        costs = []
        sentence_indexes = range(len(sentences))
        for epoch in range(epochs):
            c, _ = session.run((cost, train_op), feed_dict={tfLogX: logX, tffX: fX})
            print("epoch:", epoch, " cost:", c)
            costs.append(c)
        
        self.W, self.U = session.run([tfW, tfU])

        plt.plot(costs)
        plt.show()

    def save(self, fn):
        # function word_analogies expects a (V,D) matrx and a (D,V) matrix
        arrays = [self.W, self.U.T]
        np.savez(fn, *arrays)


def main(we_file, w2i_file, use_brown=True ,n_files=100):
    if use_brown:
        cc_matrix = 'cc_matrix_brown.npy'
    else:
        cc_matrix = 'cc_matrix_%s.npy' % n_files

    # hacky way of checking if we need to re-load the raw data or not
    # remember only the co-occurrence matirx is needed for training
    if os.path.exists(cc_matrix):
        with open(w2i_file) as f:
            word2idx = json.load(f)   # 載入就變成 dict 型別囉
        sentences = []  # dummy - we won't actually use it
    else:
        if use_brown:
            keep_words = set([
                'king', 'man', 'woman',
                'france', 'paris', 'london', 'rome', 'italy', 'britain', 'england',
                'french', 'english', 'japan', 'japanese', 'chinese', 'italian',
                'australia', 'australian', 'december', 'november', 'june',
                'january', 'february', 'march', 'april', 'may', 'july', 'august',
                'september', 'october',
            ])
            sentences, word2idx = get_sentences_with_word2idx_limit_vocab(n_vocab=5000, keep_words=keep_words)
        else:
            # sentences, word2idx = get_wikipedia_data(n_files=n_files, n_vocab=2000)   # 我根本就不會用到這個
            pass

        with open(w2i_file, 'w') as f:
            json.dump(word2idx, f)

    V = len(word2idx)
    model = Glove(100, V, 10)
    model.fit(sentences=sentences, cc_matrix=cc_matrix, epochs=200)
    model.save(we_file)


if __name__ == "__main__":
    we = 'glove_model_50.npz'
    w2i = 'glove_word2idx_50.json'
    main(we, w2i, use_brown=True)

    # load back embeddings
    npz = np.load(we)
    W1 = npz['arr_0']
    W2 = npz['arr_1']

    with open(w2i) as f:
        word2idx = json.load(f)
        idx2word = {i: w for w, i in word2idx.items()}

    for concat in (True, False):
        print("** concat:", concat)

        if concat:
            We = np.hstack([W1, W2.T])
        else:
            We = (W1 + W2.T) / 2

        find_analogies('king', 'man', 'woman', We, word2idx, idx2word)
        find_analogies('france', 'paris', 'london', We, word2idx, idx2word)
        find_analogies('france', 'paris', 'rome', We, word2idx, idx2word)
        find_analogies('paris', 'france', 'italy', We, word2idx, idx2word)
        find_analogies('france', 'french', 'english', We, word2idx, idx2word)
        find_analogies('japan', 'japanese', 'chinese', We, word2idx, idx2word)
        find_analogies('japan', 'japanese', 'italian', We, word2idx, idx2word)
        find_analogies('japan', 'japanese', 'australian', We, word2idx, idx2word)
        find_analogies('december', 'november', 'june', We, word2idx, idx2word)
       