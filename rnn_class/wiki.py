import sys
import theano
import theano.tensor as T
import numpy as np
import matplotlib.pyplot as plt
import json

from datetime import datetime
from sklearn.utils import shuffle
from gru import GRU
from lstm import LSTM
from util import init_weight, get_wikipedia_data
from brown import get_sentences_with_word2idx_limit_vocab

'''
這邊我只用拿 Brown Corpus 來玩

這篇接續 srn_language.py
忘記目的是做什麼了，好像只是訓練word vector，訓練完做比較


學習情境: 情境2+3
T label for sequence fo length T*
預測sentence的下一個字，所以fit 的input 只有 X , 沒有traget(自己做出target)

No labels, just predict next observation (unsupervised)
'''

class RNN(object):
    def __init__(self, D, hidden_layer_sizes,  V):
        self.hidden_layer_sizes = hidden_layer_sizes
        self.D = D
        self.V = V 

    def fit(self, X, learning_rate=1e-5, mu=0.99, activation=T.nnet.relu, RecurrentUnit=LSTM, normalize=True, epochs=10, show_fig=False):
        N = len(X)
        D = self.D 
        V = self.V 

        We = init_weight(V, D)   # embedding matrix
        self.hidden_layers = []
        Mi = D
        for Mo in self.hidden_layer_sizes:
            ru = RecurrentUnit(Mi, Mo, activation)
            self.hidden_layers.append(ru)
            Mi = Mo

        Wo = init_weight(Mi, V)
        bo = np.zeros(V)

        self.We = theano.shared(We)
        self.Wo = theano.shared(Wo)
        self.bo = theano.shared(bo)
        self.params = [self.Wo, self.bo]
        for ru in self.hidden_layers:
            self.params += ru.params

        thX = T.ivectors('X')
        thY = T.ivectors('Y')
        
        Z = self.We[thX]
        for ru in self.hidden_layers:
            Z = ru.output(Z)
        py_x = T.nnet.softmax(Z.dot(self.Wo) + self.bo)  # ????這裡的py_x 不是用scan function 跑出來的，所以不需要另做擷取(y[:, 0, :])

        prediction = T.argmax(py_x, axis=1)
        self.predict_op = theano.function(
            inputs = [thX], 
            outputs=[py_x, prediction], 
            allow_input_downcast=True
        )

        cost = -T.mean(T.log(py_x[T.arange(thY.shape[0]), thY]))
        grads = T.grad(cost, self.params)
        dparams = [theano.shared(p.get_value()*0) for p in self.params]

        dWe = theano.shared(self.We.get_value()*0)
        gWe = T.grad(cost, self.We)
        dWe_update = mu*dWe - learning_rate*gWe
        We_update = self.We + dWe_update
        if normalize:
            We_update /= We_update.norm(2)     # 這裡對 theano.shared 型別做 norm(2)

        updates = [
            (p, p + mu*dp - learning_rate*g) for p, dp, g in zip(self.params, dparams, grads)
        ] + [
            (dp, mu*dp - learning_rate*g) for dp, g in zip(dparams, grads)
        ] + [
            (self.We, We_update), (dWe, dWe_update)
        ]

        self.train_op = theano.function(
            inputs=[thX, thY],
            outputs=[cost, prediction, Z], 
            updates=updates
        )

        costs = []
        for i in range(epochs):
            t0 = datetime.now()
            X = shuffle(X)
            n_correct = 0
            n_total = 0
            cost = 0
            for j in range(N):
                if np.random.random() < 0.01 or len(X[j]) <=1:
                    input_sequence = [0] + X[j]
                    output_sequence = X[j] + [1]
                else:
                    input_sequence = [0] + X[j][:-1]
                    output_sequence = X[j]
                n_total += len(output_sequence)

                # test:
                try:
                    # we set 0 to start and 1 to end
                    c, p, z = self.train_op(input_sequence, output_sequence)
                    # print(z)
                except Exception as e:
                    PYX, pred = self.predict_op(input_sequence)
                    print("input_sequence len:", len(input_sequence))
                    print("PYX.shape:",PYX.shape)
                    print("pred.shape:", pred.shape)
                    raise e                    
                # print('p:', p)
                cost += c
                for pj, xj in zip(p, output_sequence):
                    if pj == xj:
                        n_correct += 1               
                if j % 200 == 0:
                    # 下面這行這個代替 pirnt() ，兩者功能一樣
                    sys.stdout.write("j/N: %d/%d correct rate so far: %f\r" % (j, N, float(n_correct)/n_total))
                    sys.stdout.flush()
            print("i:", i, "cost:", cost, "correct rate:", (float(n_correct)/n_total), 'time for epoch:', (datetime.now() - t0))
            costs.append(cost)

        if show_fig:
            plt.plot(costs)
            plt.show()


def train_wikipedia(we_file='./rnn_class/word_embeddings.npy', w2i_file='./rnn_class/wikipedia_word2idx.json', RecurrentUnit=LSTM):
    # there are 32 files
    ### note: you can pick between Wikipedia data and Brown corpus
    ###       just comment one out, and uncomment the other!

    # Wikipedia data    
    # sentences, word2idx = get_wikipedia_data(n_files=100, n_vocab=2000)
    # use brown from NLTK
    sentences, word2idx = get_sentences_with_word2idx_limit_vocab()

    print('finished retrieving data')
    print('vocab size:', len(word2idx), 'number of sentences:', len(sentences))
    rnn = RNN(30, [30], len(word2idx))
    rnn.fit(sentences ,learning_rate=1e-5, epochs=20, activation=T.nnet.relu, show_fig=True, RecurrentUnit=RecurrentUnit)

    np.save(we_file, rnn.We.get_value())
    with open(w2i_file, 'w') as f:
        json.dump(word2idx, f)


def find_analogies(w1, w2, w3, we_file='./rnn_class/lstm_word_embeddings.npy', w2i_file='./rnn_class/lstm_wikipedia_word2idx.json'):
    We = np.load(we_file)
    with open(w2i_file, 'r') as f:
        word2idx = json.load(f)
    
    king = We[word2idx[w1]]
    man = We[word2idx[w2]]
    woman = We[word2idx[w3]]
    v0 = king - man + woman

    # 歐幾里得距離
    def dist1(a, b):
        return np.linalg.norm(a - b) 
    # 夾角餘弦度量
    def dist2(a, b):
        return 1 - a.dot(b) / (np.linalg.norm(a)* np.linalg.norm(b))  # 1 -  cosine theta , 故意用"1" 去減，是因為cosine 0度 = 1

    for dist, name in [(dist1, 'Euclidean'), (dist2, 'cosine')]:
        min_dist = float('inf')
        best_word = ''
        for word, idx in word2idx.items():
            v1 = We[idx]
            d = dist(v0, v1)
            if d < min_dist:
                min_dist = d
                best_word = word
        print('closest match by', name, ', distance:', best_word)
        print(w1, "-", w2, "=", best_word, "-", w3)


if __name__ == '__main__':
    # we = 'lstm_word_embeddings2.npy'
    we = './rnn_class/lstm_word_embeddings2.npy'
    # w2i = 'lstm_wikipedia_word2idx2.json'
    w2i = './rnn_class/lstm_wikipedia_word2idx2.json'
    train_wikipedia(we, w2i, RecurrentUnit=LSTM)

    # find_analogies('king', 'man', 'woman', we, w2i)
    # find_analogies('france', 'paris', 'london', we, w2i)
    # find_analogies('france', 'paris', 'rome', we, w2i)
    # find_analogies('paris', 'france', 'italy', we, w2i)