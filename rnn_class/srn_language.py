import theano 
import theano.tensor as T  
import numpy as np 
import matplotlib.pyplot as plt 

from sklearn.utils import shuffle 
from util import init_weight, get_robert_frost 

from warnings import filterwarnings
filterwarnings('ignore')

'''
Indexing the word embedding
    1. 這個範例用的是word to index (wrod2idx) 的觀念，而不是 word to vector (word2vec)
    也就是每個word有一個對應的index

    2. 藉由　Indexing the word embedding 的觀念，將每個句子(sequence T x V ) 轉成 TxD matrix, 
    也就是 word embedding matrix 

    3. input data 是文本時，常會在模型最前端接一個 word embedding matrix
    可以降維，同時得到更緊奏的word vector

    4. 看起來很像在最前面接一個logistic regression 但是不包含bias

學習情境: 情境2+3
T label for sequence fo length T*
預測sentence的下一個字，所以fit 的input 只有 X , 沒有traget(自己做出target)

No labels, just predict next observation (unsupervised)

'''

'''
這支class 有5個功能: 
    1. 訓練模型; 
    2.將訓練完畢後的模型參數dump出去; 
    3.將模型參數load回來; 
    4. 輸入模型參數，建立一個rnn模型
    5. 利用訓練好的rnn模型，生成新句子
'''

class SimpleRNN(object):
    def __init__(self, D, M, V):
        self.D = D          # dimensionality of word embedding
        self.M = M          # hidden layer size
        self.V = V          # vocabulary size
    
    def fit(self, X, leraning_rate=10e-1, mu=0.99, reg=1.0, activation=T.tanh, epochs=500, show_fig=False):
        N = len(X)
        D = self.D 
        M = self.M
        V = self.V 
        self.f = activation 

        # initial weight 
        We = init_weight(V, D)
        Wx = init_weight(D, M)
        Wh = init_weight(M, M)
        bh = np.zeros(M)
        h0 = np.zeros(M)
        Wo = init_weight(M, V)
        bo = np.zeros(V)

        # make them theano shared
        self.We = theano.shared(We)
        self.Wx = theano.shared(Wx)
        self.Wh = theano.shared(Wh)
        self.bh = theano.shared(bh)
        self.h0 = theano.shared(h0)
        self.Wo = theano.shared(Wo) 
        self.bo = theano.shared(bo)
        self.params = [self.We, self.Wx, self.Wh, self.bh, self.h0, self.Wo, self.bo]

        thX = T.ivector('X')  # 在這邊用 word to index  的觀念
        Ei = self.We[thX]     # will be a TxD matrix  # 這就是經過 word enbedding 的結果，拿這個進入recurrence 學習
        thY = T.ivector('Y')

        # sentence input:
        # [START, w1, w2, ..., wn]
        # sentence target:
        # [w1,    w2, w3, ..., END]

        def recurrence(x_t, h_t1):
            # returns h(t), y(t)
            h_t = self.f(x_t.dot(self.Wx) + h_t1.dot(self.Wh) + self.bh)
            y_t = T.nnet.softmax(h_t.dot(self.Wo) + self.bo)
            return h_t, y_t
        ## 針對單一序列做一次循環，只走完一圈時步 
        [h, y], _ =  theano.scan(
            fn=recurrence, 
            outputs_info=[self.h0, None], 
            sequences=Ei,           # will be a TxD matrix
            n_steps=Ei.shape[0]     # T  (每一個序列(句子)的長度都不一樣喔)
        )

        py_x = y[:, 0, :]
        prediction = T.argmax(py_x, axis=1)     # 這裡的prediction 是經過 scan function 處理後的結果，是個list

        cost = -T.mean(T.log(py_x[T.arange(thY.shape[0]), thY]))
        grads = T.grad(cost, self.params)
        dparams = [theano.shared(p.get_value()*0) for p in self.params ]

        updates = []
        for p, dp, g in zip(self.params, dparams, grads):
            new_dp = mu*dp - leraning_rate*g
            updates.append((dp, new_dp))

            new_p = p + new_dp
            updates.append((p, new_p))
            
        self.predict_op = theano.function(inputs=[thX], outputs=prediction)
        self.train_op = theano.function(
            inputs=[thX, thY], 
            updates=updates, 
            outputs=[cost, prediction]
        )

        costs = []
        n_total =sum((len(sentence) + 1) for sentence in X)  # 用來算準確率
        for i in range(epochs):
            X = shuffle(X)
            n_correct = 0
            cost = 0
            for j in range(N):
                # problem! many words --> END token are overrepresented
                # result: generated lines will be very short
                # we will try to fix in a later iteration
                # BAD! magic numbers 0 and 1...
                input_sentence = [0] + X[j]
                output_sentence = X[j] + [1]

                c, p = self.train_op(input_sentence, output_sentence)  # output的c ,p 都會是一個list ，因為跑過scan的關係
                # print(p)
                cost += c
                # print("j:", j, "c:", c/len(X[j]+1))   # 這行有error 不知為何????
                for pj, xj in zip(p, output_sentence):
                    if pj == xj:
                        n_correct += 1
            print("i:", i, "cost:", cost, "correct rate:", (float(n_correct)/n_total))
            costs.append(cost)

        if show_fig:
            plt.plot(costs)
            plt.show()

    def save(self, filename):
        np.savez(filename, *[p.get_value() for p in self.params])

    @staticmethod
    def load(filename, activation):
        # TODO: would prefer to save activation to file too
        npz = np.load(filename)
        print(type(npz))
        print(npz.values())
        We = npz['arr_0']
        Wx = npz['arr_1']
        Wh = npz['arr_2']
        bh = npz['arr_3']
        h0 = npz['arr_4']
        Wo = npz['arr_5']
        bo = npz['arr_6']
        V, D = We.shape 
        _, M = Wx.shape
        
        # 載入的目的是要拿train好的參數來預測，所以要重新建立theano的computaional grapth
        # 也就是用train好的參數，重新把rnn模型建立起來        
        rnn = SimpleRNN(D, M, V)
        rnn.set(We, Wx, Wh, bh, h0, Wo, bo, activation)  
        return rnn 

    def set(self, We, Wx, Wh, bh, h0, Wo, bo, activation):
        self.f = activation

        # redundant - see how you can improve it
        self.We = theano.shared(We)
        self.Wx = theano.shared(Wx)
        self.Wh = theano.shared(Wh)
        self.bh = theano.shared(bh)
        self.h0 = theano.shared(h0)
        self.Wo = theano.shared(Wo)
        self.bo = theano.shared(bo)
        self.params = [self.We, self.Wx, self.Wh, self.bh, self.h0, self.Wo, self.bo]

        thX = T.ivector('X')
        Ei = self.We[thX] # will be a TxD matrix  # 這裡用了indexing 的技巧
        # thY = T.ivector('Y')

        def recurrence(x_t, h_t1):
            # returns h(t), y(t)
            h_t = self.f(x_t.dot(self.Wx) + h_t1.dot(self.Wh) + self.bh)
            y_t = T.nnet.softmax(h_t.dot(self.Wo) + self.bo)
            return h_t, y_t

        [h, y], _ = theano.scan(
            fn=recurrence,
            outputs_info=[self.h0, None],
            sequences=Ei,
            n_steps=Ei.shape[0],
        )

        py_x = y[:, 0, :]
        prediction = T.argmax(py_x, axis=1)
        self.predict_op = theano.function(
            inputs=[thX], 
            outputs=prediction, 
            allow_input_downcast=True,  # downcast (向下型別轉換)，例如：int64 -> int8 ，目的是節省記憶體空間，會犧牲準確度

        )


    def generate(self, pi, word2idx):  # pi is initial word distribution(機率分布)
        idx2word = {v:k for k, v in word2idx.items()}
        V = len(pi)

        n_lines = 0

        X = [ np.random.choice(V, p=pi) ]  # initial word ，list裡面只會有一個值，後來才會append新的值進去
        print(idx2word[X[0]], end=" ")

        while n_lines < 4 :
            # print('確認X:',X)
            P = self.predict_op(X)[-1]
            X += [P]
            if P > 1:
                # it's a real word, not start/end token
                word = idx2word[P]
                print(word, end=" ")
            elif P == 1:
                # end token
                n_lines += 1
                print('') # 做換行的動作
                if n_lines < 4:
                    X = [ np.random.choice(V, p=pi) ] # reset to start of line  
                    print(idx2word[X[0]], end=" ")


def train_poetry():
    sentences, word2idx = get_robert_frost()

    rnn = SimpleRNN(30, 30, len(word2idx))
    rnn.fit(sentences, leraning_rate=1e-4, show_fig=True, activation=T.nnet.relu, epochs=2000)
    rnn.save('./rnn_class/RNN_D30_M30_epochs2000_relu.npz')

def generate_poetry():
    sentences, word2idx = get_robert_frost()
    rnn = SimpleRNN.load('./rnn_class/RNN_D30_M30_epochs2000_relu.npz', T.nnet.relu)

    # determine initial state distribution for starting sentences
    V = len(word2idx)
    pi = np.zeros(V)
    for sentence in sentences:
        pi[sentence[0]] += 1
    pi /= pi.sum()

    rnn.generate(pi, word2idx)

if __name__ == '__main__':
    # train_poetry()
    generate_poetry()
    print("")
    generate_poetry()
    print("")
    generate_poetry()
    print("")





