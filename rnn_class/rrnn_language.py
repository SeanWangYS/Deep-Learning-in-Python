import theano 
import theano.tensor as T  
import numpy as np 
import matplotlib.pyplot as plt 

from sklearn.utils import shuffle 
from util import init_weight, get_robert_frost 

from warnings import filterwarnings
filterwarnings('ignore')

'''
(跟 srn.language.py 解決一樣的問題)

這個範例目的是將SimpleRNN 進化~~~~~ 成Rated Recurrent Unit 
* 首先將srn_language.py的內容整個複製過來
* SimpleRNN + 增加了low pass filter的概念 => Rated Recurrent Unit 
* poetry_classifier.py裡面的 set finction給他加進去

學習情境:
T label for sequence fo length T*
預測sentence的下一個字，所以fit 的input 只有 X , 沒有traget(自己做出target)
'''

'''
這支class 有5個功能: 
    1. fit --> 訓練模型(參數初始化之後 用set 建立模型，再做訓練); 
    2. save --> 將訓練完畢後的模型參數dump出去; 
    3. load --> 將模型參數load回來; 並建立rnn模型架構
    4. set --> 輸入模型參數，建立一個rnn模型
    5. generate --> 利用訓練好的rnn模型，生成新句子
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

        # initial weight 
        We = init_weight(V, D)
        Wx = init_weight(D, M)
        Wh = init_weight(M, M)
        bh = np.zeros(M)
        h0 = np.zeros(M)
        # z  = np.ones(M)
        Wxz = init_weight(D, M)
        Whz = init_weight(M, M)
        bz  = np.zeros(M)
        Wo = init_weight(M, V)
        bo = np.zeros(V)

        thX, thY, py_x, prediction = self.set(We, Wx, Wh, bh, h0, Wxz, Whz, bz, Wo, bo, activation)

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
        for i in range(epochs):
            X = shuffle(X)
            n_correct = 0
            n_total = 0
            cost = 0
            for j in range(N):
                if np.random.random() < 0.1:
                    # we're only going to try to learn how to get to the end of the sequence. 10 percent of the time
                    input_sentence = [0] + X[j]
                    output_sentence = X[j] + [1]
                else:
                    # we're just going to go up to the second last word to predict the last word.
                    input_sentence = [0] + X[j][:-1]
                    output_sentence = X[j]
                n_total += len(output_sentence)

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
        Wxz = npz['arr_5']
        Whz = npz['arr_6']
        bz = npz['arr_7']
        Wo = npz['arr_8']
        bo = npz['arr_9']

        V, D = We.shape 
        _, M = Wx.shape
        
        # 載入的目的是要拿train好的參數來預測，所以要重新建立theano的computaional grapth
        # 也就是用train好的參數，重新把rnn模型建立起來        
        rnn = SimpleRNN(D, M, V)
        rnn.set(We, Wx, Wh, bh, h0, Wxz, Whz, bz, Wo, bo, activation)  
        return rnn 

    # 藉由輸入參數建立 computational gragh
    # 1. T.matrix, T.vector and shared variable
    # 2. forward propagation model and predict
    # 3. prediction function
    def set(self, We, Wx, Wh, bh, h0, Wxz, Whz, bz, Wo, bo, activation):
        self.f = activation

        # redundant - see how you can improve it
        self.We = theano.shared(We)
        self.Wx = theano.shared(Wx)
        self.Wh = theano.shared(Wh)
        self.bh = theano.shared(bh)
        self.h0 = theano.shared(h0)
        self.Wxz = theano.shared(Wxz)
        self.Whz = theano.shared(Whz)
        self.bz = theano.shared(bz)
        self.Wo = theano.shared(Wo)
        self.bo = theano.shared(bo)
        self.params = [self.We, self.Wx, self.Wh, self.bh, self.h0, self.Wxz, self.Whz, self.bz, self.Wo, self.bo]

        thX = T.ivector('X')
        Ei = self.We[thX] # will be a TxD matrix  # 這裡用了indexing 的技巧
        thY = T.ivector('Y')

        def recurrence(x_t, h_t1):
            # returns h(t), y(t)
            hhat_t = self.f(x_t.dot(self.Wx) + h_t1.dot(self.Wh) + self.bh)
            z_t = T.nnet.sigmoid(x_t.dot(self.Wxz) + h_t1.dot(self.Whz) + self.bz)
            h_t  = (1 - z_t) * h_t1 + z_t * hhat_t
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
            outputs=[py_x, prediction], 
            allow_input_downcast=True,  # downcast (向下型別轉換)，例如：int64 -> int8 ，目的是節省記憶體空間，會犧牲準確度
        )
        return thX, thY, py_x, prediction

    # We are no longer using pi and we're just going to sample the output of the softmax as a probability
    def generate(self, word2idx):  
        idx2word = {v:k for k, v in word2idx.items()}
        V = len(word2idx)

        n_lines = 0

        X = [ 0 ]   # this is just going to be the start token ( 這個strat tolen 並沒有對應的詞彙喔)
        while n_lines < 4 :
            # print('確認X:',X)
            PY_X, _ = self.predict_op(X)
            # print('PY_X:', PY_X)
            PY_X = PY_X[-1].flatten()
            P = [ np.random.choice(V, p=PY_X) ] 
            # print("what P is:", P)  # 應該只會有一個值
            X += P
            # X = np.concatenate([X, P]) # append to the sequence
            # print('X:', X)
            P = P[-1] # just grab the recent prediction #?? 應該不需要加這行，裡面只有一個value阿
            if P > 1:
                # it's a real word, not start/end token
                word = idx2word[P]
                print(word, end=" ")
            elif P == 1:
                # end token
                n_lines += 1
                X = [0]
                print('') # 做換行的動作


def train_poetry():
    sentences, word2idx = get_robert_frost()

    rnn = SimpleRNN(50, 50, len(word2idx))
    rnn.fit(sentences, leraning_rate=1e-4, show_fig=True, activation=T.nnet.relu, epochs=2000)
    rnn.save('./rnn_class/RRNN_D50_M50_epochs2000_relu.npz')

def generate_poetry():
    sentences, word2idx = get_robert_frost()
    rnn = SimpleRNN.load('./rnn_class/RRNN_D50_M50_epochs2000_relu.npz', T.nnet.relu)
    rnn.generate(word2idx)

if __name__ == '__main__':
    # train_poetry()
    generate_poetry()
    # print("")
    # generate_poetry()
    # print("")
    # generate_poetry()
    # print("")





