import numpy as np
import matplotlib.pyplot as plt 
import random 
from datetime import datetime 



import os
import sys
sys.path.append(os.path.abspath('.'))
from rnn_class.brown import get_sentences_with_word2idx_limit_vocab, get_sentences_with_word2idx
from markov import get_bigram_probs


'''
Neural Bigram Model in Code
用logistic 架構，也就是最基本的neural unit
來 train 一個 Bigrame model
'''

sentences, word2idx = get_sentences_with_word2idx_limit_vocab(2000)
# idx2word = [ word2idx[i] for i in range(len(word2idx))]  # 這裡會Error


V = len(word2idx)
print('Vocab size:', V)
## 第一種使用的權重 : randomly initialize weights
W = np.random.randn(V, V) / np.sqrt(V)
# b = 這次省略bias 設置 


# we will also treat beginning of sentence and end of sentence as bigrams
# START -> first word
# last word -> END
start_idx = word2idx['START']
end_idx = word2idx['END']

## 第二種使用的權重 : 用markov 產生的 bigram language model
bigram_probs = get_bigram_probs(sentences, V, start_idx, end_idx, smoothing=1)
W_bigram = np.log(bigram_probs)


def sofmax(a):
    expA = np.exp(a)
    return expA / expA.sum(axis=1, keepdims=True)

def forword(X, W):
    return sofmax(X.dot(W))

def predict(P_Y_given_X):
    return np.argmax(P_Y_given_X, axis=1)

def cross_entropy(T, pY):
    return -np.mean(T*np.log(pY))

epochs = 1
costs = []
bigram_costs = []
learning_rate = 1e-1

t0 = datetime.now()
for i in range(epochs):
    random.shuffle(sentences) 
    j = 0
    for sentence in sentences:
        # word2idx = {'START': 0, 'END': 1}
        train_sent = [start_idx] + sentence
        target_sent = sentence + [end_idx]
        
        # 資料結構: X,Y 中每個 row(sample) 代表一個字，input第一個字，對應的target便是第二個字
        X = np.zeros((len(sentence)+1, V))
        Y = np.zeros((len(sentence)+1, V))

        for n, idx in enumerate(train_sent):
            X[n, idx] = 1
        for n, idx in enumerate(target_sent):
            Y[n, idx] = 1

        pY = forword(X, W)
        c_sent = cross_entropy(Y, pY)
        costs.append(c_sent)
        # gradient descent
        W -= learning_rate*X.T.dot(pY - Y)


        ## update bigram weight and bigram_costs
        pY_bigram = forword(X, W_bigram)
        c_sent_bigram = cross_entropy(Y, pY_bigram)
        bigram_costs.append(c_sent_bigram)
        # gradient descent
        W_bigram -= learning_rate*X.T.dot(pY_bigram - Y)

        
        if j % 1000 == 0:
            print("epoch:", i ," c_sent:", c_sent, " c_sent_bigram:", c_sent_bigram)
        
        j += 1
print("Elapsed time training:", datetime.now() - t0)


# plot smoothed losses to reduce variability
# 還不知道原理，直接拿來用，目的是讓loss grapth 變得平順一點
def smoothed_loss(x, decay=0.99):
    y = np.zeros(len(x))
    last = 0
    for t in range(len(x)):
        z = decay * last + (1 - decay) * x[t]
        y[t] = z / (1 - decay ** (t + 1))
        last = z
    return y


fig, axes = plt.subplots(1, 2)
axes[0].plot(smoothed_loss(costs))
axes[0].set_title("Logistic Model")
axes[1].plot(smoothed_loss(bigram_costs))
axes[1].set_title("Bigram Probs")
plt.show()


plt.plot(smoothed_loss(costs))
plt.plot(smoothed_loss(bigram_costs))
plt.show()