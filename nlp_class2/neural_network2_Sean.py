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
Neural Network Bigram Model

用 NN 架構(一層hidden layer)
用到auto-encoder的觀念，來 train 一個 Bigrame model

這一篇加上了 numpy 中 indexing trick 的技巧，可以加速計算


Note:
在這支code，cost 計算要放在權重更新之前就做，要不然算出來的數值會爆炸，講師說原因如下
      We do a gradient descent step after cost 
      since the calculation of doutput will overwrite predictions

'''

sentences, word2idx = get_sentences_with_word2idx_limit_vocab(2000)
# idx2word = [ word2idx[i] for i in range(len(word2idx))]  # 這裡會Error


V = len(word2idx)
D = 100
print('Vocab size:', V)
## 第一種使用的權重 : randomly initialize weights
W1 = np.random.randn(V, D) / np.sqrt(V)
W2 = np.random.randn(D, V) / np.sqrt(D)
# b = 這次省略bias 設置 

# we will also treat beginning of sentence and end of sentence as bigrams
# START -> first word
# last word -> END
start_idx = word2idx['START']
end_idx = word2idx['END']


def sofmax(a):
    a = a - a.max()    # avoid numerical overflow.  避免算出來的數字大到爆炸
    expA = np.exp(a)
    return expA / expA.sum(axis=1, keepdims=True)


epochs = 1
costs = []
learning_rate = 1e-2

t0 = datetime.now()
for epoch in range(epochs):
    random.shuffle(sentences) 
    j = 0
    for sentence in sentences:
        # word2idx = {'START': 0, 'END': 1}
        train_sent = [start_idx] + sentence
        target_sent = sentence + [end_idx]  # 在這裡 train_sent and target_sent的長度一樣
        
        n = len(sentence) + 1               # 每個句子的實際長度

        ## 這次不把input / target matrix做出來，而是指接使用list of index (train_sent / target_sent)
        ## 注意: input and target 都是one hot encoding matrix，會與 cost的寫法有關係(跟下一篇比較)
        X = train_sent
        Y = target_sent
        # get output predictions===============================================================
        Z = np.tanh(W1[X])
        pY  = sofmax(Z.dot(W2))

        # cost funciton=========================================================================
        c_sent = -np.sum(np.log(pY[np.arange(n), Y])) / n   
        costs.append(c_sent)

        # do a gradient descent step=============================================================
        # (pY - Y) 的簡化
        doutput = pY     # N x V           # 會跟doutput指到同一個記憶體位置?????pY的值之後會被改掉?????
        doutput[np.arange(n), target_sent] -= 1   #  取代 (pY - Y)， 原本的Y 是 one-hot encoding
        W2 = W2 - learning_rate*Z.T.dot(doutput)      # (D x N) (N x V)
        dhidden = doutput.dot(W2.T) * (1 - Z*Z)       # (N x V) (V x D) * (N x D)

        i = 0
        for idx in X:
            # dhidden 有 N x D 維度  ， N 會從1~N
            W1[idx] = W1[idx] - learning_rate * dhidden[i]
            i += 1

        ## 更快速的update W1 算法
        # np.subtract.at(W1, X, learning_rate * dhidden)


        if j % 1000 == 0:
            print("epoch:", epoch ," c_sent:", c_sent, "how much sentences has been trained: %i / %i" % (j, len(sentences)))
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


plt.plot(costs)
plt.plot(smoothed_loss(costs))
plt.show()

## 第二種使用的權重，bigram_probs(拿來比較用)
bigram_probs = get_bigram_probs(sentences, V, start_idx, end_idx, smoothing=0.1)
W_biagram = np.log(bigram_probs)


plt.subplot(1,2,1)
plt.title("Neural Network")
plt.imshow(np.tanh(W1).dot(W2))
plt.subplot(1,2,2)
plt.title("Bigram Probs")
plt.imshow(W_biagram)
plt.show()
