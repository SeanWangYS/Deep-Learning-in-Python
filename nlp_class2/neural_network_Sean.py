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

=================================================================
Note (重要的實驗心得):
sample code 的參數如下:
    D = 100
    lr = 1e-2
    dhidden = (pY - Y).dot(W2.T) * (1 - Z*Z)   -> 加速運算的近似解


我自己的 code 參數如下:
    D = 200
    lr = 1e-1
    dhidden = (pY - Y).dot(W2.T) * Z * (1 - Z)   -> 我記得這是數學理論的公式解
    


遇到cost value爆炸的問題，cost 算出 nan，也就是出現無限大的計算數值
觸發條件: 
    使用方程式 dhidden = (pY - Y).dot(W2.T) * (1 - Z*Z)
    同時會觸發Error，RuntimeWarning: divide by zero encountered in log，害我一直以為是程式有問題，其實程式沒錯

實驗過程:
1. 只改變 dhidden = (pY - Y).dot(W2.T) * Z * (1 - Z) 
    =>  不會出現nan ，但cost 降低的過程不太穩定，降不太下去( 跟sample code相比)，最後train出的 weight 也比不上sample code的結果

2. D = 200 -> D =100
    沒什麼差別，但我猜測 D 設得太大，在做降為計算的時候

3. 最後才發現是 lr = 1e-1 太大了，也就是造成cost爆炸的原因，當調整成lr = 1e-2
    => 上便兩種 dhidden 的公式都很穩定，cost不會爆炸，且會慢慢下降

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

def forword(X, W1, W2):
    Z = np.tanh(X.dot(W1))
    Y = sofmax(Z.dot(W2))
    return Y, Z


epochs = 1
costs = []
learning_rate = 1e-2

t0 = datetime.now()
for i in range(epochs):
    random.shuffle(sentences) 
    j = 0
    for sentence in sentences:
        # word2idx = {'START': 0, 'END': 1}
        train_sent = [start_idx] + sentence
        target_sent = sentence + [end_idx]  # 在這裡 train_sent and target_sent的長度一樣
        
        n = len(sentence) + 1               # 每個句子的實際長度
        X = np.zeros((n, V))
        Y = np.zeros((n, V))

        ## 方法一: 正規的方式來 fill in matrix
        # for n, idx in enumerate(train_sent):
        #     X[n, idx] = 1
        # for n, idx in enumerate(target_sent):
        #     Y[n, idx] = 1

        ## 方法二: 因為train_sent and target_sent 都是 list of indexes 的情況下，可以利用numpy indexing tick 來 full in matrix 
        ## 注意: input and target 都是word-embedding indexes，會與 cost的寫法有關係(跟上一篇比較)
        X[np.arange(n), train_sent] = 1      
        Y[np.arange(n), target_sent] = 1

        # get output predictions
        Z = np.tanh(X.dot(W1))   # hidden layer
        pY  = sofmax(Z.dot(W2))  # prediction

        # do a gradient descent step
        W2 -= learning_rate*Z.T.dot(pY - Y) 
        dhidden = (pY - Y).dot(W2.T) * (1 - Z*Z)  # 也可以用 dhidden = (pY - Y).dot(W2.T) * Z * (1 - Z)
        W1 -= learning_rate*X.T.dot(dhidden)

        # cost funciton
        c_sent = -np.sum(Y * np.log(pY)) / n   # 在這邊使用sum() / len(sentence) 計算的平均cost，天生就會與 mean() 的平均cost有差異，mean() 是除上matrix的所有元素個數
        costs.append(c_sent)

        if j % 100 == 0:
            print("epoch:", i ," c_sent:", c_sent, "how much sentences has been trained: %i / %i" % (j, len(sentences)))
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