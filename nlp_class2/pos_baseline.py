import os, sys
import numpy as np 
import theano 

import theano.tensor as T
import matplotlib.pyplot as plt 
from sklearn.utils import shuffle 
from sklearn.metrics import f1_score 
from sklearn.tree import DecisionTreeClassifier 

'''
pos-tag 問題:
針對輸入文本中的每一個word 學到相對應的tag
'''

class LogisticRegression(object):
    def __init__(self, ):
        pass

    def fit(self, X, Y, V=None, K=None, lr=10e-1, mu=0.99, batch_sz=100, epochs=6):
        if V is None:
            V = len(set(X))
        if K is None:
            K = len(set(Y))
        N = len(X)

        W = np.random.randn(V, K) / np.sqrt(V + K)
        b = np.zeros(K)
        self.W = theano.shared(W)
        self.b = theano.shared(b)
        self.params = [self.W, self.b]

        # on thX and thY, we use "word embedding indexes" as input instead of one-hot matrix
        # input 與 target 本來都應該是 one-hot encoding 後的matrix ，
        thX = T.ivector('X')
        thY = T.ivector('Y')

        # step1 model
        py_x = T.nnet.softmax(self.W[thX] + self.b)  # 注意，因為用了word embedding indexes(numpy indexing)手法，所以model改寫成這樣
        prediction = T.argmax(py_x, axis=1)

        # step2 cost
        cost = -T.mean(T.log(py_x[T.arange(thY.shape[0]), thY]))

        # step3 solver
        grads = T.grad(cost, self.params)
        dparams = [theano.shared(p.get_value()*0) for p in self.params]
        updates = [
            (p, p + mu*dp - lr*g) for p, dp, g in zip(self.params, dparams, grads)
        ] + [
            (dp, mu*dp - lr*g) for dp, g in zip(dparams, grads)
        ]

        self.cost_predict_op = theano.function(    # 模仿api的形式，這裡才要設計成可呼叫的function
            inputs=[thX, thY], 
            outputs=[cost, prediction], 
            allow_input_downcast=True
        )

        train_op = theano.function(
            inputs= [thX, thY], 
            outputs=[cost, prediction], 
            updates=updates,
            allow_input_downcast=True  # 自動降轉成比較節省記憶體的型別
            
        )

        n_batches = N // batch_sz
        costs = []
        for i in range(epochs):
            X, Y = shuffle(X, Y)
            for j in range(n_batches):
                Xbatch = X[j*batch_sz: (j+1)*batch_sz]
                Ybatch = Y[j*batch_sz: (j+1)*batch_sz]

                c, p = train_op(Xbatch, Ybatch)
                costs.append(c)
                if j % 200 == 0:
                    print(
                        "i:", i, "j:", j,
                        "n_batches:", n_batches,
                        "cost:", c,
                        "error:", np.mean(p != Ybatch)
                    )
        plt.plot(costs)
        plt.show()

    def score(self, X, Y):
        _, p = self.cost_predict_op(X, Y)
        return np.mean(p == Y)

    # because there are multiple classes there are going to be multiple If one scores.
    def f1_score(self, X, Y):  
        _, p = self.cost_predict_op(X, Y)
        return f1_score(Y, p, average=None).mean()


def get_data(split_sequences=False):
    # 我習慣在這個資料夾 D:\Sean\PythonWork\Machine Learning Tutorials\Lazy programming\Deep Learning in Python 啟動python
    # 所以下面路徑有修改過
    if not os.path.exists('nlp_class2/chunking'):    
        print("Please create a folder in your local directory called 'chunking'")
        print("train.txt and test.txt should be stored in there.")
        print("Please check the comments to get the download link.")
        exit()
    elif not os.path.exists('nlp_class2/chunking/train.txt'):
        print("train.txt is not in chunking/train.txt")
        print("Please check the comments to get the download link.")
        exit()
    elif not os.path.exists('nlp_class2/chunking/test.txt'):
        print("test.txt is not in chunking/test.txt")
        print("Please check the comments to get the download link.")
        exit()
    
    word2idx = {}
    tag2idx = {}
    word_idx = 0  
    tag_idx = 0
    Xtrain = []
    Ytrain = []    
    currentX = []
    currentY = []
    with open('nlp_class2/chunking/train.txt') as f:
        for line in f:
            line = line.rstrip()
            if line: 
                # if line:  非常精妙的設計
                # 如果 line 有value => 判斷結果true => 跑下面的邏輯
                # 碰到一個句子的結束換行時，會有一行是沒有值 
                # 此時 line = "" => 判斷結果 false => 跑elif的邏輯  
                r = line.split()
                word, tag, _ = r # all we need are first two elements
                if word not in word2idx:
                    word2idx[word] = word_idx
                    word_idx += 1
                currentX.append(word2idx[word])
                if tag not in tag2idx:
                    tag2idx[tag] = tag_idx
                    tag_idx += 1
                currentY.append(tag2idx[tag])
            elif split_sequences:     # 就只有在前一句子結束換行時，會走到這個判斷式
                Xtrain.append(currentX)
                Ytrain.append(currentY)
                currentX = []
                currentY = []
        if not split_sequences:
            Xtrain = currentX
            Ytrain = currentY
    

    # load and score test data
    Xtest = []
    Ytest = []    
    currentX = []
    currentY = []
    with open('nlp_class2/chunking/test.txt') as f:
        for line in f:
            line = line.rstrip()
            if line:            
                r = line.split()
                word, tag, _ = r 
                if word in word2idx:  # test資料集，只考慮train資料集有訓練過的字，所以做個if判斷
                    currentX.append(word2idx[word])
                else:
                    currentX.append(word_idx)   # use this as unknown( 此時的word_idx還沒有指定給那個word )
                currentY.append(tag2idx[tag])
            elif split_sequences:
                Xtest.append(currentX)
                Ytest.append(currentY)
                currentX = []
                currentY = []
        if not split_sequences:
            Xtest = currentX
            Ytest = currentY
    
    
    return Xtrain, Ytrain, Xtest, Ytest, word2idx


def main():
    Xtrain, Ytrain, Xtest, Ytest, word2idx = get_data()

    # convert to numpy arrays
    Xtrain = np.array(Xtrain).astype(np.int32)
    Ytrain = np.array(Ytrain).astype(np.int32)

    N = len(Xtrain)
    V = len(word2idx) + 1 # +unknown word_idx
    print('vocabulary size:', V)

    # 也可選擇把Xtrain 轉成indidator matrix(但太慢了所以不用)
    # Xtrain_indicator = np.zeros((N, V))
    # Xtrain_indicator[np.arange(N), Xtrain] = 1

    # ========== train decision tree  ===============
    dt = DecisionTreeClassifier()
    dt.fit(Xtrain.reshape(N, 1), Ytrain)
    print("dt train score:", dt.score(Xtrain.reshape(N, 1), Ytrain))
    p = dt.predict(Xtrain.reshape(N, 1))
    print('dt train f1:', f1_score(Ytrain, p, average=None).mean())

    # ========== train Logistic Regression  ==========
    model = LogisticRegression()
    model.fit(Xtrain, Ytrain, V=V)
    print('training complete')
    print('lr train score:', model.score(Xtrain, Ytrain))
    print('lr train f1:', model.f1_score(Xtrain, Ytrain))


    Ntest = len(Xtest)
    Xtest =  np.array(Xtest)
    Ytest = np.array(Ytest)
    # decision tree test score
    print('dt test score:', dt.score(Xtest.reshape(Ntest, 1), Ytest))
    p = dt.predict(Xtest.reshape(Ntest, 1))
    print('dt test f1:', f1_score(Ytest, p, average=None).mean())

    # logistic test score
    print('lr test score:', model.score(Xtest, Ytest))
    print('lr test f1:', model.f1_score(Xtest, Ytest))


if __name__ == "__main__" :
    
    print('split_sequences=True')
    Xtrain, Ytrain, Xtest, Ytest, word2idx = get_data(split_sequences=True)
    print('np.array(Xtrain).shape:', np.array(Xtrain).shape)
    print('np.array(Ytrain).shape:', np.array(Ytrain).shape)
    print('np.array(Xtest).shape:', np.array(Xtest).shape)
    print('np.array(Ytest).shape:', np.array(Ytest).shape)
    print('Xtrain[0]:', Xtrain[0])
    print('Y.train[0]:', Ytrain[0])

    print('split_sequences=False')
    Xtrain, Ytrain, Xtest, Ytest, word2idx = get_data(split_sequences=False)
    print('np.array(Xtrain).shape:', np.array(Xtrain).shape)
    print('np.array(Ytrain).shape:', np.array(Ytrain).shape)
    print('np.array(Xtest).shape:', np.array(Xtest).shape)
    print('np.array(Ytest).shape:', np.array(Ytest).shape)
    print('Xtrain[0]:', Xtrain[0])
    print('Y.train[0]:', Ytrain[0])

    # main()