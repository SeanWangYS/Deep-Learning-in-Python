import numpy as np 
import matplotlib.pyplot as plt 
import tensorflow as tf 
import os, sys 

sys.path.append(os.path.abspath('.'))
# from pos_baseline import get_data
from sklearn.utils import shuffle 
from nlp_class2.util import init_weight 
from datetime import datetime 
from sklearn.metrics import f1_score 

from tensorflow.contrib.rnn import static_rnn as get_rnn_output
from tensorflow.contrib.rnn import BasicRNNCell, GRUCell

#  tensor flow requires all your sequences to have the same length.
'''
note 1.:
    word_idx / tag_idx are started from 1, not started from 0 as theano version, 
    Well it's because tensorflow needs all the input sequences to be the same size, and in order to do
    that, we need to paddle the short sequences. the way tensorflow does is,  is that pass each sequence with zero, 
    therefore 0 is a special number that we can't use for anything else. So we need to start from 1.

note 2:
    為何要從 N x T x M tensor 轉成 NT x M tensor
    because the rest of tensor flow still isn't like 3D objects 
    in particular matmul which only works on proper matrices which are 2D tensor. (用matmul方法時，只允許 2D tensor 相乘)

    The cost function which is the cross entry because function also does not like 3D objects 
    so we need to flatten our data to put the N and T dimentions together, but keeping the features(D dim) separate
'''


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
    
    # 跟theano的版本相比，這裡刻意start from 1，
    word_idx = 1  
    tag_idx = 1
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


def flatten(l): 
    # This is useful because our data is currently a list of lists.
    return [item for sublist in l for item in sublist]


# get the data
Xtrain, Ytrain, Xtest, Ytest, word2idx = get_data(split_sequences=True)
V = len(word2idx) + 2 # vocab size (+1 for unknown, +1 b/c start from 1)
K = len(set(flatten(Ytrain)) | set(flatten(Ytest))) + 1 # num classes ?? 不懂這邊為何要+1


# training config
epochs = 20
learning_rate = 1e-2
mu = 0.99
batch_size = 32 
hidden_layer_size = 10 
embedding_dim = 10   # M dim
sequence_length = max(len(x) for x in Xtrain + Xtest)


# pad sequences
Xtrain = tf.keras.preprocessing.sequence.pad_sequences(Xtrain, maxlen=sequence_length)
Ytrain = tf.keras.preprocessing.sequence.pad_sequences(Ytrain, maxlen=sequence_length)
Xtest = tf.keras.preprocessing.sequence.pad_sequences(Xtest, maxlen=sequence_length)
Ytest = tf.keras.preprocessing.sequence.pad_sequences(Ytest, maxlen=sequence_length)
print('Xtrain.shape:', Xtrain.shape)
print('Ytrain.shape:', Ytrain.shape)

#  ================= 事先準備model 所需的參數與元件 ======================
# inputs 
inputs = tf.placeholder(tf.int32, shape=(None, sequence_length))     # input is a tensor of shape N x T 
targets = tf.placeholder(tf.int32, shape=(None, sequence_length))    # target is a tensor of shape N x T 
num_sample = tf.shape(inputs)[0]   # useful for later 

# embedding 
We = np.random.randn(V, embedding_dim).astype(np.float32)

# output layer
Wo = init_weight(hidden_layer_size, K).astype(np.float32)
bo = np.zeros(K).astype(np.float32)

# make them tensorflow variables
tfWe = tf.Variable(We)
tfWo = tf.Variable(Wo)
tfbo = tf.Variable(bo)

# make the rnn unit
rnn_unit = GRUCell(num_units=hidden_layer_size, activation=tf.nn.relu)

# ================ model + cost + solver  =======================
# get the output from enbedding layer
x = tf.nn.embedding_lookup(tfWe, inputs)   # x is a tensor of shape  N x T x M 

# converts x from a tensor of shape N x T x M
    # into a list of length T, where each element is a tensor of shape N x M
    # tensorflow的RNN 有個很怪的要求，是輸入tensor型別必須是 T x N x M
    # 還好tensorflow 有現成的方法來改變 tensor shape
x = tf.unstack(x, sequence_length, axis=1)   # axis=1 (第二個維度) 代表對T dim 做分解  # output x is a tensor of shape T x N x D

# get the rnn output
output, states = get_rnn_output(rnn_unit, x, dtype=tf.float32)


# output are now of size (T, N, M)
    # so make it (N, T, M)                   # TODO 這裡可以用unstack嗎??
outputs = tf.transpose(output, (1, 0, 2))    # TODO 確認 transpose/reshape/ unstack 運作邏輯
outputs = tf.reshape(outputs, (num_sample*sequence_length, hidden_layer_size)) # NT x M  (這裡詳閱note 1)

# final dense layer 
logits = tf.matmul(outputs, tfWo) + tfbo  # NT x K
predictions = tf.argmax(logits, axis=1) # (NT, )
predict_op = tf.reshape(predictions, (num_sample, sequence_length)) # N x T 
labels_flat = tf.reshape(targets, [-1])  # (NT, )  ，這一步的目的是為了後續計算cost,詳閱note2

# cost function 
cost_op = tf.reduce_mean(
    tf.nn.sparse_softmax_cross_entropy_with_logits(
        logits=logits, 
        labels=labels_flat
    )
)
# solver
train_op = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(cost_op)

# ===================== init stuff ========================
sess = tf.InteractiveSession()
init = tf.global_variables_initializer()
sess.run(init)

# ===================== training loop ======================
costs = []
n_batch = len(Ytrain) // batch_size
for i in range(epochs):
    n_total = 0  # 考慮資料中存在padded entries，所以另外計算n_total
    n_correct = 0

    t0 = datetime.now()
    Xtrain, Ytrain = shuffle(Xtrain, Ytrain)
    cost = 0

    for j in range(n_batch):
        x = Xtrain[j*batch_size:(j+1)*batch_size]
        y = Ytrain[j*batch_size:(j+1)*batch_size]

        # get the cost, prediction, and perform a gradient descent step
        c, p, _ = sess.run(
            (cost_op, predict_op, train_op),
            feed_dict={inputs: x, targets:y})
        cost += c
        
        # calculate the accuracy
        for yi, pi in zip(y, p):
            # we don't care about the padded entries so ignore them
            yii = yi[yi>0]
            pii = pi[pi>0]
            n_correct += np.sum(yii == pii)
            n_total += len(yii)

        # print stuff out periodically
        if j % 10 == 0:
            sys.stdout.write(
                "j/N: %d/%d correct rate so far: %f, cost so far: %f\r" %
                (j, n_batch, float(n_correct)/n_total, cost)
            )
            sys.stdout.flush()

# get test acc. too
    p = sess.run(predict_op, feed_dict={inputs: Xtest, targets: Ytest})
    n_test_correct = 0
    n_test_total = 0
    for yi, pi in zip(Ytest, p):
        yii = yi[yi>0]
        pii = pi[pi>0]
        n_test_correct += np.sum(yii==pii)
        n_test_total += len(yii)
    test_acc = float(n_test_correct)/n_test_total

    print(
        "i:", i, "cost:", "%.4f" % cost,
        "train acc:", "%.4f" % (float(n_correct)/n_total),
        "test acc:", "%.4f" % test_acc,
        "time for epoch:", (datetime.now() - t0)
    )
    costs.append(cost)

plt.plot(costs)
plt.show()

