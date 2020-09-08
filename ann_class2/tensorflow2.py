import numpy as np
import tensorflow as tf

import matplotlib.pyplot as plt

from util import get_normalized_data, y2indicator

def error_rate(p, t):
    return np.mean(p != t)


''' 沒有包成API 手刻了兩層hidden layers'''
def main():
    # setp1: get the data and define all the usual variables for training weights
    Xtrain, Xtest, Ytrain, Ytest = get_normalized_data()
    Ytrain_ind = y2indicator(Ytrain)
    Ytest_ind = y2indicator(Ytest)
    
    max_iter = 15
    print_period = 50
    
    # 1.1 data size and how much neurons in each layers
    N, D = Xtrain.shape
    M1 = 300
    M2 = 100
    K = 10

    # 1.2 weight initialize
    # np.random.seed(123)
    W1_init = np.random.randn(D, M1) / np.sqrt(D)
    b1_init = np.zeros(M1)
    W2_init = np.random.randn(M1, M2) / np.sqrt(M1)
    b2_init = np.zeros(M2)
    W3_init = np.random.randn(M2, K) / np.sqrt(M2)
    b3_init = np.zeros(K)
    '''
    這邊將randn改成rand的話，training 的效果差非常多
    '''

    # 1.3 define batch_size
    batch_sz = 500
    n_batches = N // batch_sz

    # 1.4 hyperparameter
    lr = 0.00004
    # reg = 0.01  # 在tf 的cost fun中 用不到這個參數

    # difine tensorflow variabels and express
    X = tf.placeholder(tf.float32, shape=(None, D), name='X')
    T = tf.placeholder(tf.float32, shape=(None, K), name='T')
    W1 = tf.Variable(W1_init.astype(np.float32))
    b1 = tf.Variable(b1_init.astype(np.float32))
    W2 = tf.Variable(W2_init.astype(np.float32))
    b2 = tf.Variable(b2_init.astype(np.float32))
    W3 = tf.Variable(W3_init.astype(np.float32))
    b3 = tf.Variable(b3_init.astype(np.float32))

    # setp2. define the model
    Z1 = tf.nn.relu( tf.matmul(X, W1) + b1 )
    Z2 = tf.nn.relu( tf.matmul(Z1, W2) + b2 )
    Yish = tf.matmul(Z2, W3) + b3 

    # step3. define the cost function
    cost = tf.reduce_sum(tf.nn.softmax_cross_entropy_with_logits_v2(logits=Yish, labels=T))

    # setp4. define the slover
    # we choose the optimizer but don't implement the algorithm ourselves
    # let's go with RMSprop, since we just learned about it.
    # it includes momentum!
    train_op  = tf.train.RMSPropOptimizer(lr, decay=0.99, momentum=0.9).minimize(cost)

    # we'll use this to calculate the error rate
    predict_op = tf.argmax(Yish, 1)

    init = tf.global_variables_initializer()

    costs = []
    with tf.Session() as session:
        session.run(init)

        for i in range(max_iter):
            for j in range(n_batches):
                Xbatch = Xtrain[j*batch_sz:(j*batch_sz + batch_sz),]
                Ybatch = Ytrain_ind[j*batch_sz:(j*batch_sz + batch_sz),]

                session.run(train_op, feed_dict={X: Xbatch, T: Ybatch})
                if j % print_period == 0:
                    test_cost = session.run(cost, feed_dict={X: Xtest, T:Ytest_ind})
                    prediction = session.run(predict_op, feed_dict={X: Xtest})
                    err = error_rate(prediction, Ytest)
                    print("Cost / err at iteration i=%d, j=%d, %.3f / %.3f" % (i, j, test_cost, err))
                    costs.append(test_cost)

    plt.plot(costs)
    plt.show()


''' tensorflow 比較嚴謹的寫法 + 使用layer API'''
def main2():
    # setp1: get the data and define all the usual variables for training weights
    Xtrain, Xtest, Ytrain, Ytest = get_normalized_data()
    Ytrain_ind = y2indicator(Ytrain)
    Ytest_ind = y2indicator(Ytest)
    
    max_iter = 15
    print_period = 50
    
    # 1.1 data size and how much neurons in each layers
    N, D = Xtrain.shape
    M1 = 300
    M2 = 100
    K = 10

    # 1.3 define batch_size
    batch_sz = 500
    n_batches = N // batch_sz

    # 1.4 hyperparameter
    lr = 0.00004
    # reg = 0.01

    # difine tensorflow variabels and express
    g = tf.Graph()
    with g.as_default():
        X = tf.placeholder(tf.float32, shape=(None, D), name='X')
        T = tf.placeholder(tf.float32, shape=(None, K), name='T')

        # step2. define the model with tensorflow layer API
        Z1 = tf.layers.dense(inputs=X, units=300, activation=tf.nn.relu)   
        Z2 = tf.layers.dense(inputs=Z1, units=100, activation=tf.nn.relu)
        Yish = tf.layers.dense(inputs=Z2, units=10, activation=None)

        # step3. define the cost
        cost = tf.reduce_sum(tf.nn.softmax_cross_entropy_with_logits_v2(logits=Yish, labels=T))
        
        # step4. define the slover
        train_op  = tf.train.RMSPropOptimizer(lr, decay=0.99, momentum=0.9).minimize(cost)
        predict_op = tf.argmax(Yish, 1)

        init = tf.global_variables_initializer()
        
    costs = []
    with tf.Session(graph=g) as session:
        session.run(init)

        for i in range(max_iter):
            for j in range(n_batches):
                Xbatch = Xtrain[j*batch_sz:(j*batch_sz + batch_sz),]
                Ybatch = Ytrain_ind[j*batch_sz:(j*batch_sz + batch_sz),]

                session.run(train_op, feed_dict={X: Xbatch, T: Ybatch})
                if j % print_period == 0:
                    test_cost = session.run(cost, feed_dict={X: Xtest, T:Ytest_ind})
                    prediction = session.run(predict_op, feed_dict={X: Xtest})
                    err = error_rate(prediction, Ytest)
                    print("Cost / err at iteration i=%d, j=%d, %.3f / %.3f" % (i, j, test_cost, err))
                    costs.append(test_cost)

    plt.plot(costs)
    plt.show()


if __name__ == '__main__':
    # main()
    main2()

