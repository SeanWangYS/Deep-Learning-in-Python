import numpy as np
import matplotlib.pyplot as plt
from sklearn.utils import shuffle

from util import get_normalized_data,cost,  error_rate, y2indicator
from mlp import forward, derivative_w1, derivative_b1, derivative_w2, derivative_b2

def main():
    max_iter = 20
    print_period = 20

    Xtrain, Xtest, Ytrain, Ytest = get_normalized_data()
    Ytrain_ind = y2indicator(Ytrain)   # Target of train data 
    Ytest_ind = y2indicator(Ytest)      # Target of test data 

    lr = 0.00004
    reg = 0.01

    N, D = Xtrain.shape
    M = 300
    K = 10

    np.random.seed(123)
    W1 = np.random.randn(D, M) / np.sqrt(D)   
    b1 = np.zeros(M)
    W2 = np.random.randn(M, K) / np.sqrt(M)  
    b2 = np.zeros(K)

    batch_sz = 500
    n_batches = N // batch_sz    # 82

    # save initial weights
    W1_0 = W1.copy()
    b1_0 = b1.copy()
    W2_0 = W2.copy()
    b2_0 = b2.copy()

    # 1. learning rate =  constant
    losses_batch = []
    errors_batch = []
    for i in range(max_iter):
        # Xtrain, Ytrain_ind = shuffle(Xtrain, Ytrain_ind)
        for j in range(n_batches):
            Xbatch = Xtrain[j*batch_sz:(j*batch_sz + batch_sz),]
            Ybatch = Ytrain_ind[j*batch_sz:(j*batch_sz + batch_sz),]  # Target of each batch
            pYbatch, Z = forward(Xbatch, W1, b1, W2, b2)
            # print "first batch cost:", cost(pYbatch, Ybatch)

            # updates
            W2 -= lr*(derivative_w2(Z, Ybatch, pYbatch) + reg*W2)    
            b2 -= lr*(derivative_b2(Ybatch, pYbatch) + reg*b2)
            W1 -= lr*(derivative_w1(Xbatch, Z, Ybatch, pYbatch, W2) + reg*W1)
            b1 -= lr*(derivative_b1(Z, Ybatch, pYbatch, W2) + reg*b1)

            if j % print_period == 0:
                pY, _ = forward(Xtest, W1, b1, W2, b2)
                l = cost(pY, Ytest_ind)
                losses_batch.append(l)
                print("Cost at iteration i=%d, j=%d: %.6f" % (i, j, l))

                e = error_rate(pY, Ytest)
                errors_batch.append(e)
                print("Error rate:", e)

    pY, _ = forward(Xtest, W1, b1, W2, b2)
    print("Final error rate:", error_rate(pY, Ytest))


    # 2.  RMSprop
    W1 = W1_0.copy()
    b1 = b1_0.copy()
    W2 = W2_0.copy()
    b2 = b2_0.copy()

    losses_rms = []
    errors_rms = []

    '''
    in RMSprop you can use a bigger lr, 
    but if you set this too high you'll get NaN!
    if you use the same learning rate within RMSprop and General method, there is only slight difference between them. 
    '''
    lr0 = 0.001   
    cache_W2 = 1
    cache_b2 = 1
    cache_W1 = 1
    cache_b1 = 1
    decay_rate = 0.999
    eps = 1e-10
    for i in range(max_iter):
        # Xtrain, Ytrain_ind = shuffle(Xtrain, Ytrain_ind)
        for j in range(n_batches):
            Xbatch = Xtrain[j*batch_sz:(j*batch_sz + batch_sz),]
            Ybatch = Ytrain_ind[j*batch_sz:(j*batch_sz + batch_sz),]  # Target of each batch
            pYbatch, Z = forward(Xbatch, W1, b1, W2, b2)
            # print "first batch cost:", cost(pYbatch, Ybatch)

            # # update
            # cache_W2 = decay_rate*cache_W2 + (1 - decay_rate)*np.square(derivative_w2(Z, Ybatch, pYbatch) + reg*W2)
            # W2 -= lr0 / (np.sqrt(cache_W2) + eps) *(derivative_w2(Z, Ybatch, pYbatch) + reg*W2)

            # cache_b2 = decay_rate*cache_b2 + (1 - decay_rate)*np.square(derivative_b2(Ybatch, pYbatch) + reg*b2)
            # b2 -= lr0 / (np.sqrt(cache_b2) + eps) *(derivative_b2(Ybatch, pYbatch) + reg*b2)

            # cache_W1 = decay_rate*cache_W1 + (1 - decay_rate)*np.square(derivative_w1(Xbatch, Z, Ybatch, pYbatch, W2))
            # W1 -= lr0 / (np.sqrt(cache_W1) + eps) *(derivative_w1(Xbatch, Z, Ybatch, pYbatch, W2) + reg*W1)

            # cache_b1 = decay_rate*cache_b1 + (1 - decay_rate)*np.square(derivative_b1(Z, Ybatch, pYbatch, W2) + reg*b1)
            # b1 -= lr0 / (np.sqrt(cache_b1) + eps) *(derivative_b1(Z, Ybatch, pYbatch, W2) + reg*b1)



            # updates
            # 更聰明的寫法，是把上面式子中，會重複計算到的部分提出來計算並指派給變數，讓它只計算一次，這樣會加速
            gW2 = derivative_w2(Z, Ybatch, pYbatch) + reg*W2
            cache_W2 = decay_rate*cache_W2 + (1 - decay_rate)*gW2*gW2
            W2 -= lr0 * gW2 / (np.sqrt(cache_W2) + eps)

            gb2 = derivative_b2(Ybatch, pYbatch) + reg*b2
            cache_b2 = decay_rate*cache_b2 + (1 - decay_rate)*gb2*gb2
            b2 -= lr0 * gb2 / (np.sqrt(cache_b2) + eps)

            gW1 = derivative_w1(Xbatch, Z, Ybatch, pYbatch, W2) + reg*W1
            cache_W1 = decay_rate*cache_W1 + (1 - decay_rate)*gW1*gW1
            W1 -= lr0 * gW1 / (np.sqrt(cache_W1) + eps)

            gb1 = derivative_b1(Z, Ybatch, pYbatch, W2) + reg*b1
            cache_b1 = decay_rate*cache_b1 + (1 - decay_rate)*gb1*gb1
            b1 -= lr0 * gb1 / (np.sqrt(cache_b1) + eps)


            if j % print_period == 0:
                pY, _ = forward(Xtest, W1, b1, W2, b2)
                l = cost(pY, Ytest_ind)
                losses_rms.append(l)
                print("Cost at iteration i=%d, j=%d: %.6f" % (i, j, l))

                e = error_rate(pY, Ytest)
                errors_rms.append(e)
                print("Error rate:", e)

    pY, _ = forward(Xtest, W1, b1, W2, b2)
    print("Final error rate:", error_rate(pY, Ytest))

    plt.plot(losses_batch, label='contant')
    plt.plot(losses_rms, label='RMSprop')
    plt.legend()
    plt.show()

if __name__ == '__main__':
    main()