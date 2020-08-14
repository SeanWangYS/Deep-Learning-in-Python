import numpy as np
from sklearn.utils import shuffle
import matplotlib.pyplot as plt

from util import get_normalized_data, error_rate, cost, y2indicator
from mlp import forward, derivative_w2, derivative_w1, derivative_b2, derivative_b1

def main():
    # compare 3 scenarios:
    # 1. batch SGD
    # 2. batch SGD with momentum

    '''
    # steps of training a model
    # 0. get data
    # 1. function --> forwrad --> OK
    # 2. loss --> cost --> ok
    # 3. solver --> gradient descent
    '''

    max_iter = 20 # make it 30 for sigmoid
    print_period = 50

    Xtrain, Xtest, Ytrain, Ytest = get_normalized_data()
    lr = 0.00004
    reg = 0.01

    Ytrain_ind  = y2indicator(Ytrain)   # Target of train data 
    Ytest_ind = y2indicator(Ytest)      # Target of test data 

    N, D = Xtrain.shape
    batch_sz = 500
    n_batches = N // batch_sz

    M = 300
    K = 10 

    np.random.seed(42)
    W1 = np.random.randn(D, M) / np.sqrt(D)   
    b1 = np.zeros(M)
    W2 = np.random.randn(M, K) / np.sqrt(M)  
    b2 = np.zeros(K)

    '''
    initial weight 分母除上np.sqrt(D)對cost非常有幫助
    1. 如果不除這個東東，initial weight 會太大，將造成以下後果
        1.1 計算forward()時，通過sigmoid中的np.exp(-( X.dot(W1) + b1 ))，一開始很可能會出現太大數字(overflow encountered 數據溢出)，
        溢出代表該數字物件的型別無法承載這麼大數字，會報錯，但code還是能繼續跑，因為sigmoid的值在0~1之間

        1.2 如果是通過relu，輸出值可能會是一極大值
        之後後再經過expA = np.exp(A)，大概首輪epoch中的第二個batch計算時，就會得到無限大爆掉了

        1.3 結論: activation = relu時， initial weight要夠小

    2. 除上一個極大值500 ( 約np.sqrt(D)的20倍 )，在update weight with momentum的實驗中，
        最後一個epoch的cost = 106.49

    3. 除 np.sqrt(D)，在update weight with momentum的實驗中，
        最後一個epoch的cost = 119.005

    4. 除上一個極大值5000，在update weight with momentum的實驗中，
        最後一個epoch的cost = 160.68

    5. 其他參考書上說，大部分情況直接無腦除上100即可以得到很好的訓練成果
    '''

    # save initial weights
    W1_0 = W1.copy()
    b1_0 = b1.copy()
    W2_0 = W2.copy()
    b2_0 = b2.copy()

    # 1. batch SGD
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


    # 2. batch with momentum
    W1 = W1_0.copy()
    b1 = b1_0.copy()
    W2 = W2_0.copy()
    b2 = b2_0.copy()

    losses_momentum = []
    errors_momentum = []
    mu = 0.9           # momentu的係數，大多數用0.9 就已經足夠好了
    dW2 = 0
    db2 = 0
    dW1 = 0
    db1 = 0
    for i in range(max_iter):
        for j in range(n_batches):
            Xbatch = Xtrain[j*batch_sz:(j*batch_sz + batch_sz),]
            Ybatch = Ytrain_ind[j*batch_sz:(j*batch_sz + batch_sz),]
            pYbatch, Z = forward(Xbatch, W1, b1, W2, b2)

            # gradients
            gW2 = derivative_w2(Z, Ybatch, pYbatch) + reg*W2
            gb2 = derivative_b2(Ybatch, pYbatch) + reg*b2
            gW1 = derivative_w1(Xbatch, Z, Ybatch, pYbatch, W2) + reg*W1
            gb1 = derivative_b1(Z, Ybatch, pYbatch, W2) + reg*b1

            # update velocities
            dW2 = mu*dW2 - lr*gW2
            db2 = mu*db2 - lr*gb2
            dW1 = mu*dW1 - lr*gW1
            db1 = mu*db1 - lr*gb1

            # updates
            W2 += dW2
            b2 += db2
            W1 += dW1
            b1 += db1

            if j % print_period == 0:
                pY, _ = forward(Xtest, W1, b1, W2, b2)
                l = cost(pY, Ytest_ind)
                losses_momentum.append(l)
                print('Cost at iteration i=%d, j=%d : %.6f' % (i, j, l))

                e = error_rate(pY, Ytest)
                errors_momentum.append(e)
                print('Error rate:', e)

    pY, _ = forward(Xtest, W1, b1, W2, b2)
    print('Fianl error rate:', error_rate(pY, Ytest))


    plt.plot(losses_batch, label='batch')
    plt.plot(losses_momentum, label="momentum")
    plt.legend()
    plt.show()



if __name__ == '__main__':
    main()


