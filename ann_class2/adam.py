import numpy as np
import matplotlib.pyplot as plt
from sklearn.utils import shuffle

from util import get_normalized_data,cost,  error_rate, y2indicator
from mlp import forward, derivative_w1, derivative_b1, derivative_w2, derivative_b2

''' 這篇整理出我認為最好的編寫bp 的步驟'''
def main():
    max_iter = 10
    print_period = 40

    Xtrain, Xtest, Ytrain, Ytest = get_normalized_data()
    Ytrain_ind = y2indicator(Ytrain)   # Target of train data 
    Ytest_ind = y2indicator(Ytest)      # Target of test data 

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

    lr = 0.001
    reg = 0.01

    # 1. RMSProp + Momemtum
    decay_rate = 0.999
    eps = 1e-8
    mu = 0.9

    # RMSProp cache
    cache_W2 = 1
    cache_b2 = 1
    cache_W1 = 1
    cache_b1 = 1

    # momentum
    dW2 = 0
    db2 = 0
    dW1 = 0
    db1 = 0

    losses_rms = []
    errors_rms = []
    for i in range(max_iter):
        # Xtrain, Ytrain_ind = shuffle(Xtrain, Ytrain_ind)
        for j in range(n_batches):
            Xbatch = Xtrain[j*batch_sz:(j*batch_sz + batch_sz),]
            Ybatch = Ytrain_ind[j*batch_sz:(j*batch_sz + batch_sz),]  # Target of each batch
            pYbatch, Z = forward(Xbatch, W1, b1, W2, b2)

            # updates
            ## 這邊updata momentum的形式，不是原始定義，而是模仿adam的 moment1
            ## 可以當成沒有使用 Bias correction 的 adam
            gW2 = derivative_w2(Z, Ybatch, pYbatch) + reg*W2
            cache_W2 = decay_rate*cache_W2 + (1 - decay_rate)*gW2*gW2
            dW2 = mu*dW2 + (1 - mu) * lr * gW2 / (np.sqrt(cache_W2) + eps)
            W2 -= dW2

            gb2 = derivative_b2(Ybatch, pYbatch) + reg*b2
            cache_b2 = decay_rate*cache_b2 + (1 - decay_rate)*gb2*gb2
            db2 = mu*db2 + (1 - mu) * lr * gb2 / (np.sqrt(cache_b2) + eps)
            b2 -= db2

            gW1 = derivative_w1(Xbatch, Z, Ybatch, pYbatch, W2) + reg*W1
            cache_W1 = decay_rate*cache_W1 + (1 - decay_rate)*gW1*gW1
            dW1 = mu*dW1 + (1 - mu) * lr * gW1 / (np.sqrt(cache_W1) + eps)
            W1 -= dW1

            gb1 = derivative_b1(Z, Ybatch, pYbatch, W2) + reg*b1
            cache_b1 = decay_rate*cache_b1 + (1 - decay_rate)*gb1*gb1
            db1 = mu*db1 + (1 - mu)* lr * gb1 / (np.sqrt(cache_b1) + eps)
            b1 -= db1

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


    # 2. Adam
    W1 = W1_0.copy()
    b1 = b1_0.copy()
    W2 = W2_0.copy()
    b2 = b2_0.copy()

    beta1 = 0.9  # 0.999  # 這兩個值調一下  就會發生很大變化的樣子， 也就是說參數調教超重要的阿
    beta2 = 0.999  # 0.99
    eps = 1e-8
    t = 1

    m_W2 = 0
    m_b2 = 0
    m_W1 = 0
    m_b1 = 0

    v_W2 = 0
    v_b2 = 0
    v_W1 = 0
    v_b1 = 0

    losses_adam = []
    errors_adam = []
    for i in range(max_iter):
        # Xtrain, Ytrain_ind = shuffle(Xtrain, Ytrain_ind)
        for j in range(n_batches):
            Xbatch = Xtrain[j*batch_sz:(j*batch_sz + batch_sz),]
            Ybatch = Ytrain_ind[j*batch_sz:(j*batch_sz + batch_sz),]  # Target of each batch
            pYbatch, Z = forward(Xbatch, W1, b1, W2, b2)

            # gradient
            gW2 = derivative_w2(Z, Ybatch, pYbatch) + reg*W2
            gb2 = derivative_b2(Ybatch, pYbatch) + reg*b2
            gW1 = derivative_w1(Xbatch, Z, Ybatch, pYbatch, W2) + reg*W1
            gb1 = derivative_b1(Z, Ybatch, pYbatch, W2) + reg*b1

            # new m
            m_W2 = beta1*m_W2 + (1 - beta1)*gW2
            m_b2 = beta1*m_b2 + (1 - beta1)*gb2
            m_W1 = beta1*m_W1 + (1 - beta1)*gW1
            m_b1 = beta1*m_b1 + (1 - beta1)*gb1

            # new v
            v_W2 = beta2*v_W2 + (1 - beta2)*gW2*gW2
            v_b2 = beta2*v_b2 + (1 - beta2)*gb2*gb2
            v_W1 = beta2*v_W1 + (1 - beta2)*gW1*gW1
            v_b1 = beta2*v_b1 + (1 - beta2)*gb1*gb1

            # m_hat get from bias correction
            m_W2_hat = m_W2 / (1 - beta1**t)
            m_b2_hat = m_b2 / (1 - beta1**t)
            m_W1_hat = m_W1 / (1 - beta1**t)
            m_b1_hat = m_b1 / (1 - beta1**t)

            # v_hat get from bias correction
            v_W2_hat = v_W2 / (1 - beta2**t)
            v_b2_hat = v_b2 / (1 - beta2**t)
            v_W1_hat = v_W1 / (1 - beta2**t)
            v_b1_hat = v_b1 / (1 - beta2**t)

            # update
            W2 -= lr * m_W2_hat / np.sqrt(v_W2_hat + eps)
            b2 -= lr * m_b2_hat / np.sqrt(v_b2_hat + eps)
            W1 -= lr * m_W1_hat / np.sqrt(v_W1_hat + eps)
            b1 -= lr * m_b1_hat / np.sqrt(v_b1_hat + eps)

            t += 1
            if j % print_period == 0:
                pY, _ = forward(Xtest, W1, b1, W2, b2)
                l = cost(pY, Ytest_ind)
                losses_adam.append(l)
                print("Cost at iteration i=%d, j=%d: %.6f" % (i, j, l))

                e = error_rate(pY, Ytest)
                errors_adam.append(e)
                print("Error rate:", e)

    pY, _ = forward(Xtest, W1, b1, W2, b2)
    print("Final error rate:", error_rate(pY, Ytest))

    plt.plot(losses_rms, label='RMSprop + Momentun')
    plt.plot(losses_adam, label='Adam')
    plt.legend()
    plt.show()

if __name__ == '__main__':
    main()