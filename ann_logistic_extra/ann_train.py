import numpy as np 
import matplotlib.pyplot as plt

from sklearn.utils import shuffle
from process import get_data
''' train 一個 ANN model，
    使用 NN 架構， 搭配backpropagation'''


def y2indicator(y, K):   # vector => matrix
    N = len(y)
    ind = np.zeros((N, K))
    for i in range(N):
        ind[i, y[i]] = 1
    return ind

X, Y =  get_data()
X, Y =  shuffle(X, Y)
Y = Y.astype(np.int32)

D = X.shape[1]
M = 5
K = len(set(Y))

Xtrain = X[:-100]
Ytrain = Y[:-100]
Ytrain_ind = y2indicator(Ytrain, K)
Xtest = X[-100:]
Ytest = Y[-100:]
Ytest_ind = y2indicator(Ytest, K)

# randomly initialize weights
W1 = np.random.randn(D, M)
b1 = np.zeros(M)
W2 = np.random.randn(M, K)
b2 = np.zeros(K)

def softmax(a):
    expA = np.exp(a)
    return expA / expA.sum(axis=1, keepdims=True)

def forward(X, W1, b1, W2, b2):
    Z = np.tanh(X.dot(W1) + b1)
    Y = softmax(Z.dot(W2) + b2)
    return Y, Z  # input_X and Weight  => output(Matrix)

def predict(P_Y_given_X): # Matrix => vector 
    return np.argmax(P_Y_given_X, axis=1)
    
def classification_rate(Y, P):  # target, prediction ( vector type)
    return np.mean(Y == P)
    # 在這裡是target 與 prediction 統一用vector 樣式來比較

def cross_entropy(T, pY):  # target, output (Matrix type)    # cost function
    return -np.mean(T*np.log(pY))
    # 這裡嚴謹一點的說，output (Matrix)要再經過轉換後才會變成prediction(vector)
    # 上面用 mean 或是 sum 都可以 

train_costs = []
test_costs = []
learning_rate = 0.001
for i in range(10000):
    pYtrain , Ztrain = forward(Xtrain, W1, b1, W2, b2)
    pYtest, Ztest = forward(Xtest, W1, b1, W2, b2)

    ctrain = cross_entropy(Ytrain_ind, pYtrain)
    ctest = cross_entropy(Ytest_ind, pYtest)
    train_costs.append(ctrain)
    test_costs.append(ctest)

    # gradient descent
    W2 -= learning_rate*Ztrain.T.dot(pYtrain - Ytrain_ind)
    b2 -= learning_rate*(pYtrain - Ytrain_ind).sum(axis=0)

    dZ = (pYtrain - Ytrain_ind).dot(W2.T) * Ztrain * (1 - Ztrain)
    # dZ = (pYtrain - Ytrain_ind).dot(W2.T) * (1 - Ztrain*Ztrain)  # 用這個近似的Z會算得更好，為何???  因為 this is for tanh activation
    W1 -= learning_rate*Xtrain.T.dot(dZ)
    b1 -= learning_rate*dZ.sum(axis=0)

    if i % 1000 == 0:
        print(i , ctrain, ctest)

print("Final train classification_rate:", classification_rate(Ytrain, predict(pYtrain)))
print("Final test classification_rate:", classification_rate(Ytest, predict(pYtest)))

legend1, = plt.plot(train_costs, label='train cost')
legend2, = plt.plot(test_costs, label='test cost')
plt.legend([legend1, legend2])
plt.show()

