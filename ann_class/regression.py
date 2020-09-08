import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

'''用ann架構解決regression問題'''
'''這邊完全我自己寫的，並沒有跟著課程打'''

# generate and plot the data
N = 500
X = np.random.random((N, 2))*4 - 2 # in between (-2, +2) 
Y = X[:,0]*X[:,1] # makes a saddle shape  ，shape is(500,)
# note: in this script "Y" will be the target,
#       "Yhat" will be prediction

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.scatter(X[:,0], X[:,1], Y)
plt.show()

# make a neural network and train it
D = 2
M = 100  # number of hidden unites
# K = 1 # output只有單一值的情況下，不需要特別設定K維度

W1 = np.random.randn(D, M) 
b1 = np.zeros(M)
W2 = np.random.randn(M) # W2 = (M,) 
b2 = 0

def forward(X, W1, b1, W2, b2):
    # sigmoid
    Z = 1 /( 1 + np.exp(-(X.dot(W1) + b1)))
    Y = Z.dot(W2) + b2
    return Y, Z

def SSE(T, Y):
    # Sum Squared Error
    return np.sum((T - Y)**2)

def get_cost(T, Y):
    # mean squared error
    return np.mean((T - Y)**2)

def derivative_w2(Z, T, Y):
    return Z.T.dot(T - Y) 

def derivative_b2(T, Y):
    return (T - Y).sum(axis=0)

def derivative_w1(X, Z, W2, T, Y):
    dZ = np.outer(T - Y, W2) * Z * (1 - Z)   # outer 是取外積，在這個情況下也可以用曲外積來計算微分結果(手動在紙上驗證過了)
    return X.T.dot(dZ)

def derivatieve_b1(Z, W2, T, Y):
    dZ = np.outer(T - Y, W2) * Z * (1 - Z)
    return dZ.sum(axis=0)


learning_rate = 10e-6  ## 這邊我用10-4 結果就會爆掉，就算cost值爆炸(超過float64可乘載的數值)也只會得出cost = nan ，程式不會當掉
costs =[]
for epoch in range(200):
    pY, Z = forward(X, W1, b1, W2, b2)
    cost = get_cost(Y, pY)
    costs.append(cost)

    W2 += learning_rate * derivative_w2(Z, Y, pY)
    b2 += learning_rate * derivative_b2(Y, pY)
    W1 += learning_rate * derivative_w1(X, Z, W2, Y, pY)
    b1 += learning_rate * derivatieve_b1(Z, W2, Y, pY)
    if epoch % 25==0:
        print(cost)

# plot the costs

plt.plot(costs)
plt.show()

