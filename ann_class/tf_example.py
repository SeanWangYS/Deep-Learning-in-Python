import tensorflow as tf
import  numpy as np
import matplotlib.pyplot as plt

# create the data
Nclass = 500
D = 2                           # dimensionality of input
M = 3                           # hidden layer size
K = 3                           # number of classes

X1 = np.random.randn(Nclass, D) + np.array([0, -2])
X2 = np.random.randn(Nclass, D) + np.array([2, 2])
X3 = np.random.randn(Nclass, D) + np.array([-2, 2])
X = np.vstack([X1, X2, X3])                           # (N, 2)

# traget
Y = np.array([0]*Nclass + [1]*Nclass + [2]*Nclass)    # (N, )
N = len(Y)  # 1500

# turn Y into an indicator matrix for training
T =  np.zeros((N, K))        
for i in range(N):             
    T[i, Y[i]] = 1            

# let's see what it looks like
plt.scatter(X[:,0], X[:,1], c=Y, s=100, alpha=0.5)
plt.show()




# tensor flow variables are not the same as regular Python variables
def init_weights(shape):
    return tf.Variable(tf.random_normal(shape, stddev=0.01))

def forward(X, W, b1, W2, b2):
    Z = tf.nn.sigmoid(tf.matmul(X, W1) + b1)
    return tf.matmul(Z, W2) + b2  # 這裡output不經過softmax，且在tf中直接用該output 計算 cost function

# take placeholder represent X, Y data
tfX = tf.placeholder(tf.float32, [None, D])
tfY = tf.placeholder(tf.float32, [None, K])

# create symbolic variables
W1 = init_weights([D, M])
b1 = init_weights([M])
W2 = init_weights([M, K])
b2 = init_weights([K])

# output variable(it has no value yet)
logits = forward(tfX, W1, b1, W2, b2)

cost = tf.reduce_mean(
    tf.nn.softmax_cross_entropy_with_logits_v2(
        labels=tfY, 
        logits=logits
        )
) # compute costs
# WARNING: This op expects unscaled logits,
# since it performs a softmax on logits
# internally for efficiency.
# Do not call this op with the output of softmax,
# as it will produce incorrect results.


 # construct an optimizer
train_op = tf.train.GradientDescentOptimizer(0.05).minimize(cost)  # we don't need to specifiy derivative function in tf
# input parameter is the learning rate

predict_op = tf.argmax(logits, 1)
# input parameter is the axis on which to choose the max


sess = tf.Session()
init = tf.initialize_all_variables()
sess.run(init)

for i in range(1000):
    sess.run(train_op, feed_dict={tfX: X, tfY: T})
    pred = sess.run(predict_op, feed_dict={tfX: X, tfY: T})
    if i % 100 == 0:
        print("Accuracy: ",np.mean(Y == pred))

