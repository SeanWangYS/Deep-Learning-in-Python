import tensorflow as tf 
import numpy as np 
import matplotlib.pyplot as plt 


from sklearn.utils import shuffle 
from util import init_weight, get_robert_frost 

'''

'''

class SimpleRNN(object):
    def __init__(self, D, M, V):
        self.D = D          # dimensionality of word embedding
        self.M = M          # hidden layer size
        self.V = V          # vocabulary size

    def set_session(self, session):
        self.session = session

    # 藉由輸入參數建立 computational gragh ()
    # 1. placeholder and tf.Variable
    # 2. forward propagation model and predict
    # 3. prediction function
    def build(self, We, Wx, Wh, bh, h0, Wo, bo):
        # make them tf Variables
        self.We = tf.Variable(We)
        self.Wx = tf.Variable(Wx)
        self.Wh = tf.Variable(Wh)
        self.bh = tf.Variable(bh)
        self.b0 = tf.Variable(b0)
        self.Wo = tf.Variable(Wo)
        self.bo = tf.Variable(bo)
        self.params = [self.We, self.Wx, self.Wh, self.bh, self.h0, self.Wo, self.bo]

        # for easy access
        V = self.V 
        D = self.D 
        M = self.M

        self.tfX = tf.placeholder(tf.float32, shape=(None, ), name='X')
        self.tfY = tf.placeholder(tf.float32, shape=(None, ), name='Y')

        # convert word indexes to word vectors
        # this would be equivalent to doing
        # We[tfX] in Numpy / Theano
        XW = tf.nn.embedding_lookup(We, self.tfX)

        # multiply it by input->hidden so we don't have to do
        # it inside recurrence
        XW_Wx = tf.matmul(XW, self.Wx)                
