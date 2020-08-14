import numpy as np 
import theano 
import theano.tensor as T 

'''
Learn one more theano function => scan()
sample code from tutorial is at hmm_class file
'''

x = T.vector('x')  # the vectror we set up represent every element in the loop will be a scalar

def square(x):
    return x*x 

outputs, updates = theano.scan(
    fn=square, 
    sequences=x, 
    n_steps=x.shape[0]
)

square_op = theano.function(
    inputs = [x], 
    outputs = [outputs],   
)

o_val = square_op(np.array([1, 2, 3, 4, 5]))

print('output: ', o_val)