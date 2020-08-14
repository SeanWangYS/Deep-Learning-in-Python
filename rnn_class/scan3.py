import numpy as np 
import matplotlib.pyplot as plt
import theano 
import theano.tensor as T 

'''
Learn one more theano function => scan()
sample code from tutorial is at hmm_class file

low pass filter.
這個就是 exponential smoothing
'''

X = 2*np.random.randn(300) + np.sin(np.linspace(0, 3*np.pi, 300)) # create a noisy signal
plt.plot(X)
plt.title('original')
plt.show()

decay = T.scalar('T')
sequence = T.vector('sequence') # for noisy signal 

def recurrence(x, last, decay):  # "last" means the last value of our clean up sequence
    return (1-decay)*x + decay*last

outputs, _ = theano.scan(
    fn=recurrence, 
    sequences=sequence, 
    n_steps=sequence.shape[0], 
    outputs_info=[np.float64(0)], # for "last" recurrent argument
    non_sequences=[decay]
)


lpf = theano.function(
  inputs=[sequence, decay],
  outputs=outputs,
)

Y = lpf(X, 0.99)
plt.plot(Y)
plt.title("filtered")
plt.show()