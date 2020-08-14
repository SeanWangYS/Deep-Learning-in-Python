import numpy as np
import theano 
import theano.tensor as T 

'''
Learn one more theano function => scan()
sample code from tutorial is at hmm_class file

Fibonacci 
'''

N = T.iscalars('N')

def recurrence(n, fn_1, fn_2):
    return fn_1 + fn_2, fn_1  

outputs, updates = theano.scan(   
    # 這裡的outputs會是兩個scale， 但經過 iterator 會形成兩個list
    # 因為 recurrence的 return有兩個參數，經過scan跑出來會是一個list內部包著兩個array
    fn=recurrence, 
    sequences=T.arange(N), 
    n_steps=N, 
    outputs_info=[1., 1.]
)

fabonacci = theano.function(
    inputs=[N], 
    outputs=outputs,   # 這個參數參考到scan function 的output，output的[] 可加可不加
)

o_val = fabonacci(8)  

print('output: ', o_val)
