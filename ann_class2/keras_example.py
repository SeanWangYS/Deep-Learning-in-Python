from keras.models import Sequential
from keras.layers import Dense, Activation
from util import get_normalized_data, y2indicator

import matplotlib.pyplot as plt

# get the data, same as Theano + Tensorflow examples
# no need to split now, the fit() function will do it
Xtrain, Xtest, Ytrain, Ytest = get_normalized_data()

N, D = Xtrain.shape
K = len(set(Ytrain))

Ytrain = y2indicator(Ytrain)
Ytest = y2indicator(Ytest)

model = Sequential()

model.add(Dense(units=500, input_dim=D))
model.add(Activation('relu'))
model.add(Dense(units=300))
model.add(Activation('relu'))
model.add(Dense(units=K))
model.add(Activation('softmax'))


model.compile(
  loss='categorical_crossentropy',
  optimizer='adam', 
  metrics=['accuracy']
)

r = model.fit(Xtrain, Ytrain, validation_data=(Xtest, Ytest), epochs=20, batch_size=32)
print("Returned:", r)

print(r.history.keys())

plt.plot(r.history['acc'], label='acc')
plt.plot(r.history['val_acc'], label='val_acc')
plt.legend()
plt.show()