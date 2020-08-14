import numpy as np
from sklearn.neural_network import MLPRegressor
from util import getKaggleMNIST


# auto-encoder in 1 line of code
# get data
X, _, Xt, _ = getKaggleMNIST()

# create the model and train it 
model = MLPRegressor()
model.fit(X, X)

# test the model
print("Train R^2:", model.score(X, X))
print("Test R^2:", model.score(Xt, Xt))

Xhat = model.predict(X)
mse = ((Xhat - X)**2).mean()
print("Train MSE:", mse)

Xhat = model.predict(Xt)
mse = ((Xhat - Xt)**2).mean()
print("Test MSE:", mse)