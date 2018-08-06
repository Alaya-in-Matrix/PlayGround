import autograd.numpy as np

def tanh(x):
    return np.tanh(x)

def relu(x):
    return np.maximum(0,x)

def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))
