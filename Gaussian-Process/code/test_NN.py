from NN import NN
from activations import *
import numpy as np

layer_sizes = [4,5,6]
activations = [tanh, sigmoid, relu]
x = np.random.randn(10,20)
dim, num_train = x.shape


nn = NN(layer_sizes, activations)
num_param = nn.num_param(dim)
print num_param

w = np.random.randn(num_param)
w_nobias = nn.w_nobias(w, dim)
print w_nobias.size

predict = nn.predict(w, x)
print predict.shape
