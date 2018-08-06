from GP_model import GP_model
from activations import *
import sys
import toml
import autograd.numpy as np
import cPickle as pickle
from GP_model import scale_x
import matplotlib.pyplot as plt

argv = sys.argv[1:]
conf = toml.load(argv[0])

# configurations

activation = conf['activation']
l1 = conf['l1']
l2 = conf['l2']
scale = conf['scale']
num_layers = conf['num_layers']
layer_size = conf['layer_size']
max_iter = conf['max_iter']
directory = conf['directory']

act_f = relu
if activation == 'tanh':
    act_f = tanh
elif activation == 'sigmoid':
    act_f = sigmoid

with open(directory,'rb') as f:
    dataset = pickle.load(f)

train_x = dataset['train_x']
test_x = dataset['test_x']
train_y = dataset['train_y'][0]
test_y = dataset['test_y'][0]

act = [act_f] * num_layers
layer_sizes = [layer_size] * num_layers

model = GP_model(train_x, train_y, layer_sizes, act, bfgs_iter=max_iter, l1=l1, l2=l2, debug=True)

theta0 = model.rand_theta(scale=scale)

model.fit(theta0)

py, ps2 = model.predict(test_x)
print 'py'
print py
print 'ps2'
print ps2
print 'delta'
delta = py - test_y
print np.dot(delta, delta.T)

