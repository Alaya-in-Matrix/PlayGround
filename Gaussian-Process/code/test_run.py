import autograd.numpy as np
from GP_model import GP_model
import cPickle as pickle
from activations import *
from autograd import grad
from scipy.optimize import fmin_l_bfgs_b

with open('./test_bench/optCase/test.pickle','rb') as f:
    dataset = pickle.load(f)

train_x = dataset['train_x']
train_y = dataset['train_y'][0]
test_x = dataset['test_x']
test_y = dataset['test_y'][0]

num_layers = 3
layer_size = 100
scale = 0.4

layer_sizes = [layer_size] * num_layers
activation = [relu] * num_layers

model = GP_model(train_x, train_y, layer_sizes, activation, bfgs_iter=200, l1=0.0, l2=0.0, debug=True)

theta0 = model.rand_theta(scale=scale)

model.fit(theta0)

py, ps2 = model.predict(test_x)

print 'delta'
delta = py - test_y
print delta

print 'square error'
print np.dot(delta, delta.T)

x = np.random.randn(train_x.shape[0],1)
x[0] = x[0]*5
x[1] = x[1]*3
x[2] = x[2]*2.5

'''
model.optimize(x)

print model.opt
print model.py
print model.x
print model.predict(x)
'''

