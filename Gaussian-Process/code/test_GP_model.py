import autograd.numpy as np
from activations import *
import cPickle as pickle
from GP_model import GP_model
from autograd import grad



act = [relu, relu, relu]
layer_sizes = [100,100,100]

with open('./test_bench/enb/enb.pickle','rb') as f:
    dataset = pickle.load(f)

train_x = dataset['train_x']
train_y = dataset['train_y']
test_x = dataset['test_x']
test_y = dataset['test_y']

model = GP_model(train_x, train_y[0], layer_sizes, act)
theta = model.rand_theta()

w = theta[2+model.dim:]
Phi = model.calc_Phi(w, model.train_x)

print 'num_layers:',model.num_layers
print 'dim:',model.dim
print 'num_train:',model.num_train
print 'layer_sizes:',model.layer_sizes
print 'm:',model.m
print 'num_param:',model.num_param
print 'mean:',model.mean
print 'std:',np.std(model.train_y)


loss = model.log_likelihood(theta)
print loss

model.fit(theta)

py, ps2 = model.predict(test_x)
m = np.mean(py)
t = np.mean(ps2) + np.mean(py * py) - m * m
print 'test mean:',py
print 'test_y:', test_y[0]
delta = py - test_y[0]
print 'delta:', delta
print np.mean(delta)
print np.std(delta)




