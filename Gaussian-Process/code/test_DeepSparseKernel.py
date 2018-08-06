import numpy as np
import cPickle as pickle
from DeepSparseKernel import DSK_GP
from activations import *

with open('./test_bench/enb/enb.pickle','rb') as f:
    dataset = pickle.load(f)

train_x = dataset['train_x']
train_y = dataset['train_y'][0]
test_x = dataset['test_x']
test_y = dataset['test_y'][0]

layer_sizes = [100,100,100]
act = [relu, relu, relu]
dim = 8

model = DSK_GP(train_x, train_y, layer_sizes, act)
theta = model.rand_theta()
print theta.shape
print model.num_param

w = theta[2+model.dim:]
Phi = model.calc_Phi(w, model.train_x)

print 'dim:',model.dim
print 'num_train:',model.num_train
print 'm:',model.m
print 'num_param:',model.num_param
print 'mean:',model.mean
print 'std:',np.std(model.train_y)

loss = model.log_likelihood(theta)
print loss

model.fit(theta)

py, ps2 = model.predict(test_x)
print py
print ps2[0]

