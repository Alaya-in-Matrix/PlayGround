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
train_y = dataset['train_y'][0]
train_y = train_y.reshape(1, train_y.size)
test_x = dataset['test_x']
test_y = dataset['test_y'][0]
test_y = test_y.reshape(1, test_y.size)

model = GP_model(train_x, train_y, layer_sizes, act)
theta = model.rand_theta()

w = theta[2+model.dim:]
Phi = model.calc_Phi(w, model.train_x)

print 'num_train:',model.num_train
print 'layer_sizes:',model.layer_sizes
print 'm:',model.m
print 'num_param:',model.num_param
print 'mean:',model.mean
print 'std:',np.std(model.train_y)



