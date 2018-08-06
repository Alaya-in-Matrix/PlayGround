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
test_x = dataset['test_x']
test_y = dataset['test_y'][0]

model = GP_model(train_x, train_y, layer_sizes, act, bfgs_iter=200, l1=0, l2=0, debug=True)

print 'num_train:',model.num_train
print 'm:',model.m
print 'num_param:',model.num_param
print 'mean:',model.mean
print 'std:',np.std(model.train_y_zero)

theta = model.rand_theta()
sn2, sp2, log_lscale, w = model.split_theta(theta)
w_nobias = model.nn.w_nobias(w, model.dim)

print
print 'sn2:', sn2
print 'sp2:', sp2
print 'log_lscale:', log_lscale.shape
print 'w:', w.shape
print 'w_nobias:', w_nobias.shape
print 'l1_reg:', np.abs(w_nobias).sum()
print 'l2_reg:', np.dot(w_nobias, w_nobias)

print
print 'log_likelihood:', model.log_likelihood(theta)
model.fit(theta)




