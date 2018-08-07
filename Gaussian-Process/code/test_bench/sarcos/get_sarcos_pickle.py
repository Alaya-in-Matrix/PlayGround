import numpy as np
import cPickle as pickle
import scipy.io as sio

dim = 21

train = sio.loadmat('./sarcos_inv.mat')

train = train['sarcos_inv'].T
train_x = train[:dim]
train_y = train[dim:]

test = sio.loadmat('./sarcos_inv_test.mat')
test = test['sarcos_inv_test'].T
test_x = test[:dim]
test_y = test[dim:]

dataset = {}

dataset['train_x'] = train_x
dataset['train_y'] = train_y
dataset['test_x'] = test_x
dataset['test_y'] = test_y

# save dataset as sarcos.pickle
with open('sarcos.pickle','wb') as f:
    pickle.dump(dataset, f)




