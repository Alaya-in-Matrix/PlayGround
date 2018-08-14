import numpy as np
import cPickle as pickle

def branin(x):
    a = 1.0
    b = 5.1/(4*np.pi*np.pi)
    c = 5.0/np.pi
    r = 6.0
    s = 10.0
    t = 1/(8*np.pi)
    return a * ((x[1]-b*x[0]*x[0]+c*x[0]-r)**2) + s*(1-t)*np.cos(x[0]) + s

num_train = 10000
num_test = 100
dim = 2
train_x = np.random.randn(num_train, dim)
train_y = np.array([branin(train_x[i].reshape(dim)) for i in range(num_train)]) 

test_x = np.random.randn(num_test, dim)
test_y = np.array([branin(test_x[i].reshape(dim)) for i in range(num_test)])

dataset = {}
dataset['train_x'] = train_x.T
dataset['test_x'] = test_x.T
dataset['train_y'] = train_y.reshape(1,train_y.size)
dataset['test_y'] = test_y.reshape(1,test_y.size)

with open('branin.pickle','wb') as f:
    pickle.dump(dataset,f)


