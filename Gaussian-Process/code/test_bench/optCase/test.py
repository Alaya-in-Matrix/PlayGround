import numpy as np
import cPickle as pickle

def assumed_func(x):
    return x[0]+(x[1]-1)*(x[1]-1)+(x[2]+1)*(x[2]+1)*(x[2]-1)

num_train = 1000
num_test = 68
dim = 3

x1 = np.random.randn(num_train) * 5
x2 = np.random.randn(num_train) * 3
x3 = np.random.randn(num_train) * 2.5

train_x = np.array([x1, x2, x3])
train_y = np.array([assumed_func(train_x[:,i]) for i in range(num_train)])

print train_x.shape
print train_y.shape

x1 = np.random.randn(num_test) * 5
x2 = np.random.randn(num_test) * 3
x3 = np.random.randn(num_test) * 2.5

test_x = np.array([x1, x2, x3])
test_y = np.array([assumed_func(test_x[:,i]) for i in range(num_test)])

print test_x.shape
print test_y.shape

dataset = {}
dataset['train_x'] = train_x
dataset['test_x'] = test_x
dataset['train_y'] = train_y.reshape(1,train_y.size)
dataset['test_y'] = test_y.reshape(1,test_y.size)

with open('test.pickle','wb') as f:
    pickle.dump(dataset, f)

