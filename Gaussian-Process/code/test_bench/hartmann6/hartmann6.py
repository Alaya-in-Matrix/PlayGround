import numpy as np
import cPickle as pickle

def hartmann6(x):
    A = np.array([[10.0,3.0,17.0,3.5,1.7,8.0],
        [0.05,10.0,17.0,0.1,8.0,14.0],
        [3.0,3.5,1.7,10.0,17.0,8.0],
        [17.0,8.0,0.05,10.0,0.1,14.0]])
    P = np.array([[0.1312,0.1696,0.5569,0.0124,0.8283,0.5886],
        [0.2329,0.4135,0.8307,0.3736,0.1004,0.9991],
        [0.2348,0.1451,0.3522,0.2883,0.3047,0.665],
        [0.4047,0.8828,0.8732,0.5743,0.1091,0.0381]])
    alpha = np.array([[1.0,1.2,3.0,3.2]])


    tmp = (A*np.square(x - P)).sum(axis=1)
    radian = np.exp(-tmp).reshape(tmp.size,1)
    result = -np.dot(alpha,radian).sum()

    return result


num_train = 10000
num_test = 100
dim = 6

train_x = np.random.rand(num_train,dim)
test_x = np.random.rand(num_test,dim)
train_y = np.array([hartmann6(train_x[i]) for i in range(num_train)])
test_y = np.array([hartmann6(test_x[i]) for i in range(num_test)])

print train_x.shape
print train_y.shape
print test_x.shape
print test_y.shape

dataset = {}
dataset['train_x'] = train_x.T
dataset['test_x'] = test_x.T
dataset['train_y'] = train_y.reshape(1,train_y.size)
dataset['test_y'] = test_y.reshape(1,test_y.size)

with open('hartmann.pickle','wb') as f:
    pickle.dump(dataset,f)






