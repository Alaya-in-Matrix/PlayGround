import numpy as np
import cPickle as pickle

def branin(x,i):
    x = x.reshape(x.size)
    a = 1.0
    b = 5.1/(4*np.pi*np.pi)
    c = 5.0/np.pi
    r = 6.0
    s = 10.0
    t = 1/(8*np.pi)
    return a * ((x[1]-b*x[0]*x[0]+c*x[0]-r)**2) + s*(1-t)*np.cos(x[0]) + s

def hartmann6(x,i):
    x = x.reshape(x.size)
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

def optCase(x,i):
    x = x.reshape(x.size)
    if i == 0:
        return x[0]*x[0]+(x[1]-1)*(x[1]-1)+(x[2]+1)*(x[2]+1)*(x[2]-1)*(x[2]+2)
    elif i == 1:
        return x[2]-x[1]*x[1]

def game1(x,i):
    x = x.reshape(x)
    if i == 0:
        return 5*x[:4].sum() - 5*np.square(x[:4]).sum() - x[4:].sum()
    elif i == 1:
        return 2*x[0]+2*x[1]+x[9]+x[10]-10
    elif i == 2:
        return 2*x[0]+2*x[2]+x[9]+x[11]-10
    elif i == 3:
        return 2*x[1]+2*x[2]+x[10]+x[11]-10
    elif i == 4:
        return -8*x[0]+x[9]
    elif i == 5:
        return -8*x[1]+x[10]
    elif i == 6:
        return -8*x[2]+x[11]
    elif i == 7:
        return -2*x[3]-x[4]+x[9]
    elif i == 8:
        return -2*x[5]-x[6]+x[10]
    else:
        return -2*x[7]-x[8]+x[11]

def get_dataset(main_f, num_train, num_test, dim, outdim, bounds):
    train_x = np.zeros((dim, num_train))
    test_x = np.zeros((dim, num_test))
    for i in range(len(bounds)):
        train_x[i] = np.random.uniform(bounds[i][0], bounds[i][1], (num_train))
        test_x[i] = np.random.uniform(bounds[i][0], bounds[i][1], (num_test))

    train_y = np.zeros((outdim, num_train))
    test_y = np.zeros((outdim, num_test))
    for i in range(outdim):
        train_y[i] = np.array([main_f(train_x[:,j],i) for j in range(num_train)])
        test_y[i] = np.array([main_f(test_x[:,j],i) for j in range(num_test)])

    dataset = {}
    dataset['train_x'] = train_x
    dataset['test_x'] = test_x
    dataset['train_y'] = train_y
    dataset['test_y'] = test_y
    return dataset

def get_main_f(funct):
    if funct == 'game1':
        main_f = game1
    elif funct == 'hartmann6':
        main_f = hartmann6
    elif funct == 'optCase':
        main_f = optCase
    else:
        main_f = branin
    return main_f

