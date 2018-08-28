import numpy as np
# import cPickle as pickle

def branin(x):
    a = 1.0
    b = 5.1/(4*np.pi*np.pi)
    c = 5.0/np.pi
    r = 6.0
    s = 10.0
    t = 1/(8*np.pi)
    y = np.zeros((1,x.shape[1]))
    for i in range(x.shape[1]):
        y[0,i] = a * ((x[1,i]-b*x[0,i]*x[0,i]+c*x[0,i]-r)**2) + s*(1-t)*np.cos(x[0,i]) + s
    return y

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
    
    y = np.zeros((1,x.shape[1]))
    for i in range(x.shape[1]):
        tmp = (A*np.square(x[:,i] - P)).sum(axis=1)
        radian = np.exp(-tmp).reshape(tmp.size,1)
        y[0,i] = -np.dot(alpha,radian).sum()

    return y

def optCase(x):
    y = np.zeros((2, x.shape[1]))
    for i in range(x.shape[1]):
        y[0,i] = x[0,i]*x[0,i]+(x[1,i]-1)*(x[1,i]-1)+(x[2,i]+1)*(x[2,i]+1)*(x[2,i]-1)*(x[2,i]+2)
        y[1,i] = x[2,i]-x[1,i]*x[1,i]
    return y

def game1(x):
    y = np.zeros((10,x.shape[1]))
    for i in range(x.shape[1]):
        y[0,i] = 5*x[:4,i].sum() - 5*np.square(x[:4,i]).sum() - x[4:,i].sum()
        y[1,i] = 2*x[0,i]+2*x[1,i]+x[9,i]+x[10,i]-10
        y[2,i] = 2*x[0,i]+2*x[2,i]+x[9,i]+x[11,i]-10
        y[3,i] = 2*x[1,i]+2*x[2,i]+x[10,i]+x[11,i]-10
        y[4,i] = -8*x[0,i]+x[9,i]
        y[5,i] = -8*x[1,i]+x[10,i]
        y[6,i] = -8*x[2,i]+x[11,i]
        y[7,i] = -2*x[3,i]-x[4,i]+x[9,i]
        y[8,i] = -2*x[5,i]-x[6,i]+x[10,i]
        y[9,i] = -2*x[7,i]-x[8,i]+x[11,i]
    return y

def get_dataset(main_f, num_train, num_test, dim, outdim, bounds):
    train_x = np.zeros((dim, num_train))
    test_x = np.zeros((dim, num_test))
    for i in range(len(bounds)):
        train_x[i] = np.random.uniform(bounds[i][0], bounds[i][1], (num_train))
        test_x[i] = np.random.uniform(bounds[i][0], bounds[i][1], (num_test))

    train_y = main_f(train_x)
    test_y = main_f(test_x)

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

