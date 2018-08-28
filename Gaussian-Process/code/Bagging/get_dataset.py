import numpy as np
import pickle
import os

def test2(x):
    conf_file = './test_bench/test2/conf'
    param_file = './test_bench/test2/circuit/param'
    result_file = './test_bench/test2/circuit/result.po'
    name = []
    for l in open(conf_file, 'r'):
        l = l.strip().split(' ')
        if l[0] == 'des_var':
            name.append(l[1])

    y = np.zeros((3, x.shape[1]))
    for p in range(x.shape[1]):
        with open(param_file, 'w') as f:
            for i in range(len(x)):
                f.write('.param '+name[i]+' = '+str(x[i, p])+'\n')

        os.system('bash sim2.sh')

        with open(result_file, 'r') as f:
            line = f.readline().strip().split(' ')
            for i in range(len(line)):
                y[i, p] = float(line[i])

    return y

def circuit(x):
    # get param name
    conf_file = './test_bench/circuit/conf'
    param_file = './test_bench/circuit/circuit/param'
    result_file = './test_bench/circuit/circuit/result.po'
    name = []
    for l in open(conf_file, 'r'):
        l = l.strip().split(' ')
        if l[0] == 'des_var':
            name.append(l[1])
    
    y = np.zeros((7, x.shape[1])) 
    for p in range(x.shape[1]):
        # write out param file
        with open(param_file, 'w') as f:
            for i in range(len(x)):
                f.write('.param '+name[i]+' = '+str(x[i, p])+'\n')

        # hspice simulation
        os.system('bash sim.sh')

        # get results
        with open(result_file, 'r') as f:
            line = f.readline().strip().split(' ')
            for i in range(len(line)):
                y[i, p] = float(line[i])

    return y
>>>>>>> tmp/master

def branin(x):
    a = 1.0
    b = 5.1/(4*np.pi*np.pi)
    c = 5.0/np.pi
    r = 6.0
    s = 10.0
    t = 1.0/(8*np.pi)
    y = np.zeros((1, x.shape[1]))
    for i in range(x.shape[1]):
        y[0, i] = 2*((x[1, i]-b*x[0, i]*x[0, i]+c*x[0, i]-r)**2) + s*(1-t)*np.cos(x[0, i]) + s
    return y

def optCase(x):
    y = np.zeros((2, x.shape[1]))
    for i in range(x.shape[1]):
        y[0, i] = x[0, i]*x[0, i]+(x[1, i]-1)*(x[1, i]-1)+(x[2, i]+1)*(x[2, i]+1)*(x[2, i]-1)*(x[2, i]+2)
        y[1, i] = x[2, i]-x[1, i]*x[1, i]
    return y

def game1(x):
    y = np.zeros((10, x.shape[1]))
    for i in range(x.shape[1]):
        y[0, i] = 5*x[:4, i].sum() - 5*np.square(x[:4, i]).sum() - x[:4, i].sum()
        y[1, i] = 2*x[0, i]+2*x[1, i]+x[9, i]+x[10, i]-10
        y[2, i] = 2*x[0, i]+2*x[2, i]+x[9, i]+x[11, i]-10
        y[3, i] = 2*x[1, i]+2*x[2, i]+x[10, i]+x[11, i]-10
        y[4, i] = -8*x[0, i]+x[9, i]
        y[5, i] = -8*x[1, i]+x[10, i]
        y[6, i] = -8*x[2, i]+x[11, i]
        y[7, i] = -2*x[3, i]-x[4, i]+x[9, i]
        y[8, i] = -2*x[5, i]-x[6, i]+x[10, i]
        y[9, i] = -2*x[7, i]-x[8, i]+x[11, i]
    return y

def get_dataset(main_f, num_train, num_test, dim, outdim, bounds):
    train_x = np.zeros((dim, num_train))
    test_x = np.zeros((dim, num_test))
    for i in range(dim):
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
    elif funct == 'optCase':
        main_f = optCase
    elif funct == 'branin':
        main_f = branin
    elif funct == 'circuit':
        main_f = circuit
    else:
        main_f = test2
    return main_f


