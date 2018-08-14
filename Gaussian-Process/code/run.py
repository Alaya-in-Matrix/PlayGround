from GP_model import GP_model
from activations import *
import sys
import toml
import autograd.numpy as np
import cPickle as pickle
from GP_model import scale_x
import matplotlib.pyplot as plt

## optCase main_function
def main_f(x):
    return x[0]*x[0]+(x[1]-1)*(x[1]-1)+(x[2]+1)*(x[2]+1)*(x[2]-1)*(x[2]+2)

## test bench branin
def branin(x):
    a = 1.0
    b = 5.1/(4*np.pi*np.pi)
    c = 5.0/np.pi
    r = 6.0
    s = 10.0
    t = 1/(8*np.pi)
    return a * ((x[1]-b*x[0]*x[0]+c*x[0]-r)**2) + s*(1-t)*np.cos(x[0]) + s

def construct_model(funct, directory, bounds, num_layers, layer_size, act, max_iter=200, l1=0.0, l2=0.0, debug=True):
    with open(directory+funct+'.pickle','rb') as f:
        dataset = pickle.load(f)

    train_x = dataset['train_x']
    train_y = dataset['train_y'][0]
    test_x = dataset['test_x']
    test_y = dataset['test_y'][0]

    activations = [get_act_f(act)]*num_layers
    layer_sizes = [layer_size]*num_layers

    main = GP_model(train_x, train_y, layer_sizes=layer_sizes, activations=activations, bfgs_iter=max_iter, l1=l1, l2=l2, debug=True)
    theta0 = main.rand_theta()
    main.fit(theta0)

    py, ps2 = main.predict(test_x)
    print 'py'
    print py
    print 'ps2'
    print ps2
    print 'delta'
    delta = py - test_y
    print delta
    print 'square error', np.dot(delta, delta.T)
    
    for i in range(5):
        x = np.random.randn(main.dim,1)
        main.optimize(x, bounds)
        print 'true',main_f(main.x)

    return main

argv = sys.argv[1:]
conf = toml.load(argv[0])

# configurations

activation = conf['activation']
l1 = conf['l1']
l2 = conf['l2']
scale = conf['scale']
num_layers = conf['num_layers']
layer_size = conf['layer_size']
max_iter = conf['max_iter']
directory = conf['directory']
funct = conf['main_funct']
bounds = conf['bounds']

model = construct_model(funct,directory,bounds,num_layers, layer_size, activation, max_iter, l1, l2, debug=True)

