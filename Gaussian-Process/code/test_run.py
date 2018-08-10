import autograd.numpy as np
from GP_model import GP_model
import cPickle as pickle
from activations import *
from autograd import grad
from scipy.optimize import fmin_l_bfgs_b
import toml
import sys
import traceback
from Constr_model import Constr_model

def get_act_f(act):
    act_f = relu
    if act == 'tanh':
        act_f = tanh
    elif act == 'sigmoid':
        act_f = sigmoid
    return act_f

def construct_model(funct, directory, num_layers, layer_size, act, max_iter=200, l1=0.0, l2=0.0, debug=True):
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

    return main

def optimize(x):
    py, ps2 = main.predict(x)
    best_y = py
    best_x = np.copy(x)
    best_loss = np.inf
    def loss(x):
        py, ps2 = main.predict(x)
        ps = np.sqrt(ps2)
        py = py.sum()
        ps2 = ps2.sum()
        EI = (best_y - py)*cdf((best_y - py)/ps) + ps * pdf((best_y - py)/ps)
        if py < best_y:
            best_y = py
        py, ps2 = constrain1.predict(np.array([x[0]]).reshape(1,1))
        py = py.sum()
        ps2 = ps2.sum()
        constr1 = cdf(-py / np.sqrt(ps2))
        py, ps2 = constrain2.predict(np.array([x[1]]).reshape(1,1))
        py = py.sum()
        ps2 = ps2.sum()
        constr2 = cdf(-py / np.sqrt(ps2))
        py, ps2 = constrain3.predict(np.array([x[2]]).reshape(1,1))
        py = py.sum()
        ps2 = ps2.sum()
        constr3 = cdf(-py / np.sqrt(ps2))
        loss = EI * constr1 * constr2 * constr3
        if loss < best_loss:
            best_loss = loss.copy()
            best_x = x.copy()
        return loss

    gloss = grad(loss)

    try:
        fmin_l_bfgs_b(loss, x, gloss, maxiter=200, m=100, iprint=1)
    except np.linalg.LinAlgError:
        print('Increase noise term and re-oprimization')
        x0 = np.copy(best_x)
        x[0] -= 0.01
        try:
            fmin_l_bfgs_b(loss, x0, gloss, maxiter=200, m=10, iprint=1)
        except:
            print('Exception caught, L-BFGS early stopping...')
            print(traceback.format_exc())
    except:
        print('Exception caught, L-BFGS early stopping...')
        print(traceback.format_exc())

    print('Optimized loss is %g' % loss(best_x))
    print best_x

argv = sys.argv[1:]
conf = toml.load(argv[0])

# configurations
l1 = conf['l1']
l2 = conf['l2']
scale = conf['scale']
max_iter = conf['max_iter']
act = conf['activation']
num_layers = conf['num_layers']
layer_size = conf['layer_size']

constrain_l1 = conf['constrain_l1']
constrain_l2 = conf['constrain_l2']
constrain_scale = conf['constrain_scale']
constrain_max_iter = conf['constrain_max_iter']
constrain_act = conf['constrain_activation']
constrain_num_layers = conf['constrain_num_layers']
constrain_layer_size = conf['constrain_layer_size']

directory = conf['directory']

## main

main = construct_model('main_function', directory, num_layers, layer_size, act, max_iter, l1, l2)

## constrain1

constrain1 = construct_model('constrain1',directory, constrain_num_layers, constrain_layer_size, constrain_act, constrain_max_iter, constrain_l1, constrain_l2)

## constrain2

constrain2 = construct_model('constrain2',directory, constrain_num_layers, constrain_layer_size, constrain_act, constrain_max_iter, constrain_l1, constrain_l2)

## constrain3

constrain3 = construct_model('constrain3',directory, constrain_num_layers, constrain_layer_size, constrain_act, constrain_max_iter, constrain_l1, constrain_l2)

## rand_x

x = np.random.randn(3,1)

print 'x', x
py, ps2 = main.predict(x)
print 'py',py,'ps2',ps2
print x[0] + (x[1]-1)*(x[1]-1) + (x[2]+1)*(x[2]+1)*(x[2]-1)

model = Constr_model(x, main, [constrain1, constrain2, constrain3])
model.optimize()
