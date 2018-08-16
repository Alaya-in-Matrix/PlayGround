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

## test bench optCase main_function
def main_function(x):
    return x[0]*x[0]+(x[1]-1)*(x[1]-1)+(x[2]+1)*(x[2]+1)*(x[2]-1)*(x[2]+2)

argv = sys.argv[1:]
conf = toml.load(argv[0])

# configurations
directory = conf['directory']

num_layers = [conf['num_layers'], conf['constrain_num_layers']]
layer_size = [conf['layer_size'], conf['constrain_layer_size']]
act = [conf['activation'], conf['constrain_activation']]
max_iter = [conf['max_iter'], conf['constrain_max_iter']]
l1 = [conf['l1'], conf['constrain_l1']]
l2 = [conf['l2'], conf['constrain_l2']]
main_funct = conf['main_funct']
constr = conf['constr']
bounds = np.array(conf['bounds'])

model = Constr_model(main_funct, constr, directory, bounds, num_layers, layer_size, act, max_iter, l1, l2)

for i in range(5):
    x0 = model.rand_x(scale=0.1)
    model.fit(x0)
    print('true',main_function(model.x))


