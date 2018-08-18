import sys
import toml
from get_dataset import *
from Constr_model import Constr_model

argv = sys.argv[1:]
conf = toml.load(argv[0])

l1 = conf['l1']
l2 = conf['l2']
scale = conf['scale']
num_layers = conf['num_layers']
layer_size = conf['layer_size']
act = conf['activation']
max_iter = conf['max_iter']
bounds = conf['bounds']
dim = conf['dim']
outdim = conf['outdim']
num_train = conf['num_train']
num_test = conf['num_test']
funct = conf['funct']


main_f = get_main_f(funct)

dataset = get_dataset(main_f, num_train, num_test, dim, outdim, bounds)

all_x = np.zeros((dim,1))
all_loss = np.inf
for i in range(100):
    model = Constr_model(main_f, dataset, dim, outdim, bounds,scale,num_layers,layer_size,act,max_iter,l1=l1,l2=l2,debug=True)
    for j in range(10):
        x0 = model.rand_x()
        x0 = model.fit(x0)
        best_loss = np.inf
        best_x = np.zeros((model.dim,1))
        if main_f(x0,0) < best_loss:
            best_loss = main_f(x0,0)
            best_x = x0.copy()
    if main_f(best_x,0) < all_loss:
        all_loss = main_f(best_x,0)
        all_x = best_x
    print('all_x',all_x.T,'true',main_f(all_x,0))
    print('----------------------------------------------------------------------')
    dataset['train_x'] = np.concatenate((dataset['train_x'].T, best_x.T)).T
    y = np.zeros((1,outdim))
    for i in range(outdim):
        y[0,i] = main_f(best_x,0)
    dataset['train_y'] = np.concatenate((dataset['train_y'].T, y)).T

all_x += np.random.randn(dim,1)*0.001
x0 = model.fit(all_x)
print('all_x',all_x.T,'true',main_f(all_x,0))



