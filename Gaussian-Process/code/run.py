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
iteration = conf['iter']
K = conf['K']

main_f = get_main_f(funct)

dataset = get_dataset(main_f, num_train, num_test, dim, outdim, bounds)

all_constr = np.inf
all_loss = np.inf
all_x = np.zeros((dim,1))
for i in range(iteration):
    model = Constr_model(main_f, dataset, dim, outdim, bounds,scale,num_layers,layer_size,act,max_iter,l1=l1,l2=l2,debug=True)
    best_constr = np.inf
    best_loss = np.inf
    best_x = np.zeros((model.dim,1))
    for j in range(K):
        x0 = model.rand_x()
        x0 = model.fit(x0)
        p = main_f(x0)[:,0].T
        if best_constr > 0 and p[1:].sum() < best_constr:
            best_constr = p[1:].sum()
            best_loss = p[0]
            best_x = x0.copy()
        elif best_constr <= 0 and p[0] < best_loss:
            best_constr = p[1:].sum()
            best_loss = p[0]
            best_x = x0.copy()
    if all_constr > 0 and best_constr < all_constr:
        all_constr = best_constr
        all_loss = best_loss
        all_x = best_x.copy()
    elif all_constr <= 0 and best_loss < all_loss:
        all_constr = best_constr
        all_loss = best_loss
        all_x = best_x.copy()
    print('all_x',all_x.T)
    print('true',main_f(all_x).T)
    print('-----------------------------------------------------------------------------')
    dataset['train_x'] = np.concatenate((dataset['train_x'].T, best_x.T)).T
    dataset['train_y'] = np.concatenate((dataset['train_y'].T, main_f(best_x).T)).T

x0 = all_x + np.random.randn(dim,1)*0.001
x0 = model.fit(x0)
print('all_x',all_x.T)
print('true',main_f(all_x).T)

