import sys
import toml
from get_dataset import *
from Bagging_Constr_model import Bagging_Constr_model
import multiprocessing

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
num_models = conf['num_models']

main_f = get_main_f(funct)

dataset = get_dataset(main_f, num_train, num_test, dim, outdim, bounds)

all_constr = np.inf
all_loss = np.inf
all_x = np.zeros((dim,1))
for i in range(dataset['train_x'].shape[1]):
    p = dataset['train_y'][:,i].T
    if all_constr > 0 and np.maximum(p[1:],0).sum() < all_constr:
        all_constr = np.maximum(p[1:],0).sum()
        all_loss = p[0]
        all_x = dataset['train_x'][:,i:i+1]
    elif all_constr <= 0 and np.maximum(p[1:],0).sum() <= 0 and p[0] < all_loss:
        all_constr = np.maximum(p[1:],0).sum()
        all_loss = p[0]
        all_x = dataset['train_x'][:,i:i+1]
print('all_constr',all_constr)
print('all_loss',all_loss)
print('all_x',all_x.T)
print('true',main_f(all_x).T)
print('-----------------------------------------------------------------------------')

for i in range(iteration):
    model = Bagging_Constr_model(num_models, main_f, dataset, dim, outdim, bounds,scale,num_layers,layer_size,act,max_iter,l1=l1,l2=l2,debug=True)
    def task(i):
        x0 = model.rand_x()
        x0 = model.fit(x0)
        p, _ = model.predict(x0)
        return x0, p[0]
    pool = multiprocessing.Pool(processes=5)
    results = pool.map(task, range(K))
    pool.close()   
    best_constr = np.inf
    best_loss = np.inf
    best_x = np.zeros((model.dim,1))
    for j in range(K):
        p = results[j][1]
        x0 = results[j][0]
        if best_constr > 0 and np.maximum(p[1:],0).sum() < best_constr:
            best_constr = np.maximum(p[1:],0).sum()
            best_loss = p[0]
            best_x = x0.copy()
        elif best_constr <= 0 and np.maximum(p[1:],0).sum() <= 0 and p[0] < best_loss:
            best_constr = np.maximum(p[1:],0).sum()
            best_loss = p[0]
            best_x = x0.copy()
    p = main_f(best_x)[:,0].T
    if all_constr > 0 and np.maximum(p[1:],0).sum() < all_constr:
        all_constr = np.maximum(p[1:],0).sum()
        all_loss = p[0]
        all_x = best_x.copy()
    elif all_constr <= 0 and np.maximum(p[1:],0).sum() <= 0 and p[0] < all_loss:
        all_constr = np.maximum(p[1:],0).sum()
        all_loss = p[0]
        all_x = best_x.copy()
    print('all_constr',all_constr)
    print('all_loss',all_loss)
    print('all_x',all_x.T)
    print('true',main_f(all_x).T)
    print('-----------------------------------------------------------------------------')
    dataset['train_x'] = np.concatenate((dataset['train_x'].T, best_x.T)).T
    dataset['train_y'] = np.concatenate((dataset['train_y'].T, main_f(best_x).T)).T

x0 = all_x + np.random.randn(dim,1)*0.001
x0 = model.fit(x0)
print('all_constr',all_constr)
print('all_loss',all_loss)
print('all_x',all_x.T)
print('true',main_f(all_x).T)

