from   DeepSparseKernel  import np
import matplotlib.pyplot as plt
import sys
import DeepSparseKernel  as dsk
import toml
import cPickle as pickle

def trans(data):
    if data.ndim == 1:
        return data.reshape(data.size, 1)
    else:
        return data

argv = sys.argv[1:]
conf = toml.load(argv[0])

# configurations
num_shared_layer     = conf["num_shared_layer"]
num_non_shared_layer = conf["num_non_shared_layer"]
hidden_shared        = conf["hidden_shared"]
hidden_non_shared    = conf["hidden_non_shared"]
l1                   = conf["l1"]
l2                   = conf["l2"]
scale                = conf["scale"]
max_iter             = conf["max_iter"]
K                    = conf["K"]
activation           = conf["activation"];

act_f = dsk.tanh
if activation == "relu":
    act_f = dsk.relu
elif activation == "erf":
    act_f = dsk.erf
elif activation == "sigmoid":
    act_f = dsk.sigmoid
else:
    act_f = dsk.tanh

with open('./test_bench/enb/enb.pickle','rb') as f:
    dataset = pickle.load(f)

train_x = dataset['train_x']
test_x = dataset['test_x']
train_y = dataset['train_y'][0]
train_y = train_y.reshape(train_y.size, 1)
test_y = dataset['test_y'][0]
test_y = test_y.reshape(test_y.size, 1)
dim, num_train = train_x.shape
num_obj = train_y.shape[1]
num_test = test_x.shape[1]


print 'dim, num_train:', train_x.shape
print 'num_obj:', num_obj
print 'num_test:', num_test

'''
train_x              = trans(np.loadtxt('train_x')).T
train_y              = trans(np.loadtxt('train_y'))
test_x               = trans(np.loadtxt('test_x')).T
test_y               = trans(np.loadtxt('test_y'))
dim, num_train       = train_x.shape
num_obj              = train_y.shape[1]
num_test             = train_x.shape[1]
'''

shared_layers_sizes     = [hidden_shared]     * num_shared_layer
shared_activations      = [dsk.tanh]          * num_shared_layer
non_shared_layers_sizes = [hidden_non_shared] * num_non_shared_layer
non_shared_activations  = [dsk.tanh]          * num_non_shared_layer

shared_nn      = dsk.NN(shared_layers_sizes, shared_activations)
non_shared_nns = []

for i in range(num_obj):
    non_shared_nns += [dsk.NN(non_shared_layers_sizes, non_shared_activations)]

modsk = dsk.MODSK(train_x, train_y, shared_nn, non_shared_nns, debug=True, max_iter=max_iter, l1=l1, l2=l2)

py, ps2 = modsk.mix_predict(K, test_x, scale=scale);
np.savetxt('pred_y', py);
np.savetxt('pred_s2', ps2);
print("Finished")

