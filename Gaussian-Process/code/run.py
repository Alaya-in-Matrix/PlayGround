import DeepSparseKernel as dsk
from DeepSparseKernel import np
import matplotlib.pyplot as plt
import cPickle as pickle

'''
train_x = np.loadtxt('train_x')
train_y = np.loadtxt('train_y')
train_y = train_y.reshape(1, train_y.size)

test_x = np.loadtxt('test_x')
test_y = np.loadtxt('test_y')
test_y = test_y.reshape(1, test_y.size)
'''

with open('./test_bench/enb/enb.pickle','rb') as f:
    dataset = pickle.load(f)

train_x = dataset['train_x']
test_x = dataset['test_x']
train_y = dataset['train_y'][0]
train_y = train_y.reshape(1, train_y.size)
test_y = dataset['test_y'][0]
test_y = test_y.reshape(1, test_y.size)

'''
num_train = train_x.shape[0]
num_test  = test_x.shape[0]
dim       = int(train_x.size / num_train)
train_x   = train_x.reshape(num_train, dim).T;
test_x    = test_x.reshape(num_test,  dim).T;

print(dim)
print(train_x.shape)
print(test_x.shape)
'''

dim, num_train = train_x.shape
num_test = test_x.shape[1]

layer_sizes = [50, 50, 50, 50]
activations = [dsk.relu, dsk.tanh, dsk.relu, dsk.tanh]
scale       = 0.1

dim = train_x.shape[0]

gp    = dsk.DSK_GP(train_x, train_y, layer_sizes, activations, bfgs_iter=200, l1=0, l2=0.0, debug=True);
theta = gp.rand_theta(scale=scale)
gp.fit(theta)
py, ps2             = gp.predict(test_x)
py_train, ps2_train = gp.predict(train_x)


log_lscales = gp.theta[2:2+dim];
Phi_train   = gp.calc_Phi(gp.theta[2+dim:], dsk.scale_x(train_x, log_lscales));
Phi_test    = gp.calc_Phi(gp.theta[2+dim:], dsk.scale_x(test_x, log_lscales));

np.savetxt('pred_y', py)
np.savetxt('pred_s2', ps2)
np.savetxt('theta', gp.theta)
np.savetxt('Phi_train', Phi_train)
np.savetxt('Phi_test', Phi_test)

# plt.plot(test_y.reshape(test_y.size), py.reshape(py.size), 'r.', train_y.reshape(train_y.size), py_train.reshape(train_y.size), 'b.')
# plt.show()

gp.debug = True
print(gp.log_likelihood(gp.theta))
np.savetxt('loss', gp.log_likelihood(gp.theta))


