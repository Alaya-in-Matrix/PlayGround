import autograd.numpy as np
from autograd import grad
import traceback
from scipy.optimize import fmin_l_bfgs_b
from activations import * 
import sys
import cPickle as pickle

# fully connected neural network
class NN:
    def __init__(self, layer_sizes, activations):
        self.num_layers = np.copy(len(layer_sizes))
        self.layer_sizes = np.copy(layer_sizes)
        self.activations = activations

    def num_param(self, dim):
        '''
        get the parameter number of the neural network
        
        dim: input vector size
        layer_sizes only contains size of weights
        remember to take bias into consideration
        '''
        xs = [dim]
        results = 0
        for l in self.layer_sizes:
            xs.append(l)
        for i in range(self.num_layers):
            results += (1+xs[i])*xs[i+1]
        return results

    def w_nobias(self, w, dim):
        '''
        get the weights matrix for l1/l2 regularization

        w: weights + bias
        dim: input vector size

        w_layer: only contains weights, without bias
        '''
        prev_size = dim
        start_idx = 0
        wnb = np.array([])
        for i in range(self.num_layers):
            layer_size = self.layer_sizes[i]
            w_num_layer = (1+prev_size)*layer_size
            w_layer = np.reshape(w[start_idx:start_idx+w_num_layer],(prev_size+1,layer_size))[:prev_size]
            wnb = np.concatenate((wnb, w_layer.reshape(w_layer.size)))
            start_idx += w_num_layer
            prev_size = layer_size
        return wnb

    def predict(self, w, x):
        '''
        get the prediction results

        x.shape: dim, num_train
        w_layer.shape: dim, next_layer_dim
        out.shape: dim, num_train
        '''
        dim, num_train = x.shape
        out = x
        start_idx = 0
        prev_size = dim
        bias = np.ones((1, num_train))
        for i in range(self.num_layers):
            layer_size = self.layer_sizes[i]
            w_num_layer = (prev_size+1)*layer_size
            w_layer = np.reshape(w[start_idx:start_idx+w_num_layer], (prev_size+1,layer_size))
            out = np.concatenate((out, bias))
            out = self.activations[i](np.dot(w_layer.T, out))
            start_idx += w_num_layer
            prev_size = layer_size
        return out


def scale_x(log_lscale, x):
    lscale = np.exp(log_lscale).repeat(x.shape[1], axis=0).reshape(x.shape)
    return x/lscale

def chol_inv(L,y):
    '''
    K = L * L.T
    return inv(K)*y
    '''
    v = np.linalg.solve(L, y)
    return np.linalg.solve(L.T, v)

class GP_model:
    def __init__(self, train_x, train_y, layer_sizes, activations, bfgs_iter=100, l1=0, l2=0, debug=False):
        self.train_x = np.copy(train_x)
        self.train_y = np.copy(train_y)
        self.dim = train_x.shape[0]
        self.num_train = train_x.shape[1]
        self.nn = NN(layer_sizes, activations)
        self.num_param = 2 + self.dim + self.nn.num_param(self.dim)
        self.bfgs_iter = bfgs_iter
        self.l1 = l1
        self.l2 = l2
        self.debug = debug
        self.m = layer_sizes[-1]
        self.mean = np.mean(self.train_y)
        self.train_y_zero = self.train_y - self.mean
        self.loss = np.inf

    def rand_theta(self, scale=1):
        '''
        generate an initial theta, the weights of NN are randomly initialized
        '''
        theta = scale * np.random.randn(self.num_param)
        theta[0] = np.log(np.std(self.train_y_zero) / 2)
        theta[1] = np.log(np.std(self.train_y_zero))
        for i in range(self.dim):
            theta[2+i] = np.maximum(-100, np.log(0.5*(self.train_x[i].max() - self.train_x[i].min())))
        return theta

    def split_theta(self, theta):
        '''
        split the theta into sn2, sp2, log_lscale, ws
        '''
        log_sn = theta[0]
        log_sp = theta[1]
        log_lscale = theta[2:2+self.dim]
        ws = theta[2+self.dim:]
        sn2 = np.exp(2 * log_sn)
        sp2 = np.exp(2 * log_sp)
        return (sn2, sp2, log_lscale, ws)

    def calc_Phi(self, w, x):
        '''
        Phi.shape: self.m, self.num_train
        '''
        return self.nn.predict(w, x)

    def log_likelihood(self, theta):
        sn2, sp2, log_lscale, w = self.split_theta(theta)
        scaled_x = scale_x(log_lscale, self.train_x)
        Phi = self.calc_Phi(w, scaled_x)
        Phi_y = np.dot(Phi, self.train_y_zero.T)
        A = np.dot(Phi, Phi.T) + self.m * sn2 / sp2 * np.eye(self.m) # A.shape: self.m, self.m
        LA = np.linalg.cholesky(A)
        
        logDetA = 0
        for i in range(self.m):
            logDetA = 2 * np.log(LA[i][i])

        datafit = (np.dot(self.train_y_zero, self.train_y_zero.T) - np.dot(Phi_y.T, chol_inv(LA, Phi_y))) / sn2
        neg_likelihood = 0.5 * (datafit + self.num_train * np.log(2 * np.pi * sn2) + logDetA - self.m * np.log(self.m * sn2 / sp2))
        if(np.isnan(neg_likelihood)):
            neg_likelihood = np.inf

        w_nobias = self.nn.w_nobias(w, self.dim)
        l1_reg = self.l1 * np.abs(w_nobias).sum()
        l2_reg = self.l2 * np.dot(w_nobias, w_nobias.T)
        neg_likelihood += l1_reg + l2_reg

        if neg_likelihood < self.loss:
            self.loss = neg_likelihood
            self.theta = theta.copy()
            self.A = A.copy()
            self.LA = LA.copy()

        return neg_likelihood

    def fit(self, theta):
        self.loss = np.inf
        theta0 = np.copy(theta)

        def loss(theta):
            nlz = self.log_likelihood(theta)
            return nlz

        gloss = grad(loss)
        
        try:
            fmin_l_bfgs_b(loss, theta0, gloss, maxiter=self.bfgs_iter, m=100, iprint=1)
        except np.linalg.LinAlgError:
            print('Increase noise term and re-optimization')
            theta0 = np.copy(self.theta)
            theta0[0] += np.log(10)
            try:
                fmin_l_bfgs_b(loss, theta0, gloss, maxiter=self.bfgs_iter, m=10, iprint=1)
            except:
                print('Exception caught, L-BFGS early stopping...')
                if self.debug:
                    print(traceback.format_exc())
        except:
            print('Exception caught, L-BFGS early stopping...')
            if self.debug:
                print(traceback.format_exc())
                
        print('Optimized loss is %g' % self.loss)
        if(np.isinf(self.loss) or np.isnan(self.loss)):
            print('Fail to build GP model')
            sys.exit(1)

        sn2, sp2, log_lscale, w = self.split_theta(self.theta)
        Phi = self.calc_Phi(w, scale_x(log_lscale, self.train_x))
        self.alpha = chol_inv(self.LA, np.dot(Phi, self.train_y_zero.T))

    def predict(self, test_x):
        sn2, sp2, log_lscale, w = self.split_theta(self.theta)
        scaled_x = scale_x(log_lscale, test_x)
        Phi_test = self.calc_Phi(w, scaled_x)
        py = self.mean + np.dot(Phi_test.T, self.alpha)
        # ps2 = sn2 + sn2 * np.diagonal(np.dot(Phi_test.T, chol_inv(self.LA, Phi_test)))
        ps2 = sn2 + sn2 * np.dot(Phi_test.T, chol_inv(self.LA, Phi_test))
        return py, ps2
        

class Constr_model:
    def __init__(self, main_funct, constr, directory, num_layers, layer_size, act, max_iter, l1, l2):
        '''
        generate the main function model
        '''
        self.main_function = self.construct_model(main_funct, directory, num_layers=num_layers[0], layer_size=layer_size[0], act=act[0], max_iter=max_iter[0], l1=l1[0], l2=l2[0])
        '''
        generate the constrain model
        '''
        self.constr_list = []
        for i in constr:
            model = self.construct_model(i, directory, num_layers=num_layers[1], layer_size=layer_size[1], act=act[1], max_iter=max_iter[1], l1=l1[1], l2=l2[1])
            self.constr_list.append(model)
        # main function input dimension
        self.dim = self.main_function.dim

    def rand_x(self, scale=1.0):
        '''
        randomly generate a initial input
        '''
        return scale * np.random.randn(self.dim,1)

    def construct_model(self, funct, directory, num_layers, layer_size, act, max_iter=200, l1=0.0, l2=0.0, debug=True):
        def get_act_f(act):
            act_f = relu
            if act == 'tanh':
                act_f = tanh
            elif act == 'sigmoid':
                act_f = sigmoid
            elif act == 'erf':
                act_f = erf
            return act_f

        with open(directory+funct+'.pickle','rb') as f:
            dataset = pickle.load(f)

        train_x = dataset['train_x']
        train_y = dataset['train_y'][0]
        test_x = dataset['test_x']
        test_y = dataset['test_y'][0]

        activations = [get_act_f(act)]*num_layers
        layer_sizes = [layer_size]*num_layers

        model = GP_model(train_x, train_y, layer_sizes=layer_sizes, activations=activations, bfgs_iter=max_iter, l1=l1, l2=l2, debug=True)
        theta0 = model.rand_theta()
        model.fit(theta0)

        py, ps2 = model.predict(test_x)
        print 'py'
        print py
        print 'ps2'
        print ps2
        print 'delta'
        delta = py - test_y
        print delta
        print 'square error',np.dot(delta, delta.T)

        return model

    def fit(self, x):
        x0 = np.copy(x)
        self.x = x0
        self.loss = np.inf
        self.best_y = 0.0
        self.tmp_py = np.array([0.0])
        '''
        we need one particular variable self.tmp_py to store best_y temperately
        and set self.best_y = self.tmp_py in the next loop 
        '''
        def loss(x):
            self.best_y = self.tmp_py.sum()
            x = x.reshape(self.dim, x.size/self.dim)
            tmp_py, ps2 = self.main_function.predict(x)
            py = tmp_py.sum()
            ps = np.sqrt(ps2.sum())
            tmp = (self.best_y - py)/ps
            EI = (self.best_y - py)*cdf(tmp) + ps*pdf(tmp)
            EI = -np.log(EI)
            for i in range(len(self.constr_list)):
                py, ps2 = self.constr_list[i].predict(x[i:i+1])
                py = py.sum()
                ps = np.sqrt(ps2.sum())
                EI -= 0.1*np.log(cdf(-py/ps))
            print 'loss x',x
            if EI < self.loss:
                self.loss = EI
                self.tmp_py = tmp_py.copy()
                self.x = np.copy(x)
            return EI
        
        '''
        def loss(x):
            x = x.reshape(self.dim, x.size/self.dim)
            tmp_py, ps2 = self.main_function.predict(x)
            py = tmp_py.sum()
            ps = np.sqrt(ps2.sum())
            k = self.best_y.sum()
            tmp = (k - py)/ps
            EI = -(k - py)*cdf(tmp) - ps*pdf(tmp)
            for i in range(len(self.constr_list)):
                py, ps2 = self.constr_list[i].predict(x[i:i+1])
                py = py.sum()
                ps = np.sqrt(ps2.sum())
                EI = EI*cdf(-py/ps)
            if EI < self.loss:
                self.loss = EI
                self.best_y = tmp_py
            return EI
        '''
        gloss = grad(loss)

        try:
            fmin_l_bfgs_b(loss, x0, gloss, maxiter=200, m=100, iprint=1)
        except np.linalg.LinAlgError:
            print('Increase noise term and re-optimization')
            x0 = np.copy(self.x)
            x0[0] += 0.01
            try:
                fmin_l_bfgs_b(loss, x0, gloss, maxiter=200, m=10, iprint=1)
            except:
                print('Exception caught, L-BFGS early stopping...')
                print(traceback.format_exc())
        except:
            print('Exception caught, L-BFGS early stopping...')
            print(traceback.format_exc())
                
        print('Optimized loss is %g' % self.loss)
        if(np.isinf(self.loss) or np.isnan(self.loss)):
            print('Fail to build GP model')
            sys.exit(1)

        print self.x
        print self.loss
        print self.best_y


