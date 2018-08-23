import autograd.numpy as np
from autograd import grad
from scipy.optimize import fmin_l_bfgs_b
from .NN import NN
import traceback
import sys


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
        sp2 = sp2 + 0.000001
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
        self.theta = np.copy(theta)

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

    def optimize(self, x, bounds):
        x0 = np.copy(x)
        self.x = np.copy(x)
        self.loss = np.inf
        
        def loss(x):
            x = x.reshape(self.dim, x.size/self.dim)
            py, ps2 = self.predict(x)
            py = py.sum()
            if py < self.loss:
                self.loss = py
                self.x = x.copy()
            return py

        gloss = grad(loss)

        try:
            fmin_l_bfgs_b(loss, x0, gloss, bounds=bounds, maxiter=200, m = 100, iprint=1)
        except np.linalg.LinAlgError:
            print('Increase noise term and re-optimization')
            x0 = np.copy(self.x)
            x0[0] += 1.0
            try:
                fmin_l_bfgs_b(loss, x0, gloss, bounds=bounds, maxiter=200, m = 10, iprint=1)
            except:
                print('Exception caught, L-BFGS early stopping...')
                print(traceback.format.exc())
        except:
            print('Exception caught, L-BFGS early stopping...')
            print(traceback.format_exc())

        print('Optimized loss is %g' % self.loss)
        if(np.isinf(self.loss) or np.isnan(self.loss)):
            print('Fail to build GP model')
            sys.exit(1)

        print('best_x',self.x)
        print('predict',self.predict(self.x))
        print('loss',self.loss)



