import autograd.numpy as np
from autograd import grad
import traceback
import sys
import pickle
from scipy.optimize import fmin_l_bfgs_b
from .activations import *
from .Bagging_GP_model import Bagging_GP_model
import random

class Bagging_Constr_model:
    def __init__(self, num_models, main_f, dataset, dim, outdim, bounds, scale, num_layers, layer_size, act, max_iter, l1, l2, debug=True):
        self.num_models = num_models
        self.dim = dim
        self.outdim = outdim
        self.main_f = main_f
        self.bounds = np.copy(bounds)
        self.train_x = dataset['train_x'].copy()
        self.train_y = dataset['train_y'].copy()
        self.best_y = np.inf
        self.best_constr = np.inf

        self.model = []
        for i in range(self.outdim):
            layer_sizes = [layer_size[i]]*num_layers[i]
            activations = [get_act_f(act[i])]*num_layers[i]
            m = Bagging_GP_model(num_models, self.train_x, self.train_y[i], layer_sizes, activations, bfgs_iter=max_iter[i], l1=l1[i], l2=l2[i], debug=True)
            m.fit(scale=scale[i])
            self.model.append(m)

    def construct_model(self,idx):
        layer_sizes = [self.layer_size[idx]]*self.num_layers[idx]
        activations = [get_act_f(self.act[idx])]*self.num_layers[idx]
        model = Bagging_GP_model(self.num_models, self.train_x, self.train_y[idx], layer_sizes, activations, bfgs_iter=self.max_iter[idx], l1=self.l1[idx], l2=self.l2[idx], debug=True)
        model.fit(scale=self.scale[idx])
        return model

    def rand_x(self):
        x = np.zeros((self.dim,1))
        for i in range(self.dim):
            x[i] = random.uniform(self.bounds[i,0], self.bounds[i,1])
        return x

    def predict(self, x):
        pys = np.zeros((x.shape[1], self.outdim))
        ps2s = np.zeros((x.shape[1], self.outdim))
        for i in range(self.outdim):
            py, ps2 = self.model[i].predict(x)
            pys[:,i] = py
            ps2s[:,i] = np.diagonal(ps2)
        return pys, ps2s

    def get_best_y(self, x, y):
        for i in range(y.shape[1]):
            constr = np.maximum(y[1:,i],0).sum()
            if self.best_constr > 0 and constr < self.best_constr:
                self.best_constr = constr
                self.best_y = y[0,i]
                self.best_x = x[:,i]
            elif self.best_constr <= 0 and constr <= 0 and y[0,i] < self.best_y:
                self.best_constr = constr
                self.best_y = y[0,i]
                self.best_x = x[:,i]

    def fit(self, x):
        x0 = np.copy(x)
        self.x = np.copy(x)
        self.loss = np.inf
        self.get_best_y(self.train_x, self.train_y)

        def loss(x):
            x = x.reshape(self.dim, int(x.size/self.dim))
            EI = 1.0
            if self.best_constr <= 0:
                py, ps2 = self.model[0].predict(x)
                py = py.sum()
                ps = np.sqrt(ps2.sum())
                tmp = (self.best_y - py)/ps
                EI = ps*(tmp*cdf(tmp)+pdf(tmp))
                # print('py',py,'ps',ps,'best_y',self.best_y,'EI',EI)
            PI = 1.0
            for i in range(1,self.outdim):
                py, ps2 = self.model[i].predict(x)
                py = py.sum()
                ps = np.sqrt(ps2.sum())
                PI = PI*cdf(-py/ps)
            loss = -EI*PI
            if loss < self.loss:
                self.loss = loss
                self.x = np.copy(x)
            return loss

        gloss = grad(loss)

        try:
            fmin_l_bfgs_b(loss, x0, gloss, bounds=self.bounds, maxiter=200, m=100, iprint=0)
        except np.linalg.LinAlgError:
            print('Increase noise term and re-optimization')
            x0 = np.copy(self.x)
            x0[0] += 0.01
            try:
                fmin_l_bfgs_b(loss, x0, gloss, bounds=self.bounds, maxiter=200, m=10, iprint=0)
            except:
                print('Exception caught, L-BFGS early stopping...')
                print(traceback.format_exc())
        except:
            print('Exception caught, L-BFGS early stopping...')
            print(traceback.format_exc())

        # print('Optimized loss is %g' % self.loss)
        if(np.isnan(self.loss) or np.isinf(self.loss)):
            print('Fail to build GP model')
            sys.exit(1)
        '''
        print('best_y',self.best_y)
        print('predict',self.model[0].predict(self.x),'loss',self.loss)
        print('x',self.x.T)
        print('true',self.main_f(self.x))
        '''
        return self.x




















