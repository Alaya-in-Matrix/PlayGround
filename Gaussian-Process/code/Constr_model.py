import autograd.numpy as np
from autograd import grad
import traceback
from scipy.optimize import fmin_l_bfgs_b
from activations import *
from GP_model import GP_model
import sys

class Constr_model:
    def __init__(self, x, main_function, constr_list):
        self.dim = x.shape[0]
        self.main_function = main_function
        self.constr_list = constr_list
        self.best_y = 1000
        self.loss = 10000
        self.best_x = x.copy()
    
    def loss(self, x):
        x = x.reshape(self.dim, x.size/self.dim)
        py, ps2 = self.main_function.predict(x)
        tmp_py = py
        py = py.sum()
        ps = np.sqrt(ps2).sum()
        EI = (self.best_y - py)*cdf(self.best_y, py, ps) + ps*pdf(self.best_y, py, ps)
        for i in range(len(self.constr_list)):
            py, ps2 = self.constr_list[i].predict(x[i:i+1])
            py = py.sum()
            ps = np.sqrt(ps2.sum())
            EI = EI * cdf(0, py, ps)
        if -EI < self.loss:
            self.best_loss = -EI
            self.best_x = x
            self.best_y = tmp_py
        return -EI

    def optimize(self):
        x0 = np.copy(self.best_x)

        def loss(x):
            nlz = self.loss(x)
            print nlz
            return nlz

        gloss = grad(loss)

        try:
            fmin_l_bfgs_b(loss, x0, gloss, maxiter=200, m=100, iprint=1)
        except np.linalg.LinAlgError:
            print('Increase noise term and re-optimization')
            x0 = np.copy(self.best_x)
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

        print self.best_x
        print self.best_y
        print self.loss







