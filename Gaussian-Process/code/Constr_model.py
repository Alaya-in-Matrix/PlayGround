import autograd.numpy as np
from autograd import grad
import traceback
from scipy.optimize import fmin_l_bfgs_b
from activations import * 
import sys
import cPickle as pickle
from NN import NN
from GP_model import GP_model


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

    def fit(self, x, bounds):
        x0 = np.copy(x)
        self.x = np.copy(x)
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
            if py < self.tmp_py.sum():
                self.tmp_py = tmp_py.copy()
            ps = np.sqrt(ps2.sum())
            tmp = (self.best_y - py)/ps
            EI = (self.best_y - py)*cdf(tmp) + ps*pdf(tmp)
            EI = -100*np.log(EI+0.000001)
            for i in range(len(self.constr_list)):
                py, ps2 = self.constr_list[i].predict(x)
                py = py.sum()
                ps = np.sqrt(ps2.sum())
                EI -= np.log(cdf(-py/ps)+0.000001)
                # EI = EI * cdf(-py/ps)
            if EI < self.loss:
                self.loss = EI
                # self.tmp_py = tmp_py.copy()
                self.x = np.copy(x)
            return EI
        
        gloss = grad(loss)

        try:
            fmin_l_bfgs_b(loss, x0, gloss, bounds=bounds, maxiter=200, m=100, iprint=1)
        except np.linalg.LinAlgError:
            print('Increase noise term and re-optimization')
            x0 = np.copy(self.x)
            x0[0] += 0.01
            try:
                fmin_l_bfgs_b(loss, x0, gloss, bounds=bounds, maxiter=200, m=10, iprint=1)
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


