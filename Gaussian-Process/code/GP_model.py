import autograd.numpy as np
from NN import NN
from scipy.optimize import fmin_l_bfgs_b
from autograd import grad

def scale_x(log_lscale, x):
    lscale = np.exp(log_lscale).repeat(x.shape[1], axis=0).reshape(x.shape)
    return x / lscale

def chol_inv(L, y):
    '''
    K = L * L.T
    return inv(K) * y
    '''
    v = np.linalg.solve(L, y)
    return np.linalg.solve(L.T, v)

class GP_model:
    def __init__(self, train_x, train_y, layer_sizes, activations, bfgs_iter=500, l1=0, l2=0, debug=False):
        self.train_x = np.copy(train_x)
        self.train_y = np.copy(train_y)
        self.num_layers = np.copy(len(layer_sizes))
        self.dim = train_x.shape[0]
        self.num_train = train_x.shape[1]
        self.mean = np.mean(self.train_y)
        self.layer_sizes = np.copy(layer_sizes)
        self.activations = activations
        self.l1 = l1
        self.l2 = l2
        self.debug = debug
        self.m = layer_sizes[-1]
        self.nn = NN(layer_sizes, activations)
        '''
        theta_n^2: noise for y
        theta_p^2: noise for w
        scale: self.dim (preprocess input x)
        NN parameter number
        '''
        self.num_param = 2 + self.dim + self.nn.num_param(self.dim) 
        self.train_y.reshape(1, train_y.size)
        self.train_y_zero = self.train_y - self.mean
        self.loss = np.inf
        self.bfgs_iter = bfgs_iter

    def rand_theta(self, scale=0.1):
        '''
        generate an initial theta, the weights of NN are randomly initialized
        '''
        theta = scale * np.random.randn(self.num_param)
        theta[0] = np.log(np.std(self.train_y_zero) / 2)
        theta[1] = np.log(np.std(self.train_y_zero))
        for i in range(self.dim):
            theta[2+i] = np.maximum(-100, np.log(0.5*(self.train_x[i].max() - self.train_x[i].min())))
        return theta

    def calc_Phi(self, w, x):
        '''
        Phi.shape: self.m, self.num_train
        '''
        return self.nn.predict(w, x)

    def log_likelihood(self, theta):
        log_sn = theta[0]
        log_sp = theta[1]
        log_lscale = theta[2:2+self.dim]
        w = theta[2+self.dim:]
        sn2 = np.exp(2 * log_sn)
        sp2 = np.exp(2 * log_sp)

        scaled_x = scale_x(log_lscale, self.train_x)
        Phi = self.calc_Phi(w, scaled_x) # Phi.shape: (self.m, self.num_train)
        Phi_y = np.dot(Phi, self.train_y_zero.T)
        A = np.dot(Phi, Phi.T) + (self.m * sn2 / sp2) * np.eye(self.m) # A.shape: (self.m, self.m)
        LA = np.linalg.cholesky(A)

        logDetA = 0
        for i in range(self.m):
            logDetA = 2 * np.log(LA[i][i])

        data_fit = (np.dot(self.train_y_zero, self.train_y_zero.T) - np.dot(Phi_y.T, chol_inv(LA, Phi_y))) / sn2
        neg_likelihood = 0.5 * (data_fit + logDetA + self.num_train * np.log(2 * np.pi * sn2) - self.m * np.log(self.m * sn2 / sp2))
        if(np.isnan(neg_likelihood)):
            neg_likelihood = np.inf

        return neg_likelihood

    def fit(self, theta):
        theta0 = np.copy(theta)
        self.loss = np.inf
        
        def loss(theta_tmp):
            nlz = self.log_likelihood(theta_tmp)
            w_nobias = self.nn.w_nobias(w, self.dim)
            l1_reg = self.l1 * np.abs(w_nobias).sum()
            l2_reg = self.l2 * np.dot(w_nobias, w_nobias.T)
            nlz += l1_reg + l2_reg
            if nlz < self.loss:
                self.loss = nlz
                self.theta = np.copy(theta_tmp)
            return nlz
        print 'get loss function'
        gloss = grad(loss)

        try:
            fmin_l_bfgs_b(loss, theta0, gloss, maxiter = self.bfgs_iter, m = 100, iprint=1)
        except np.linalg.LinAlgError:
            print("Increase noise term and re-optimization")
            theta0     = np.copy(self.theta);
            theta0[0] += np.log(10);
            try:
                fmin_l_bfgs_b(loss, theta0, gloss, maxiter = self.bfgs_iter, m = 10, iprint=1)
            except:
                print("Exception caught, L-BFGS early stopping...")
                if self.debug:
                    print(traceback.format_exc())
        except:
            print("Exception caught, L-BFGS early stopping...")
            if self.debug:
                print(traceback.format_exc())

        print("Optimized loss is %g" % self.loss)
        if(np.isinf(self.loss) or np.isnan(self.loss)):
            print("Fail to build GP model")

        log_sn = self.theta[0]
        log_sp = self.theta[1]
        log_lscale = self.theta[2:2+self.dim]
        w = self.theta[2+self.dim:]
        sn2 = np.exp(2*log_sn)
        sp2 = np.exp(2*log_sp)
        Phi = self.calc_Phi(w, scale_x(log_lscale, self.train_x))
        self.A = np.dot(Phi, Phi.T) + (self.m * sn2 / sp2) * np.eye(self.m) # A.shape: (self.m, self.m)
        self.LA = np.linalg.cholesky(self.A)
        self.alpha = chol_inv(self.LA, np.dot(Phi, self.train_y_zero.T))

    def predict(self, test_x):
        log_sn = self.theta[0]
        log_sp = self.theta[1]
        log_lscale = self.theta[2:2+self.dim]
        w = self.theta[2+self.dim:]
        sn2 = np.exp(2 * log_sn)
        sp2 = np.exp(2 * log_sp)
        Phi_test = self.calc_Phi(w, scale_x(log_lscale, test_x))
        py = self.mean + np.dot(Phi_test.T, self.alpha)
        ps2 = sn2 + sn2 * np.diagonal(np.dot(Phi_test.T, chol_inv(self.LA, Phi_test)))
        return py, ps2

    def test(self):
        theta = self.rand_theta()
        
        def loss(theta_tmp):
            nlz = self.log_likelihood(theta_tmp)
            return nlz

        gloss = grad(loss)
        H = gloss(theta)
        print H[:20]










