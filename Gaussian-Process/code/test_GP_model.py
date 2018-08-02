import numpy as np
from scipy.optimize import fmin_l_bfgs_b
from autograd import grad


'''
def chol_inv(L, y):
    v = np.linalg.solve(L, y)
    return np.linalg.solve(L.T, v)

m = 20
num_train = 100
Phi = np.random.randn(m, num_train)
A = np.dot(Phi, Phi.T) + 0.01 * np.eye(m)
LA = np.linalg.cholesky(A)

A_inv = chol_inv(LA, np.eye(m))

I = np.dot(A_inv, A)
print I
'''

def ll(x):
    y = x*x*x*x + x*x + 2*x + 1
    print 'x:',x,'y:',y
    return y


gll = grad(ll)

fmin_l_bfgs_b(ll, 0.0, gll, maxiter=1000, m=100, iprint=1)






