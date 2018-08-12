import autograd.numpy as np
import traceback
import sys
from autograd import grad
from scipy.optimize import fmin_l_bfgs_b


def tanh(x):
    return np.tanh(x)

def relu(x):
    return np.maximum(0,x)

def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))

def pdf(x):
    # x = (x-mu)/theta
    return np.exp(-x**2 / 2)/np.sqrt(2*np.pi)


'''
from scipy.special import erf

def cdf(x, mu, theta):
    x = (x-mu)/theta
    return 0.5 + erf(np.sqrt(2)/2 * x)/2

# as erf from scipy.special won't work for autograd
# we decided to implement erf in autograd.numpy
'''
# the code reference: www.johndcook.com/blog/python_erf/

def erf(x):
    # constants
    a1 =  0.254829592
    a2 = -0.284496736
    a3 =  1.421413741
    a4 = -1.453152027
    a5 =  1.061405429
    p  =  0.3275911
                            
    # Save the sign of x
    sign = np.sign(x)
    x = np.abs(x)
                                                        
    # A&S formula 7.1.26
    t = 1.0/(1.0 + p*x)
    y = 1.0 - (((((a5*t + a4)*t) + a3)*t + a2)*t + a1)*t*np.exp(-x**2)
                                                                    
    return sign*y

def cdf(x):
    # x = (x-mu)/theta
    return 0.5 + erf(x/np.sqrt(2))/2

'''
x0 = [0.5, 1.0, 2.0]

def loss(x):
    nlz = cdf(x[0], x[1], x[2])
    print x, nlz
    return nlz

gloss = grad(loss)

try:
    fmin_l_bfgs_b(loss, x0, gloss, maxiter=200, m=100, iprint=1)
except np.linalg.LinAlgError:
    print('Increase noise term and re-optimization')
    x0 = 0.3
    try:
        fmin_l_bfgs_b(loss, x0, gloss, maxiter=200, m=10, iprint=1)
    except:
        print('Exception caught, L-BFGS early stopping...')
        print(traceback.format_exc())
except:
    print('Exception caught, L-BFGS early stopping...')
    print(traceback.format_exc())
                
'''


