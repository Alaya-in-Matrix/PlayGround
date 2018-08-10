import autograd.numpy as np



def tanh(x):
    return np.tanh(x)

def relu(x):
    return np.maximum(0,x)

def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))

def pdf(x, mu, theta):
    x = (x-mu)/theta
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

def cdf(x, mu, theta):
    x = (x-mu)/theta
    return 0.5 + erf(np.sqrt(2)/2 * x)/2
