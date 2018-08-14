# Gaussian Process

### enb

* It turned out that scale=0.4 has the smallest mean and std after cross-validation

### branin

* The test bench should be uniformly distributed. `np.random.uniform`
* layer_size = 30, num_layers = 3

### cdf

* In order to furthur calculate the integration of **Probability Distribution Function(PDF)**, which is the **Cumulative Distribution Function(CDF)**, we import sympy for experiment.

```bash
>>> from sympy import *
>>> x = symbols('x')
>>> print(integrate(exp(-x**2 / 2)/sqrt(2*pi), (x, -1, 1)))
erf(sqrt(2)/2)
```
* Thus, we can get cdf(x) = 0.5 + erf(x/sqrt(2))/2.

```python
import numpy as np
from scipy.special import erf

def cdf(x):
    return 0.5 + erf(x/sqrt(2))/2

def pdf(x):
    return np.exp(-x**2 / 2)/np.sqrt(2 * np.pi) 
```
* Considering the fact that scipy.special doesn't work for autograd, we decide implement erf function in autograd.numpy. The code comes from website: <a href="www.johndcook.com/blog/python_erf/" target="_blank">erf function in numpy</a>

### debug

* remember to use `import autograd.numpy as np` instead of `import numpy as np` for furthur grad computation 
* there is no way to get `grad(np.diagonal)`. Considering the fact that ps2.shape=(self.num_test, self.num_test), we just reduce np.diagonal function. And we can get the same ps2 as before.
* autograd only works for function that has one parameter. Thus, we need to change `cdf(x, mu, theta)` and `pdf(x, mu, theta)` into `cdf(x)` and `pdf(x)`. Compute `x=(x-mu)/theta` before feed it to `cdf` or `pdf` function.
* autograd only works for function whose parameter.ndim is 1
* use `np.sqrt(np.sum())` instead of `np.sqrt().sum()` to greatly reduce the computation complexity
