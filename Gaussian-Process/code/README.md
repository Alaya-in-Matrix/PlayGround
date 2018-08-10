# Gaussian Process

### enb

* It turned out that scale=0.4 has the smallest mean and std after cross-validation

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
* Considering the fact that scipy.special doesn't work for autograd, we decide implement erf function in autograd.numpy. The code comes from website: [erf function code in numpy](www.johndcook.com/blog/python_erf/).

### debug

* remember to use `import autograd.numpy as np` instead of `import numpy as np` for furthur grad computation 


