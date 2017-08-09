# Gradient descent

Examples of first order gradient descent optimization methods commonly used in convolutional neural networks.

Optimizers currently include:
1. Gradient descent
2. Gradient descent + momentum (part of the `GD` class)
3. Nesterov acclerated gradient (part of the `GD` class)
4. Adagrad
5. Adadelta
6. RMSprop
7. Adam
8. Adamax
9. Nadam


## Usage

All optimizers inherit from `BaseOptimizer` class. Therefore, they can all be used in a similar manner.

Since all of the optimizers are first-order gradient descent methods, you must specify an objective function `fun` to minimize,
as well as a function that computes the gradient `jac` of your objective function.

For example, imagine that we wish to minimize the following (very boring) function of two parameters:

```python
import numpy as np

def objective_fun(x):
    f = (x[0]-5.0)**2 + 3.0*x[1]**2
    return f

def objective_fun_grad(x):
    fx = 2.0*(x[0]-5.0)
    fy = 6.0*x[1]
    return np.array([fx, fy])
```

To minimize this function using Nesterov acclerated gradient descent:

```python
from optimizers import GD

# Starting point
x0 = np.array([1.7, 6.3])

# Initialize optimizer
opt = GD(objective_fun, objective_fun_grad, lr=0.0001, momentum=0.5, nesterov=True)

# Minimize the function
opt.optimize(x0)
```

In the example above, the final value of the parameters can be obtained using `opt.x`.
Similarly, the values of the parameters and objective function from all iterations can be obtained using `opt.xall` and `opt.feval`, respectively.

A full list of parameters for each optimization method can be found in `optimizers.py`.

See `minimise_rosenbrock.py` for examples on how to use the optimizers in practice.


## Credits

Liam Cattell
