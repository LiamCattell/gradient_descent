import numpy as np
from functions import rosenbrock, rosenbrock_jac, fun_mesh
from optimizers import GD, Adagrad, Adadelta, RMSprop, Adam
from figures import contour, surface, contour_animation

# The objective function and its gradient
fun = rosenbrock
jac = rosenbrock_jac

# Initial x
x0 = np.array([-2., -1.])

# Some global settings
max_iter = 5000
tol = 1e-8

# Optimization methods
gd = GD(fun, jac, lr=0.0005, max_iter=max_iter, tol=tol)
mom1 = GD(fun, jac, lr=0.0005, momentum=0.5, max_iter=max_iter, tol=tol)
mom2 = GD(fun, jac, lr=0.0005, momentum=0.9, max_iter=max_iter, tol=tol)
nest = GD(fun, jac, lr=0.0005, momentum=0.5, nesterov=True, max_iter=max_iter, tol=tol)
agrad = Adagrad(fun, jac, lr=0.1, max_iter=max_iter, tol=tol)
adelta = Adadelta(fun, jac, lr=1., max_iter=max_iter, tol=tol)
rms = RMSprop(fun, jac, lr=0.001, max_iter=max_iter, tol=tol)
adam = Adam(fun, jac, lr=0.01, max_iter=max_iter, tol=tol)

optimizers = [gd, mom1, nest, agrad, adelta, rms, adam]
labels = ['GD', 'Momentum', 'Nesterov', 'Adagrad', 'Adadelta', 'RMSprop', 'Adam']

# Initialise lists for x-values at each iteration, and final x-value for each
# optimisation method
xall = []
xfinal = []
feval = []

# Loop over all optimizers
for opt in optimizers:
    # Minimise the function
    opt.optimize(x0)

    # Get the values of x at all iterations, and pad the list so that it's the
    # same length as max_iter
    xi = opt.xall
    xi += [xi[-1]]*(max_iter-len(xi))
    xall.append(np.array(opt.xall))

    # Get the value of f(x) at each iteration
    fi = opt.feval
    fi += [fi[-1]]*(max_iter-len(fi))
    feval.append(fi)

    # Final value of x
    xfinal.append(opt.x)
    print('FINAL X: ', opt.x)

# Plot the contour, surface, and animation
contour(fun, fname='rosenbrock_cont.png', xlim=[-4.5,4.5], ylim=[-4.5,4.5], step=0.01)
surface(fun, fname='rosenbrock_surf.png', xlim=[-4.5,4.5], ylim=[-4.5,4.5], step=0.01)
contour_animation(fun, xall, labels, fname='rosenbrock_all.mp4', xlim=[-2.5,2.5], ylim=[-1.5,1.5])
