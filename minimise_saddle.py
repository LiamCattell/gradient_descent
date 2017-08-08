import numpy as np
import matplotlib.pyplot as plt
from functions import saddle, saddle_jac
from optimizers import GD, Adagrad, Adadelta, RMSprop, Adam
from figures import contour_animation, surface_animation

# The objective function and its gradient
fun = saddle
jac = saddle_jac

# Initial x
x0 = np.array([1., -0.001])

# Some global settings for all of the optimizers that we're going to use
max_iter = 500
tol = 1e-8

# Optimisation methods
gd = GD(fun, jac, lr=0.001, max_iter=max_iter, tol=tol)
mom = GD(fun, jac, lr=0.001, momentum=0.99, max_iter=max_iter, tol=tol)
nest = GD(fun, jac, lr=0.001, momentum=0.99, nesterov=True, max_iter=max_iter, tol=tol)
agrad = Adagrad(fun, jac, lr=0.1, max_iter=max_iter, tol=tol)
adelta = Adadelta(fun, jac, lr=1., max_iter=max_iter, tol=tol)
rms = RMSprop(fun, jac, lr=0.01, max_iter=max_iter, tol=tol)
adam = Adam(fun, jac, lr=0.001, max_iter=max_iter, tol=tol)

optimizers = [gd, mom, nest, agrad, adelta, rms, adam]
labels = ['GD', 'Momentum', 'Nesterov', 'Adagrad', 'Adadelta', 'RMSprop', 'Adam']

# Initialise lists for x-values at each iteration, and final x-value for each
# optimisation method
xall = []
xfinal = []
feval = []

for opt in optimizers:
    # Minimise the function
    opt.optimize(x0)

    # Get the values of x at all iterations, and pad the list so that it's the
    # same length as max_iter
    xi = opt.xall
    xi += [xi[-1]]*(max_iter-len(xi))
    xall.append(np.array(opt.xall))

    fi = opt.feval
    fi += [fi[-1]]*(max_iter-len(fi))
    feval.append(fi)

    # Final value of x
    xfinal.append(opt.x)
    print('FINAL X: ', opt.x)

# Animate results
# contour_animation(fun, xall, labels, fname='saddle_all_cont.mp4', xlim=[-2.,2.], ylim=[-2.,2.])
surface_animation(fun, xall, feval, labels,  fname='saddle_all_surf.mp4', xlim=[-2.,2.], ylim=[-2.,2.], step=0.01)
