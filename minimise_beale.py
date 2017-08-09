import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
from functions import beale, beale_jac, fun_mesh
from optimizers import GD, Adagrad, Adadelta, RMSprop, Adam, Adamax, Nadam
from figures import surface, contour, contour_animation

# The objective function and its gradient
fun = beale
jac = beale_jac

# Initial x
x0 = np.array([1.1, 1.6])

# Some global settings for all of the optimizers that we're going to use
max_iter = 5000
tol = 1e-8

# Optimisation methods
gd = GD(fun, jac, lr=0.0001, max_iter=max_iter, tol=tol)
mom = GD(fun, jac, lr=0.0001, momentum=0.95, max_iter=max_iter, tol=tol)
nest = GD(fun, jac, lr=0.0001, momentum=0.95, nesterov=True, max_iter=max_iter, tol=tol)
agrad = Adagrad(fun, jac, lr=0.1, max_iter=max_iter, tol=tol)
adelta = Adadelta(fun, jac, lr=1., max_iter=max_iter, tol=tol)
rms = RMSprop(fun, jac, lr=0.001, max_iter=max_iter, tol=tol)
adam = Adam(fun, jac, lr=0.001, max_iter=max_iter, tol=tol)
amax = Adamax(fun, jac, lr=0.001, max_iter=max_iter, tol=tol)
nad = Nadam(fun, jac, lr=0.01, max_iter=max_iter, tol=tol)

optimizers = [gd, mom, nest, agrad, adelta, rms, adam, amax, nad]
labels = ['GD', 'Momentum', 'Nesterov', 'Adagrad', 'Adadelta', 'RMSprop', 'Adam', 'Adamax', 'Nadam']

# Initialise lists for x-values at each iteration, and final x-value for each
# optimisation method
xall = []
xfinal = []

# Loop over all optimizers
for opt in optimizers:
    # Minimise the function
    opt.optimize(x0)

    # Get the values of x at all iterations, and pad the list so that it's the
    # same length as max_iter
    xi = opt.xall
    xi += [xi[-1]]*(max_iter-len(xi))
    xall.append(np.array(opt.xall))

    # Final value of x
    xfinal.append(opt.x)
    print('FINAL X: ', opt.x)


# Plot the contour, surface, and animation
contour(fun, xlim=[-4.5,4.5], ylim=[-4.5,4.5], step=0.01)
surface(fun, xlim=[-4.5,4.5], ylim=[-4.5,4.5], step=0.01)
contour_animation(fun, xall, labels, nframes=300, xlim=[-1.9,4.5], ylim=[-2.5,2.5])
