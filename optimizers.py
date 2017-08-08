import numpy as np

class BaseOptimizer(object):
    """
    Optimizer base class.

    Liam Cattell -- July 2017
    """
    def __init__(self, fun, jac, **kwargs):
        self._fun = fun
        self._jac = jac

        # Set default decay rate, maximum iterations, stopping tolerance and
        # verbosity
        self._decay = kwargs.pop('decay', 0.)
        self._max_iter = kwargs.pop('max_iter', 200)
        self._tol = kwargs.pop('tol', 0.001)
        self._verbose = kwargs.pop('verbose', True)
        return

    def optimize(self, x0):
        self.xall = [x0]
        self.feval = [self._fun(x0)]

        x = np.copy(x0)

        # Iterate!
        for i in range(self._max_iter):
            self._iteration = i

            # Decay learning rate
            self._lr *= (1. / (1. + self._decay*i))

            # Update parameters
            update = self._get_update(x)
            x -= update

            # Evaluate function at new x-value
            feval = self._fun(x)

            # Stop iterating if function value is nan or inf
            if np.isnan(feval) or np.isinf(feval):
                break

            # Stop iterating if change in function value is negligible
            if np.abs(self.feval[i] - feval) < self._tol:
                break

            if self._verbose:
                print('Iteration: ', i, ' -- ', self._lr, ' -- ', feval)

            # Save this x-value
            self.xall.append(np.copy(x))
            self.feval.append(feval)

        self.x = x

        return

    def _get_update(self):
        raise NotImplementedError


class GD(BaseOptimizer):
    """
    Gradient descent optimizer.

    Includes support for momentum, learning rate decay, and Nesterov momentum.

    Liam Cattell -- July 2017

    Parameters
    ----------
    fun : callable
        Objective function
    jac : callable
        Jacobian (gradient) of objective function
    lr : float, default=0.01
        Learning rate
    momentum : float, default=0.
        Momentum
    nesterov : bool, default=False
        Whether to apply Nesterov momentum
    decay : float, default=0.
        Learning rate decay over each update
    max_iter : int, default=200
        Maximum iterations
    tol : float, default=0.001
        Stop iterating when change in function value is below this threshold
    verbose : int, default=True
        Print stuff during optimisation?
    """
    def __init__(self, fun, jac, lr=0.01, momentum=0., nesterov=False, **kwargs):
        super().__init__(fun, jac, **kwargs)
        self._lr = lr
        self._momentum = momentum
        self._nesterov = nesterov

        self._update_prev = 0.
        return

    def _get_update(self, x):
        x_tmp = x * 1.

        if self._nesterov:
            # "Look ahead"
            x_tmp = x - self._momentum*self._update_prev

        grad = self._jac(x_tmp)

        # Parameter update
        update = self._momentum*self._update_prev + self._lr*grad

        # Set previous update
        self._update_prev = np.copy(update)

        return update


class Adagrad(BaseOptimizer):
    """
    Adagrad optimizer.

    Liam Cattell -- July 2017

    Parameters
    ----------
    fun : callable
        Objective function
    jac : callable
        Jacobian (gradient) of objective function
    lr : float, default=0.01
        Learning rate
    epsilon : float, default=1e-8
        Fudge factor to stop divide-by-zero errors
    decay : float, default=0.
        Learning rate decay over each update
    max_iter : int, default=200
        Maximum iterations
    tol : float, default=0.001
        Stop iterating when change in function value is below this threshold
    verbose : int, default=True
        Print stuff during optimisation?

    See also
    --------
    [Adaptive Subgradient Methods for Online Learning and Stochastic
    Optimization] (http://www.jmlr.org/papers/volume12/duchi11a/duchi11a.pdf)
    """
    def __init__(self, fun, jac, lr=0.01, epsilon=1e-8, **kwargs):
        super().__init__(fun, jac, **kwargs)
        self._lr = lr
        self._epsilon = epsilon

        # Initialise sum of squared gradients
        self._grad2 = 0
        return

    def _get_update(self, x):
        grad = self._jac(x)

        self._grad2 += grad**2

        update = self._lr * grad / np.sqrt(self._grad2 + self._epsilon)

        return update


class Adadelta(BaseOptimizer):
    """
    Adadelta optimizer.

    Liam Cattell -- July 2017

    Parameters
    ----------
    fun : callable
        Objective function
    jac : callable
        Jacobian (gradient) of objective function
    lr : float, default=1.0
        Learning rate
    rho : float, default=0.95
    epsilon : float, default=1e-8
        Fudge factor to stop divide-by-zero errors
    decay : float, default=0.
        Learning rate decay over each update
    max_iter : int, default=200
        Maximum iterations
    tol : float, default=0.001
        Stop iterating when change in function value is below this threshold
    verbose : int, default=True
        Print stuff during optimisation?

    See also
    --------
    [Adadelta - an adaptive learning rate method]
    (http://arxiv.org/abs/1212.5701)
    """
    def __init__(self, fun, jac, lr=1.0, rho=0.95, epsilon=1e-8, **kwargs):
        super().__init__(fun, jac, **kwargs)
        self._lr = lr
        self._rho = rho
        self._epsilon = epsilon

        # Initialise decaying averages
        self._gradavg = 0.
        self._updateavg = 0.
        return

    def _get_update(self, x):
        grad = self._jac(x)

        # Decaying average of squared gradients
        self._gradavg = self._rho * self._gradavg + (1. - self._rho) * grad**2

        # RMS
        rmsgrad = np.sqrt(self._gradavg + self._epsilon)
        rmsupdate = np.sqrt(self._updateavg + self._epsilon)

        # Compute update
        update = grad * rmsupdate / rmsgrad

        # Decaying average of squared updates
        self._updateavg = self._rho * self._updateavg + (1. - self._rho) * update**2

        # Just in case the user sets a learning rate != 1.0
        update *= self._lr

        return update


class RMSprop(BaseOptimizer):
    """
    RMSprop optimizer.

    Liam Cattell -- July 2017

    Parameters
    ----------
    fun : callable
        Objective function
    jac : callable
        Jacobian (gradient) of objective function
    lr : float, default=0.001
        Learning rate
    rho : float, default=0.9
        RMSprop decay rate
    epsilon : float, default=1e-8
        Fudge factor to stop divide-by-zero errors
    decay : float, default=0.
        Learning rate decay over each update
    max_iter : int, default=200
        Maximum iterations
    tol : float, default=0.001
        Stop iterating when change in function value is below this threshold
    verbose : int, default=True
        Print stuff during optimisation?

    See also
    --------
    [rmsprop: Divide the gradient by a running average of its recent magnitude]
    (http://www.cs.toronto.edu/~tijmen/csc321/slides/lecture_slides_lec6.pdf)
    """
    def __init__(self, fun, jac, lr=0.001, rho=0.9, epsilon=1e-8, **kwargs):
        super().__init__(fun, jac, **kwargs)
        self._lr = lr
        self._rho = rho
        self._epsilon = epsilon

        # Initialise decaying average of squared gradients
        self._gradavg = 0.
        return

    def _get_update(self, x):
        grad = self._jac(x)

        # Decaying average of squared gradients
        self._gradavg = self._rho * self._gradavg + (1. - self._rho) * grad**2

        # RMS
        rmsgrad = np.sqrt(self._gradavg + self._epsilon)

        # Compute update
        update = self._lr * grad / rmsgrad

        return update


class Adam(BaseOptimizer):
    """
    Adam optimizer.

    Default parameters follow those provided in the original paper.

    Liam Cattell -- July 2017

    Parameters
    ----------
    fun : callable
        Objective function
    jac : callable
        Jacobian (gradient) of objective function
    lr : float, default=0.001
        Learning rate
    beta1 : float, default=0.9
    beta2 : float, default=0.999
    epsilon : float, default=1e-8
        Fudge factor to stop divide-by-zero errors
    decay : float, default=0.
        Learning rate decay over each update
    max_iter : int, default=200
        Maximum iterations
    tol : float, default=0.001
        Stop iterating when change in function value is below this threshold
    verbose : int, default=True
        Print stuff during optimisation?

    See also
    --------
    [Adam - A Method for Stochastic Optimization]
    (http://arxiv.org/abs/1412.6980v8)
    """
    def __init__(self, fun, jac, lr=0.001, beta1=0.9, beta2=0.999, epsilon=1e-8,
                **kwargs):
        super().__init__(fun, jac, **kwargs)
        self._lr = lr
        self._beta1 = beta1
        self._beta2 = beta2
        self._epsilon = epsilon

        # Initialise 1st and 2nd moments
        self._m = 0.
        self._v = 0.
        return


    def _get_update(self, x):
        grad = self._jac(x)

        # Modify learning rate to include bias correction
        t = self._iteration + 1
        lrt = self._lr * np.sqrt(1. - self._beta2**t) / (1. - self._beta1)

        # 1st moment (mean)
        self._m = self._beta1 * self._m + (1. - self._beta1) * grad

        # 2nd moment (uncentred variance)
        self._v = self._beta2 * self._v + (1. - self._beta2) * grad**2

        update = lrt * self._m / (np.sqrt(self._v) + self._epsilon)

        return update


if __name__ == "__main__":
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d import Axes3D
    from matplotlib import cm
    from functions import rosenbrock, rosenbrock_jac, beale, beale_jac, quadratic, quadratic_jac, fun_mesh

    x0 = np.array([-2.0, -1.])

    fun = rosenbrock
    jac = rosenbrock_jac

    opt = GD(fun, jac, lr=0.0001, max_iter=10000, tol=1e-8)
    opt.optimize(x0)
    print('FINAL X: ', opt.x)

    X, Z = fun_mesh(fun)
    levels = np.logspace(0, np.log10(Z.max()), 30)

    plt.figure()
    plt.contour(X[0], X[1], Z, levels)

    xi = np.array(opt.xall)
    plt.plot(xi[:,0], xi[:,1], linestyle='-', linewidth=2)

    plt.show()
