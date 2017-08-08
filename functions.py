import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm


def rosenbrock(x):
    f = 100*(x[1] - x[0]**2)**2 + (1 - x[0])**2
    return f

def rosenbrock_jac(x):
    fx = 400*x[0]**3 - 400*x[0]*x[1] + 2*x[0] - 2
    fy = 200*(x[1] - x[0]**2)
    return np.array([fx, fy])

def quadratic(x):
    f = x[0]**2 + x[1]**2
    return f

def quadratic_jac(x):
    fx = 2*x[0]
    fy = 2*x[1]
    return np.array([fx, fy])

def beale(x):
    f = (1.5-x[0]+x[0]*x[1])**2 + (2.25-x[0]+x[0]*x[1]**2)**2 + (2.625-x[0]+x[0]*x[1]**3)**2
    return f

def beale_jac(x):
    fx = 2*(x[1]-1)*(1.5-x[0]+x[0]*x[1]) + 2*(x[1]**2-1)*(2.25-x[0]+x[0]*x[1]**2) + 2*(x[1]**3-1)*(2.625-x[0]+x[0]*x[1]**3)
    fy = 2*x[0]*(1.5-x[0]+x[0]*x[1]) + 4*x[0]*x[1]*(2.25-x[0]+x[0]*x[1]**2) + 6*x[0]*(x[1]**2)*(2.625-x[0]+x[0]*x[1]**3)**2
    return np.array([fx, fy])

def saddle(x):
    f = x[0]**2 - x[1]**2 + 25
    return f

def saddle_jac(x):
    fx = 2*x[0]
    fy = -2*x[1]
    return np.array([fx, fy])

def fun_mesh(fun, xlim=[-4.5,4.5], ylim=[-4.5,4.5], step=0.05):
    x = np.arange(xlim[0], xlim[1], step)
    y = np.arange(ylim[0], ylim[1], step)
    X = np.asarray(np.meshgrid(x, y))
    Z = fun(X)
    return X, Z

if __name__ == "__main__":
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d import Axes3D
    from matplotlib import cm

    fun = saddle

    X, Z = fun_mesh(fun, xlim=[-4.5,4.5], ylim=[-4.5,4.5], step=0.05)

    coords = np.unravel_index(Z.argmin(), Z.shape)
    print('Minimum at: ', (X[0,coords[0],coords[1]], X[1,coords[0],coords[1]]))
    print('Min/max Z: ', Z.min(), Z.max())

    levels = np.logspace(0, np.log10(Z.max()), 20)

    plt.figure()
    plt.contour(X[0], X[1], Z, levels)
    plt.show()

    # Plot the surface.
    fig = plt.figure()
    ax = fig.gca(projection='3d')
    surf = ax.plot_surface(X[0], X[1], Z, cmap=cm.jet, linewidth=0, antialiased=False)
    ax.view_init(azim=-135, elev=35)
    plt.show()
