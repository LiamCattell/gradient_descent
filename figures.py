import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
from functions import fun_mesh

def contour_animation(fun, xall, labels, fname=None, nframes=200,
                      xlim=[-4.5,4.5], ylim=[-4.5,4.5], step=0.05):

    # Get mesh for plotting function contour
    X, Z = fun_mesh(fun, xlim, ylim, step)

    # Initialise figure
    fig = plt.figure(facecolor='w', figsize=(6,4))

    # Plot contour
    low = 1e-3
    if Z.min() > low:
        low = np.log10(Z.min())
    levels = np.logspace(low, np.log10(Z.max()), 25)
    plt.contour(X[0], X[1], Z, levels)

    # Initialise lines
    lines = [None]*len(labels)
    for i, lab in enumerate(labels):
        lines[i], = plt.plot([], [], ls='-', lw=2, marker='o', markersize=7, label=lab)

    # Plot star indicating the location of the minimum
    xmin = np.unravel_index(Z.argmin(), Z.shape)
    plt.plot(X[0,xmin[0],xmin[1]], X[1,xmin[0],xmin[1]], c='y', marker='*', markersize=20)

    # Format axes/figure
    plt.legend(loc=2)
    plt.tick_params(which='both', left=0, right=0, bottom=0, top=0, labelleft=0, labelbottom=0)
    fig.tight_layout()

    # Function to update the lines
    def update_lines(t):
        framestep = xall[0].shape[0] // nframes
        for i, line in enumerate(lines):
            line.set_data(xall[i][:t*framestep,0], xall[i][:t*framestep,1])
            line.set_markevery([t*framestep-1])
        return tuple(lines)

    # Create animation
    anim = animation.FuncAnimation(fig, update_lines, frames=nframes, interval=30, blit=True)

    # Save animated gif if required
    if fname is not None:
        print('Saving {}...'.format(fname))
        # anim.save(fname, dpi=40, writer='imagemagick')
        anim.save(fname, dpi=100, bitrate=1000000, writer='ffmpeg')

    plt.show()
    return


def surface_animation(fun, xall, feval, labels, fname=None, nframes=200,
                      xlim=[-4.5,4.5], ylim=[-4.5,4.5], step=0.05):
    # Get mesh for plotting function contour
    X, Z = fun_mesh(fun, xlim, ylim, step)

    # Initialise figure
    fig = plt.figure(facecolor='w')
    ax = fig.gca(projection='3d')

    # Plot the surface
    surf = ax.plot_surface(X[0], X[1], Z, cmap=cm.jet, lw=0, antialiased=True)

    # Initialise lines
    lines = [None]*len(labels)
    for i, lab in enumerate(labels):
        lines[i], = plt.plot([], [], [], ls='-', lw=2, marker='o', markersize=7, label=lab)

    # Plot star indicating the location of the minimum
    xmin = np.unravel_index(Z.argmin(), Z.shape)
    plt.plot([X[0,xmin[0],xmin[1]]], [X[1,xmin[0],xmin[1]]], [Z.min()], c='y', marker='*', markersize=20)

    # Format axes/figure
    plt.legend(loc=2)
    ax.set_xticklabels([])
    ax.set_yticklabels([])
    ax.set_zticklabels([])
    ax.view_init(azim=-125, elev=50)
    fig.tight_layout()

    # Function to update the lines
    def update_lines(t):
        framestep = xall[0].shape[0] // nframes
        for i, line in enumerate(lines):
            line.set_data(xall[i][:t*framestep,0], xall[i][:t*framestep,1])
            line.set_3d_properties(feval[i][:t*framestep])
            line.set_markevery([t*framestep-1])
        return tuple(lines)

    # Create animation
    anim = animation.FuncAnimation(fig, update_lines, frames=nframes, interval=30, blit=True)

    # Save animated gif if required
    if fname is not None:
        print('Saving {}...'.format(fname))
        # anim.save(fname, dpi=70, writer='imagemagick')
        anim.save(fname, dpi=100, bitrate=1000000, writer='ffmpeg')

    plt.show()
    return


def contour(fun, fname=None, xlim=[-4.5,4.5], ylim=[-4.5,4.5], step=0.05):

    # Get mesh for plotting function contour
    X, Z = fun_mesh(fun, xlim, ylim, step)

    # Initialise figure
    fig = plt.figure(facecolor='w')

    # Plot contour
    low = 1e-3
    if Z.min() > low:
        low = np.log10(Z.min())
    levels = np.logspace(low, np.log10(Z.max()), 25)
    plt.contour(X[0], X[1], Z, levels)

    # Plot star indicating the location of the minimum
    xmin = np.unravel_index(Z.argmin(), Z.shape)
    plt.plot(X[0,xmin[0],xmin[1]], X[1,xmin[0],xmin[1]], c='y', marker='*', markersize=20)

    # Format axes/figure
    plt.tick_params(which='both', left=0, right=0, bottom=0, top=0, labelleft=0, labelbottom=0)
    fig.tight_layout()

    # Save figure if required
    if fname is not None:
        plt.savefig(fname, bbox_inches='tight')

    plt.show()
    return


def surface(fun, fname=None, xlim=[-4.5,4.5], ylim=[-4.5,4.5], step=0.05):
    # Get mesh for plotting function contour
    X, Z = fun_mesh(fun, xlim, ylim, step)

    # Initialise figure
    fig = plt.figure(facecolor='w')
    ax = fig.gca(projection='3d')

    # Plot the surface
    surf = ax.plot_surface(X[0], X[1], Z, cmap=cm.jet, lw=0, antialiased=True)

    # Plot star indicating the location of the minimum
    xmin = np.unravel_index(Z.argmin(), Z.shape)
    plt.plot([X[0,xmin[0],xmin[1]]], [X[1,xmin[0],xmin[1]]], [Z.min()], c='y', marker='*', markersize=20)

    # Format axes/figure
    ax.set_xticklabels([])
    ax.set_yticklabels([])
    ax.set_zticklabels([])
    ax.view_init(azim=-125, elev=50)
    fig.tight_layout()

    # Save figure if required
    if fname is not None:
        plt.savefig(fname, bbox_inches='tight')

    plt.show()
    return
