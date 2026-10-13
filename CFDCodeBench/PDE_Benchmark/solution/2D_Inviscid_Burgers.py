"""
2D inviscid Burgers equation
---------------------------------------------------------------
- Exercise 6 - Two-dimensional inviscid Burgers equation
- First-order upwind
- REF: https://github.com/okcfdlab/engr491
---------------------------------------------------------------
Author: Weichao Li
Date: 2025/03/13
"""
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from mpl_toolkits.mplot3d import Axes3D


def initialize(nx, ny, dx, dy):
    # assign initial conditions
    # set hat function such that u(0.5<=x<=1 && 0.5<=y<=1) is 2
    u = np.ones((ny, nx))
    v = np.ones((ny, nx))
    u[int(.5 / dy):int(1 / dy + 1), int(.5 / dx):int(1 / dx + 1)] = 2
    # set hat function such that v(0.5<=x<=1 && 0.5<=y<=1) is 2
    v[int(.5 / dy):int(1 / dy + 1), int(.5 / dx):int(1 / dx + 1)] = 2
    return u, v


def plot2d(x, y, u, figtitle):
    # This function yields a pretty 3D plot of a 2D field. It takes three
    # arguments: the x array of dimension (1,nx), y arrary of dim (1,ny), and
    # solution array of dimension (ny,nx)

    fig = plt.figure(figsize=(11, 7), dpi=100)
    ax = fig.add_subplot(111, projection='3d')
    X, Y = np.meshgrid(x, y)
    ax.plot_surface(X, Y, u, cmap=cm.viridis, rstride=2, cstride=2)
    ax.set_xlabel('$x$')
    ax.set_ylabel('$y$')
    ax.set_title(figtitle)


def upwind(u, v, dt, dx, dy, nt):
    # This function solves 2D inviscid Burgers equation in 2D using
    # first-order upwind method. It takes 5 arguments: the initial u and
    # v arrays, time step size, spatial step size in x, spatial step size
    # in y, and the number of time steps to integrate.

    # loop across time steps
    fig = plt.figure(figsize=(11, 7), dpi=100)
    plt.title('Upwind')
    for n in range(0, nt + 1):
        un = u.copy()  # un and vn are the u and v fields at time n
        vn = v.copy()
        u[1:, 1:] = un[1:, 1:] \
                    - (un[1:, 1:] * dt / dx * (un[1:, 1:] - un[1:, :-1])) \
                    - (vn[1:, 1:] * dt / dy * (un[1:, 1:] - un[:-1, 1:]))
        v[1:, 1:] = vn[1:, 1:] \
                    - (un[1:, 1:] * dt / dx * (vn[1:, 1:] - vn[1:, :-1])) \
                    - (vn[1:, 1:] * dt / dy * (vn[1:, 1:] - vn[:-1, 1:]))
        u[0, :] = 1
        u[-1, :] = 1
        u[:, 0] = 1
        u[:, -1] = 1
        v[0, :] = 1
        v[-1, :] = 1
        v[:, 0] = 1
        v[:, -1] = 1

        if (n % 20 == 0):
            string = 'n =' + str(n)
            plt.plot(x, u[70, :], label=string)
            plt.legend()

    return u, v


# define parameters
Lx = 2.
Ly = 2.
nx = 151
ny = 151
nt = 300
dx = Lx / (nx - 1)
dy = Ly / (ny - 1)
sigma = 0.2
dt = sigma * min(dx, dy) / 2

print('dx = ', dx, ', dy = ', dy, ', dt = ', dt, 's')

# define mesh arrays
x = np.linspace(0., Lx, nx)
y = np.linspace(0., Ly, ny)

# define initial solution arrays
u, v = initialize(nx, ny, dx, dy)
plot2d(x, y, u, 'Initial values')

# Solve using upwind method
u1, v1 = upwind(u, v, dt, dx, dy, nt)
plot2d(x, y, u1, 'Upwind')

plt.show()

##########################################################################
import os

# === Paths ===
ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))  # PDE_Benchmark root
OUTPUT_FOLDER = os.path.join(ROOT_DIR, "results", "solution")

# Ensure the output directory exists
os.makedirs(OUTPUT_FOLDER, exist_ok=True)
# Get the current Python file name without extension
python_filename = os.path.splitext(os.path.basename(__file__))[0]

# Define the file name dynamically
output_file_u = os.path.join(OUTPUT_FOLDER, f"u_{python_filename}.npy")
output_file_v = os.path.join(OUTPUT_FOLDER, f"v_{python_filename}.npy")

# Save the array u in the results folder
np.save(output_file_u, u1)
np.save(output_file_v, v1)

