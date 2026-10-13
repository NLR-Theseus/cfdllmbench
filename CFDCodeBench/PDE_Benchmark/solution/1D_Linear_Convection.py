"""
1D linear convection equation
---------------------------------------------------------------
- Exercise 4 - Finite difference solution of the 1D linear convection equation
- Adams-Bashforth method
- REF: https://github.com/okcfdlab/engr491
---------------------------------------------------------------
Author: Weichao Li
Date: 2025/03/13
"""
import numpy as np
import matplotlib.pyplot as plt

nx = 101
xmin = -5
xmax = 5
x = np.linspace(xmin, xmax, nx)
dx = (xmax - xmin) / (nx - 1)
c = 1


def func(u, epsilon):
    # Sub-routine to evaluate the RHS function. Uses 2nd-order centered differences to discretize the spatial
    # derivatives
    f = 0 * u.copy()
    f[0] = epsilon * (u[1] - 2 * u[0] + u[-2]) / (dx ** 2) - (u[1] - u[-2]) / (2 * dx)
    f[1:-1] = epsilon * (u[2:] - 2 * u[1:-1] + u[:-2]) / (dx ** 2) - (u[2:] - u[:-2]) / (2 * dx)
    f[-1] = epsilon * (u[1] - 2 * u[-1] + u[-2]) / (dx ** 2) - (u[1] - u[-2]) / (2 * dx)
    return f


def L2norm(u):
    # Sub-routine to evaluate the L2 norm of the error
    return (np.sum((u - np.exp(-x ** 2)) ** 2)) ** 0.5


# Explicit Euler method
# Initial condition
u = np.exp(-x ** 2)

# Define time step
CFL = 0.2
dt = CFL * dx / c

## Undamped case
epsilon = 0
n = 0
t = 0
tmax = 10  # This is the time required for the wave to convect back to its original position
plt.plot(x, u, label='$t=0$')
while t < tmax:
    # For the first time step, assign u_{n-1} = u_{n-2} = u0.
    # For subsequent time steps, assign u_{n-2} = u_{n-1} and u_{n-1} = u_n from prior time step
    # assign u_{n-2} to u_{n-1} from the prior time step
    if n == 0:
        un2 = u.copy()  # u_{n-2}
        un1 = u.copy()  # u_{n-1}
    else:
        un2 = un1.copy()  # u_{n-2}
        un1 = un.copy()  # u_{n-1}
    un = u.copy()  # u_n
    u = un + dt / 12.0 * (23.0 * func(un, epsilon) - 16.0 * func(un1, epsilon) + 5.0 * func(un2, epsilon))
    t += dt
    n += 1

plt.plot(x, u, label='$\epsilon = 0$')
print('Undamped L2norm(error) =', L2norm(u))

## Damped case
u = np.exp(-x ** 2)
epsilon = 5e-4
t = 0
n = 0
while (t < tmax):
    if (n == 0):
        un2 = u.copy()  # u_{n-2}
        un1 = u.copy()  # u_{n-1}
    else:
        un2 = un1.copy()  # u_{n-2}
        un1 = un.copy()  # u_{n-1}
    un = u.copy()  # u_n
    u = un + dt / 12.0 * (23.0 * func(un, epsilon) - 16.0 * func(un1, epsilon) + 5.0 * func(un2, epsilon))
    t += dt
    n += 1
plt.plot(x, u, label='$\epsilon = 5\\times 10^{-4}$')
plt.title('Adams-Bashforth')
plt.grid()
plt.xlabel('$x$')
plt.ylabel('$u$')
plt.legend()
print('Damped L2norm(error) =', L2norm(u))

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
output_file = os.path.join(OUTPUT_FOLDER, f"u_{python_filename}.npy")

# Save the array u in the results folder
np.save(output_file, u)
