"""
1D non-linear convection equation
---------------------------------------------------------------
- Exercise 5 - Finite difference solution of the one-dimensional non-linear convection equation
- Lax method
- REF: https://github.com/okcfdlab/engr491
---------------------------------------------------------------
Author: Weichao Li
Date: 2025/03/13
"""
import numpy as np
import matplotlib.pyplot as plt
import math

nu = 0.5       # CFL number
dt = 0.01      # time step
T = 500        # max number of time steps
dx = dt/nu     # space step
L = 2*math.pi  # domain length

## Define space vector
x = np.linspace(0,L,math.ceil(L/(dt/nu)))
dx = x[1]-x[0]
print('Step size =', dx, 's')
nx = x.size
print('Number of x nodes =', nx)

## Define initial conditions
u0 = np.sin(x) + 0.5*np.sin(0.5*x)
u = np.copy(u0)
ustar = np.zeros(u.size)
unp1 = np.zeros(u.size)
for t in range(0, T):
    unp1[0] = 0.5 * (u[1] + u[nx - 2]) - 0.25 * nu * (u[1] ** 2 - u[nx - 2] ** 2)
    for i in range(1, nx - 1):
        unp1[i] = 0.5 * (u[i + 1] + u[i - 1]) - 0.25 * nu * (u[i + 1] ** 2 - u[i - 1] ** 2)
    unp1[nx - 1] = unp1[0]
    u = np.copy(unp1)

    if (t % 100 == 0):
        string = '$t =$' + str(t * dt) + ' s'
        plt.plot(x, u, label=string)

uL = np.copy(u)
string = '$t =$' + str((t + 1) * dt) + ' s'
plt.plot(x, u, label=string)
plt.xlabel('$x$')
plt.ylabel('$u$')
plt.legend(loc='center left', bbox_to_anchor=(1, 0.5))

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