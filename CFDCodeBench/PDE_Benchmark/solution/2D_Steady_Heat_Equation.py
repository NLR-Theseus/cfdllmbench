"""
2D Steady Heat Equation
---------------------------------------------------------------
- Exercise 2 - Solution of the steady two-dimensional heat equation
- REF: https://github.com/okcfdlab/engr491
---------------------------------------------------------------
Author: Weichao Li
Date: 2025/03/12
"""

import numpy as np
import matplotlib.pyplot as plt

max_k = 20000
xlen = 5.
ylen = 4.
dx = 0.05
dy = 0.05
nx = int(xlen / dx + 1)
ny = int(ylen / dy + 1)
beta = dx / dy
omega = 1.78

eps_max = 1e-10

x = np.linspace(0., xlen, nx)
y = np.linspace(0., ylen, ny)

T = np.zeros((nx, ny))

for i in range(1, nx - 1):
    T[i, 0] = 20  # lower wall = 20C
    T[i, ny - 1] = 0  # top wall = 0C

for i in range(ny - 1):
    T[0, i] = 10  # left wall = 10C
    T[nx - 1, i] = 40  # right wall = 40C

Tkp1 = np.copy(T)
eps = []

print('k', '|', 'Residual')
print('------------')

for k in range(max_k):
    for i in range(1, nx - 1):
        for j in range(1, ny - 1):
            # Point Gauss-Seidel method
            Tkp1[i, j] = 0.5 / (1.0 + beta ** 2) * (T[i + 1, j] + Tkp1[i - 1, j]) + (0.5 * beta ** 2) / (
                        1.0 + beta ** 2) * (T[i, j + 1] + Tkp1[i, j - 1])

    # Define residual error as the maximum absolute difference between subsequent iterates
    eps = np.append(eps, np.amax(np.absolute(np.subtract(Tkp1, T))))

    # Copy new temperature field to old temperature array
    T = np.copy(Tkp1)

    # Print the iteration number and residual to standard output every 25 iterations
    if (k % 25) == 0:
        print(k, '|', eps[k])

    # Test to see if residual error is below threshold; break if yes
    if eps[k] < eps_max:
        print('Residual threshold reached in', k, 'iterations')
        break

# Create subfigures
fig, axes = plt.subplots(1, 2, figsize=(12, 5))

X, Y = np.meshgrid(x, y, indexing='ij')
# Temperature distribution plot
contour = axes[0].contourf(X, Y, T, 30, cmap='coolwarm')
cbar = plt.colorbar(contour, ax=axes[0])
cbar.ax.set_ylabel('T ($^{\circ}$C)')
axes[0].set_xlabel('x')
axes[0].set_ylabel('y')
axes[0].set_title('Temperature Distribution')

# Convergence plot
axes[1].plot(np.linspace(0, k, k + 1), eps, label='PSOR', color='b')
axes[1].set_yscale('log')
axes[1].set_xlabel('Iteration')
axes[1].set_ylabel('$\epsilon$')
axes[1].legend()
axes[1].set_title('Convergence History')

plt.tight_layout()
plt.savefig("output.png", dpi=300, bbox_inches="tight")

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
output_file = os.path.join(OUTPUT_FOLDER, f"T_{python_filename}.npy")

# Save the array u in the results folder
np.save(output_file, T.T)  # saved as (ny, nx)
