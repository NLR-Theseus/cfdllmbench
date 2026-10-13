"""
Euler's equation for compressible flow in a shock tube
---------------------------------------------------------------
- Exercise 7 - Solving Euler's equation for compressible flow in a shock tube
- MacCormack method
- REF: https://github.com/okcfdlab/engr491
---------------------------------------------------------------
Author: Weichao Li
Date: 2025/03/13
"""
import numpy as np
import matplotlib.pyplot as plt


# Sub-routine that extracts primitive variables from the solution vector

def primitive(U):
    rho = U[:, 0]
    u = U[:, 1] / rho
    E = U[:, 2] / rho
    p = (gamma - 1) * (U[:, 2] - 0.5 * rho * u ** 2)
    return rho, u, E, p


# Sub-routine that generates a flux vector from primitive variables

def flux(rho, u, p, E):
    FF = np.zeros((np.size(rho), 3))
    FF[:, 0] = rho * u
    FF[:, 1] = rho * u ** 2 + p
    FF[:, 2] = rho * u * E + p * u
    return FF


# Define spatial coordinates
xL = -1
xR = 1

# Define number of mesh points
n = 81

# Define spatial step size
dx = (xR - xL) / n

# Define initial conditions on the L (left) and R (right) side of the membrane
rhoL = 1.0
pL = 1.0
uL = 0
rhoR = 0.125
pR = 0.1
uR = 0

# Gas specific heat ratio
gamma = 1.4

# Define how long the solution will be computed for (in seconds)
tmax = 0.25

# Initialize the solution and flux vectors
U = np.zeros((n, 3))
FF = np.zeros((n, 3))

U[0:int(n / 2), 0] = rhoL
U[int(n / 2):, 0] = rhoR
U[0:int(n / 2), 1] = rhoL * uL
U[int(n / 2):, 1] = rhoR * uR
U[0:int(n / 2), 2] = pL / (gamma - 1) + 0.5 * rhoL * uL ** 2
U[int(n / 2):, 2] = pR / (gamma - 1) + 0.5 * rhoR * uR ** 2

FF[0:int(n / 2), 0] = rhoL * uL
FF[int(n / 2):, 0] = rhoR * uR
FF[0:int(n / 2), 1] = rhoL * uL ** 2 + pL
FF[int(n / 2):, 1] = rhoR * uR ** 2 + pR
FF[0:int(n / 2), 2] = (pL / (gamma - 1) + 0.5 * rhoL * uL ** 2 + pL) * uL
FF[int(n / 2):, 2] = (pR / (gamma - 1) + 0.5 * rhoR * uR ** 2 + pR) * uR

# Get primitive variable vectors from the solution vector.
[rho, u, E, p] = primitive(U)

# Define solution information
eps = 0.01  # - dissipation constant - used for smoothing the solution
t = 0  # - initializet time to zero
l = 0  # - this is a counter for counting the number of time steps
CFL = 1  # - CFL number - it is defined for specifying the time step size

# Solve in time
while (t < tmax):

    # Evaluate the time step size based on the user-specified CFL condition:
    # CFL = (|umax| + a) dx / dt for all cells

    amax = np.max((gamma * p / rho) ** 0.5)  # - max speed of sound in all cells
    umax = np.max(np.abs(u))  # - max absolute velocity in all cells
    dt = CFL * dx / (umax + amax)  # - define time step size

    # Print an output message every ten time steps
    if (l % 10 == 0):
        print('Step', l, ' t=', t, 'umax=', umax, 'amax=', amax, 'dt =', dt)

    # Evaluate predictor step
    Ubar = U.copy()
    Ubar[1:-1, :] = U[1:-1, :] - dt / dx * (FF[2:, :] - FF[1:-1, :])

    # Add smoothing to remove numerical dispersion error
    Ubar[2:-2, :] = Ubar[2:-2, :] - eps * (
                Ubar[4:, :] - 4 * Ubar[3:-1, :] + 6 * Ubar[2:-2, :] - 4 * Ubar[1:-3, :] + Ubar[:-4, :])

    # Convert solution vector to primitive variables
    [rho, u, E, p] = primitive(Ubar)
    # Reconstruct flux vector FF using the updated primitive variables
    FFbar = flux(rho, u, p, E)

    # Evaluate corrector step
    U[1:-1, :] = 0.5 * (U[1:-1, :] + Ubar[1:-1, :]) - 0.5 * dt / dx * (FFbar[1:-1, :] - FFbar[:-2, :])

    # Add smoothing
    U[2:-2, :] = U[2:-2, :] - eps * (U[4:, :] - 4 * U[3:-1, :] + 6 * U[2:-2, :] - 4 * U[1:-3, :] + U[:-4, :])

    # Update flux vector
    [rho, u, E, p] = primitive(U)
    FF = flux(rho, u, p, E)

    # Update time
    t += dt
    l += 1

print('Solution obtained in', l - 1, 'time steps.')

##########################################################################
import os

# === Paths ===
ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))  # PDE_Benchmark root
OUTPUT_FOLDER = os.path.join(ROOT_DIR, "results", "solution")

# Ensure the output directory exists
os.makedirs(OUTPUT_FOLDER, exist_ok=True)
# Get the current Python file name without extension
python_filename = os.path.splitext(os.path.basename(__file__))[0]

# # Define the file name dynamically
# output_file_u = os.path.join(OUTPUT_FOLDER, f"U_{python_filename}.npy")
# output_file_f = os.path.join(OUTPUT_FOLDER, f"F_{python_filename}.npy")
#
# # Save the array u in the results folder
# np.save(output_file_u, U)
# np.save(output_file_f, FF)
output_file_rho = os.path.join(OUTPUT_FOLDER, f"rho_{python_filename}.npy")
output_file_u = os.path.join(OUTPUT_FOLDER, f"u_{python_filename}.npy")
output_file_E = os.path.join(OUTPUT_FOLDER, f"E_{python_filename}.npy")
np.save(output_file_rho, rho)
np.save(output_file_u, u)
np.save(output_file_E, E)
