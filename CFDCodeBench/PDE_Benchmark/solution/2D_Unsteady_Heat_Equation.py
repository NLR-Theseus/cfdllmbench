"""
2D Unsteady Heat Equation
---------------------------------------------------------------
- Exercise 3 - Unsteady two-dimensional heat equation
- This code will use Alternating direction implicit (ADI)
- REF: https://github.com/okcfdlab/engr491
---------------------------------------------------------------
Author: Weichao Li
Date: 2025/03/12
"""
import numpy as np
import time
import matplotlib.pyplot as plt


# Sub-routine that defines the source term:
def Q(x, y):
    sigma = 0.1
    return np.exp(-(x ** 2 + y ** 2) / (2 * sigma ** 2))


# Initial and boundary conditions
def ICBC(X, Y, nx, ny, T0, Q0):
    q = Q0 * Q(X, Y)
    T = T0 * np.ones((ny, nx)) + q
    T[:, 0] = T0
    T[:, nx - 1] = T0
    T[0, :] = T0
    T[ny - 1, :] = T0
    return T


def TDMA(a, b, c, d, x):
    nf = np.size(d)  # number of equations

    # Step 1 - forward sweep
    for i in range(1, nf):
        w = a[i - 1] / b[i - 1]
        b[i] = b[i] - w * c[i - 1]
        d[i] = d[i] - w * d[i - 1]

    # Step 2 - back substitution
    x[-1] = d[-1] / b[-1]
    for i in range(nf - 2, -1, -1):
        x[i] = (d[i] - c[i] * x[i + 1]) / b[i]

    return x


# Simple explicit method
def ADI(T, X, Y, q, alpha, beta, dx, tmax, nx, ny, r):
    # ADI sub-routine. Output is a tuple resultsADI:
    #   resultsADI[0] = solution time vector
    #   resultsADI[1] = vector of max temperature in domain as a function of time
    #   resultsADI[2] = number of time steps taken

    r = r / (1 + beta ** 2)
    dt = r * dx ** 2 / alpha
    nsteps = int(np.ceil(tmax / dt))
    dt = tmax / nsteps
    r = alpha * dt / dx ** 2
    n = 0
    Tmax = np.max(T)
    t = [0]
    while (n < nsteps):
        Tn = T.copy()
        # Alternate the order of the implicit direction on odd-even time steps. For even, solve x implicit first, then y
        if (n % 2 == 0):
            # Step 1: implicit in x, explicit in y
            for j in range(1, ny - 1):
                a = -0.5 * r * np.ones(nx - 1)
                b = (1 + r) * np.ones(nx)
                c = a.copy()
                d = 0.5 * (beta ** 2) * r * (Tn[j + 1, :] - 2 * Tn[j, :] + Tn[j - 1, :]) + Tn[j, :] + 0.5 * dt * q[j, :]

                # enforce boundary conditions
                b[0] = 1
                c[0] = 0
                a[nx - 2] = 0
                b[nx - 1] = 1
                d[0] = 1
                d[nx - 1] = 1

                TDMA(a, b, c, d, Tn[j, :])

                # Step 2: implicit in y, explicit in x
            for i in range(1, nx - 1):
                a = -0.5 * (beta ** 2) * r * np.ones(ny - 1)
                b = (1 + (beta ** 2) * r) * np.ones(ny)
                c = a.copy()
                d = 0.5 * r * (Tn[:, i + 1] - 2 * Tn[:, i] + Tn[:, i - 1]) + Tn[:, i] + 0.5 * dt * q[:, i]

                # enforce boundary conditions
                b[0] = 1
                c[0] = 0
                a[ny - 2] = 0
                b[ny - 1] = 1
                d[0] = 1
                d[ny - 1] = 1

                TDMA(a, b, c, d, T[:, i])

        # Alternate the order of the implicit direction on odd-even time steps. For odd, solve y implicit first, then x
        else:
            # Step 1: implicit in y, explicit in x
            for i in range(1, nx - 1):
                a = -0.5 * (beta ** 2) * r * np.ones(ny - 1)
                b = (1 + (beta ** 2) * r) * np.ones(ny)
                c = a.copy()
                d = 0.5 * r * (Tn[:, i + 1] - 2 * Tn[:, i] + Tn[:, i - 1]) + Tn[:, i] + 0.5 * dt * q[:, i]

                # enforce boundary conditions
                b[0] = 1
                c[0] = 0
                a[ny - 2] = 0
                b[ny - 1] = 1
                d[0] = 1
                d[ny - 1] = 1

                TDMA(a, b, c, d, Tn[:, i])

            # Step 1: implicit in x, explicit in y
            for j in range(1, ny - 1):
                a = -0.5 * r * np.ones(nx - 1)
                b = (1 + r) * np.ones(nx)
                c = a.copy()
                d = 0.5 * (beta ** 2) * r * (Tn[j + 1, :] - 2 * Tn[j, :] + Tn[j - 1, :]) + Tn[j, :] + 0.5 * dt * q[j, :]

                # enforce boundary conditions
                b[0] = 1
                c[0] = 0
                a[nx - 2] = 0
                b[nx - 1] = 1
                d[0] = 1
                d[nx - 1] = 1

                TDMA(a, b, c, d, T[j, :])
        n += 1
        Tmax = np.append(Tmax, np.max(T))
        t = np.append(t, n * dt)
    return t, Tmax, n, T


### Define grid and simulation parameters ###
xmin, xmax, ymin, ymax = -1, 1, -1, 1
nx, ny = 41, 41
dx = (xmax - xmin) / (nx - 1)
beta = 1
x, y = np.linspace(xmin, xmax, nx), np.linspace(ymin, ymax, ny)
X, Y = np.meshgrid(x, y)

alpha = 1
tmax = 3
T0, Q0 = 1, 200

q = Q0 * Q(X, Y)
T = ICBC(X, Y, nx, ny, T0, Q0)

print('Running ADI...')
r = 0.5
tic = time.perf_counter()
resultADI = ADI(T, X, Y, q, alpha, beta, dx, tmax, nx, ny, r)
toc = time.perf_counter()
print(f'ADI finished in {resultADI[2]} timesteps. Execution took {toc - tic} seconds')

# Plot results
plt.figure(figsize=(4, 3), dpi=200)
plt.loglog(resultADI[0], resultADI[1], ls='solid', label='Simple Explicit')
plt.xlabel('$t$ (s)')
plt.ylabel('$T_{max}$ ($^{\circ}$C)')
plt.legend(frameon=False, fontsize=8)

# Plot only the final temperature distribution
plt.figure(figsize=(5, 4), dpi=200)
c = plt.contourf(X, Y, resultADI[-1], cmap='coolwarm')
plt.colorbar(label='$T$ ($^{\circ}$C)')
plt.xlabel('x')
plt.ylabel('y')
plt.title(f'Temperature Distribution at final time: {tmax} s')

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
output_file = os.path.join(OUTPUT_FOLDER, f"T_{python_filename}.npy")

# Save the array u in the results folder
np.save(output_file, T)
