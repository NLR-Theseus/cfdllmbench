"""
2D Navier-Stokes equation in a cavity
---------------------------------------------------------------
- Exercise 8 - 2D Navier-Stokes equation in a cavity
- REF: https://github.com/okcfdlab/engr491
---------------------------------------------------------------
Author: Weichao Li
Date: 2025/03/13
"""


import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import pandas as pd


def build_up_b(b, rho, dt, u, v, dx, dy):
    b[1:-1, 1:-1] = (rho * (1 / dt *
                            ((u[1:-1, 2:] - u[1:-1, 0:-2]) /
                             (2 * dx) + (v[2:, 1:-1] - v[0:-2, 1:-1]) / (2 * dy)) -
                            ((u[1:-1, 2:] - u[1:-1, 0:-2]) / (2 * dx)) ** 2 -
                            2 * ((u[2:, 1:-1] - u[0:-2, 1:-1]) / (2 * dy) *
                                 (v[1:-1, 2:] - v[1:-1, 0:-2]) / (2 * dx)) -
                            ((v[2:, 1:-1] - v[0:-2, 1:-1]) / (2 * dy)) ** 2))

    return b


def pressure_poisson(p, dx, dy, b):
    # Solve pressure Poisson equation using Jacobi iteration

    TOL = 1E-4
    MAX = 10000
    l2norm = 1
    l = 0
    while (l2norm > TOL):
        if (l > MAX):
            raise Exception('Pressure system not converged after 10,000 iterations')
        pn = p.copy()
        p[1:-1, 1:-1] = (((pn[1:-1, 2:] + pn[1:-1, 0:-2]) * dy ** 2 +
                          (pn[2:, 1:-1] + pn[0:-2, 1:-1]) * dx ** 2) /
                         (2 * (dx ** 2 + dy ** 2)) -
                         dx ** 2 * dy ** 2 / (2 * (dx ** 2 + dy ** 2)) *
                         b[1:-1, 1:-1])

        p[:, -1] = p[:, -2]  # dp/dx = 0 at x = 2
        p[0, :] = p[1, :]  # dp/dy = 0 at y = 0
        p[:, 0] = p[:, 1]  # dp/dx = 0 at x = 0
        p[-1, :] = 0  # p = 0 at y = 2
        l2norm = (np.sum((p - pn) ** 2)) ** 0.5
        l += 1
    return p, l


def feval(p, u, v, dx, dy, rho, nu):
    # Sub-routine to evaluate the RHS function of the u-velocity equation.
    # Uses 2nd-order centered differences to discretize the spatial derivatives
    # in the convective and viscous terms
    f = 0 * u.copy()

    f[1:-1, 1:-1] = (-u[1:-1, 1:-1] / (2 * dx) *
                     (u[1:-1, 2:] - u[1:-1, 0:-2]) -
                     v[1:-1, 1:-1] / (2 * dy) *
                     (u[2:, 1:-1] - u[0:-2, 1:-1]) -
                     1 / (2 * rho * dx) * (p[1:-1, 2:] - p[1:-1, 0:-2]) +
                     nu * (
                             1 / dx ** 2 * (u[1:-1, 2:] - 2 * u[1:-1, 1:-1] + u[1:-1, 0:-2]) +
                             1 / dy ** 2 * (u[2:, 1:-1] - 2 * u[1:-1, 1:-1] + u[0:-2, 1:-1])
                     )
                     )

    return f


def geval(p, u, v, dx, dy, rho, nu):
    # Sub-routine to evaluate the RHS function of the v-velocity equation.
    # Uses 2nd-order centered differences to discretize the spatial derivatives
    # in the convective and viscous terms
    g = 0 * v.copy()

    g[1:-1, 1:-1] = (-u[1:-1, 1:-1] / (2 * dx) *
                     (v[1:-1, 2:] - v[1:-1, 0:-2]) -
                     v[1:-1, 1:-1] / (2 * dy) *
                     (v[2:, 1:-1] - v[0:-2, 1:-1]) -
                     1 / (2 * rho * dy) * (p[2:, 1:-1] - p[0:-2, 1:-1]) +
                     nu * (1 / dx ** 2 *
                           (v[1:-1, 2:] - 2 * v[1:-1, 1:-1] + v[1:-1, 0:-2]) +
                           1 / dy ** 2 *
                           (v[2:, 1:-1] - 2 * v[1:-1, 1:-1] + v[0:-2, 1:-1])))

    return g


def solveU(p, un, vn, dt, dx, dy, rho, nu, DDT):
    if (DDT == 'EE'):
        # Euler explicit time integration
        us = un + dt * feval(p, un, vn, dx, dy, rho, nu)
        vs = vn + dt * geval(p, un, vn, dx, dy, rho, nu)
    else:
        # Runge-Kutta 4th order
        f1 = feval(p, un, vn, dx, dy, rho, nu)
        g1 = geval(p, un, vn, dx, dy, rho, nu)
        u1 = un + 0.5 * dt * f1
        v1 = vn + 0.5 * dt * g1
        f2 = feval(p, u1, v1, dx, dy, rho, nu)
        g2 = geval(p, u1, v1, dx, dy, rho, nu)
        u2 = un + 0.5 * dt * f2
        v2 = vn + 0.5 * dt * g2
        f3 = feval(p, u2, v2, dx, dy, rho, nu)
        g3 = geval(p, u2, v2, dx, dy, rho, nu)
        u3 = un + 1.0 * dt * f3
        v3 = vn + 1.0 * dt * g3
        f4 = feval(p, u3, v3, dx, dy, rho, nu)
        g4 = geval(p, u3, v3, dx, dy, rho, nu)
        us = un + (1.0 / 6.0) * dt * (f1 + 2.0 * f2 + 2.0 * f3 + f4)
        vs = vn + (1.0 / 6.0) * dt * (g1 + 2.0 * g2 + 2.0 * g3 + g4)

    # Enforce boundary conditions
    us[0, :] = 0
    us[:, 0] = 0
    us[:, -1] = 0
    us[-1, :] = 1  # set velocity on cavity lid equal to 1
    vs[0, :] = 0
    vs[-1, :] = 0
    vs[:, 0] = 0
    vs[:, -1] = 0

    return us, vs


# Define problem parameters
Re = 100.0
CFL = 0.25
nt = 2000

# Select time scheme
# Options: 'RK' = Runge-Kutta 4th order; 'EE' = Euler explicit
DDT = 'RK'

# Define spatial grid info
Lx = 1.0
Ly = 1.0
nx = 51
ny = 51
dx = Lx / (nx - 1)
dy = Ly / (ny - 1)
rho = 1
x = np.linspace(0, Lx, nx)
y = np.linspace(0, Ly, ny)

[X, Y] = np.meshgrid(x, y)

# Define time step info
dt = CFL * dx / 1.0
nu = Lx * 1.0 / Re
print('Time step=', dt, 's')

# Define initial conditions
u = np.zeros((ny, nx))
v = np.zeros((ny, nx))
p = np.zeros((ny, nx))

# Define boundary conditions
u[-1, :] = 1.0

un = np.empty_like(u)
vn = np.empty_like(v)
b = np.zeros((ny, nx))
l2normU = np.zeros(1)
l2normV = np.zeros(1)

for n in range(nt):
    if (n % 10 == 0):
        print('Timestep:', n)

    un = u.copy()
    vn = v.copy()

    # Solve velocity equation for intermediate velocity
    [u, v] = solveU(p, un, vn, dt, dx, dy, rho, nu, DDT)

    # Compute mass source term
    b = build_up_b(b, rho, dt, u, v, dx, dy)

    # Solve pressure Poisson equation for pressure correction
    [p, l] = pressure_poisson(p, dx, dy, b)

    if (n % 10 == 0):
        print('  Iterations required for pressure Poisson eqn:', l)

    # Evaluate equation L2 norms
    l2normU = np.append(l2normU, (np.sum((u - un) ** 2)) ** 0.5)
    l2normV = np.append(l2normV, (np.sum((v - vn) ** 2)) ** 0.5)

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
output_file_p = os.path.join(OUTPUT_FOLDER, f"p_{python_filename}.npy")

# Save the array u in the results folder
np.save(output_file_u, u)
np.save(output_file_v, v)
np.save(output_file_p, p)