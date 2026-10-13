"""
1D Burgers' Equation
---------------------------------------------------------------
- Step 4: Burgers' Equation
- REF: https://github.com/barbagroup/CFDPython
---------------------------------------------------------------
Author: Weichao Li
Date: 2025/03/13
"""
import numpy
import sympy

from sympy import init_printing
init_printing(use_latex=True)

x, nu, t = sympy.symbols('x nu t')
phi = (sympy.exp(-(x - 4 * t)**2 / (4 * nu * (t + 1))) +
       sympy.exp(-(x - 4 * t - 2 * sympy.pi)**2 / (4 * nu * (t + 1))))
phiprime = phi.diff(x)



from sympy.utilities.lambdify import lambdify

u = -2 * nu * (phiprime / phi) + 4


ufunc = lambdify((t, x, nu), u)

from matplotlib import pyplot

###variable declarations
nx = 101
nt = 100
dx = 2 * numpy.pi / (nx - 1)
nu = .07
dt = dx * nu

x = numpy.linspace(0, 2 * numpy.pi, nx)
un = numpy.empty(nx)
t = dt * nt

u = numpy.asarray([ufunc(t, x0, nu) for x0 in x])



pyplot.figure(figsize=(11, 7), dpi=100)
pyplot.plot(x, u, marker='o', lw=2)
pyplot.xlim([0, 2 * numpy.pi])
pyplot.ylim([0, 10])
pyplot.savefig("output.png", dpi=300, bbox_inches="tight")

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
numpy.save(output_file, u)

