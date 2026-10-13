"""
1D Diffusion Equation
---------------------------------------------------------------
- Step 3: Diffusion Equation in 1-D
- REF: https://github.com/barbagroup/CFDPython
---------------------------------------------------------------
Author: Weichao Li
Date: 2025/03/13
"""

import numpy  # loading our favorite library
from matplotlib import pyplot  # and the useful plotting library
nx = 41
dx = 2 / (nx - 1)
nt = 20  # the number of timesteps we want to calculate
nu = 0.3  # the value of viscosity
sigma = .2  # sigma is a parameter, we'll learn more about it later
dt = sigma * dx ** 2 / nu  # dt is defined using sigma ... more later!

u = numpy.ones(nx)  # a numpy array with nx elements all equal to 1.
u[int(.5 / dx):int(1 / dx + 1)] = 2  # setting u = 2 between 0.5 and 1 as per our I.C.s

un = numpy.ones(nx)  # our placeholder array, un, to advance the solution in time

for n in range(nt):  # iterate through time
    un = u.copy()  ##copy the existing values of u into un
    for i in range(1, nx - 1):
        u[i] = un[i] + nu * dt / dx ** 2 * (un[i + 1] - 2 * un[i] + un[i - 1])

pyplot.plot(numpy.linspace(0, 2, nx), u)

pyplot.show()

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

