"""
2D Laplace Equation
---------------------------------------------------------------
- Step 9: 2D Laplace Equation
- REF: https://github.com/barbagroup/CFDPython
---------------------------------------------------------------
v2 reference: the boundary-value problem
    p_xx + p_yy = 0 on [0, 2] x [0, 1]
    p = 0 at x = 0,  p = y at x = 2,  dp/dy = 0 at y = 0 and y = 1
has the exact solution
    p = x/4 - sum_{n odd} 4/(n pi)^2 * sinh(n pi x)/sinh(2 n pi) * cos(n pi y),
which is evaluated here on the 31 x 31 output grid (the v1 Jacobi solve used the wrong dy
and stopped before converging).
"""
import os
import numpy

nx = 31
ny = 31
x = numpy.linspace(0, 2, nx)
y = numpy.linspace(0, 1, ny)
X, Y = numpy.meshgrid(x, y)          # arrays are (ny, nx)

p = X / 4
for n in range(1, 4001, 2):
    a = n * numpy.pi
    # sinh(a X)/sinh(2a) written in an overflow-free form
    ratio = numpy.exp(a * (X - 2)) * (1 - numpy.exp(-2 * a * X)) / (1 - numpy.exp(-4 * a))
    p -= 4 / a ** 2 * ratio * numpy.cos(a * Y)
p[:, 0], p[:, -1] = 0.0, y           # boundary values exactly (removes series truncation at the corner)

##########################################################################
ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))  # PDE_Benchmark root
OUTPUT_FOLDER = os.path.join(ROOT_DIR, "results", "solution")
os.makedirs(OUTPUT_FOLDER, exist_ok=True)
python_filename = os.path.splitext(os.path.basename(__file__))[0]
numpy.save(os.path.join(OUTPUT_FOLDER, f"p_{python_filename}.npy"), p)
