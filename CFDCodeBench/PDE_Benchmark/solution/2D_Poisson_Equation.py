"""
2D Poisson Equation
---------------------------------------------------------------
- Step 10: 2D Poisson Equation
- REF: https://github.com/barbagroup/CFDPython
---------------------------------------------------------------
v2 reference: p_xx + p_yy = b on [0, 2] x [0, 1], p = 0 on all boundaries, with two Gaussian sources
    b = 100 exp(-r1^2 / (2 sigma^2)) - 100 exp(-r2^2 / (2 sigma^2)),  sigma = 0.05,
centred at (Lx/4, Ly/4) and (3Lx/4, 3Ly/4). The v1 single-grid-point sources made the answer depend on
the grid, and the v1 solve stopped after 100 Jacobi iterations. Here the system is solved directly on a
fine grid (second-order central differences) and sampled on the 51 x 101 output grid.
"""
import os
import numpy
import scipy.sparse as sp
import scipy.sparse.linalg as spla

Lx, Ly, sigma = 2.0, 1.0, 0.05
fine = 4                                   # fine-grid refinement factor relative to the output grid
nx_out, ny_out = 101, 51                   # output grid (ny, nx), spacing 0.02 in both directions
nx, ny = (nx_out - 1) * fine + 1, (ny_out - 1) * fine + 1
x = numpy.linspace(0, Lx, nx)
y = numpy.linspace(0, Ly, ny)
X, Y = numpy.meshgrid(x, y)
dx, dy = x[1] - x[0], y[1] - y[0]


def source(X, Y):
    g = lambda xc, yc: numpy.exp(-((X - xc) ** 2 + (Y - yc) ** 2) / (2 * sigma ** 2))
    return 100 * g(Lx / 4, Ly / 4) - 100 * g(3 * Lx / 4, 3 * Ly / 4)


b = source(X, Y)
D = lambda n, h: sp.diags([1.0, -2.0, 1.0], [-1, 0, 1], shape=(n, n)) / h ** 2
A = sp.kron(sp.identity(ny - 2), D(nx - 2, dx)) + sp.kron(D(ny - 2, dy), sp.identity(nx - 2))
p = numpy.zeros((ny, nx))
p[1:-1, 1:-1] = spla.spsolve(A.tocsc(), b[1:-1, 1:-1].ravel()).reshape(ny - 2, nx - 2)
p_out = p[::fine, ::fine]                  # (51, 101), both end points included

##########################################################################
ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))  # PDE_Benchmark root
OUTPUT_FOLDER = os.path.join(ROOT_DIR, "results", "solution")
os.makedirs(OUTPUT_FOLDER, exist_ok=True)
python_filename = os.path.splitext(os.path.basename(__file__))[0]
numpy.save(os.path.join(OUTPUT_FOLDER, f"p_{python_filename}.npy"), p_out)
