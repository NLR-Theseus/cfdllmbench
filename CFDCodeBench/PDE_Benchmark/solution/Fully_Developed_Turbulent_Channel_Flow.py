"""
Fully-developed turbulent flow in a channel
---------------------------------------------------------------
- Exercise 9 - Fully-developed turbulent flow in a channel
- SA, SST, V2F, KE
- REF: https://github.com/okcfdlab/engr491
---------------------------------------------------------------
Author: Weichao Li
Date: 2025/03/13
"""


def solveRANS(rho, mu, mesh, turbModel, ReTau):
    n = mesh.nPoints
    u = np.zeros(n)  # velocity
    mut = np.zeros(n)  # eddy viscosity

    k = 0.01 * np.ones(n)  # turbulent kinetic energy
    e = 0.001 * np.ones(n)  # turbulent dissipation
    v2 = 1 / 3 * k  # wall normal turbulent fluctuations for V2F model
    om = np.ones(n)  # specific turbulent dissipation for omega in SST
    nuSA = np.ones(n) / ReTau  # eddy viscisty for SA model

    residual = 1.0e20
    iterations = 0

    print("Start iterating with", turbModel, "turbulence model")

    while residual > 1.0e-6 and iterations < 10000:

        # Solve turbulence model to calculate eddy viscosity
        if turbModel == "Cess":
            mut = Cess(rho, mu, ReTau, mesh)
        elif turbModel == "SA":
            mut, nuSA = SA(u, nuSA, rho, mu, mesh)
        elif turbModel == "MK":
            mut, k, e = MK(u, k, e, rho, mu, ReTau, mesh)
        elif turbModel == "SST":
            mut, k, om = SST(u, k, om, rho, mu, mesh)
        elif turbModel == "V2F":
            mut, k, e, v2 = V2F(u, k, e, v2, rho, mu, mesh)
        else:
            mut = np.zeros(n)

        # Solve momentum equation:  0 = d/dy[(mu+mut)dudy] - 1
        # diffusion matrix: mueff*d2phi/dy2 + dmueff/dy dphi/dy
        A = np.einsum('i,ij->ij', mesh.ddy @ (mu + mut), mesh.ddy) + np.einsum('i,ij->ij', mu + mut, mesh.d2dy2)

        # Solve
        u_old = u.copy()
        u[1:n - 1] = np.linalg.solve(A[1:n - 1, 1:n - 1], -np.ones(n - 2))
        residual = np.linalg.norm(u - u_old) / n

        # Printing residuals
        if iterations % 100 == 0: print("iteration: ", iterations, ", Residual(u) = ", residual)
        iterations = iterations + 1

    print("iteration: ", iterations, ", Residual(u) = ", residual)

    return u, mut, k, e, om


def Cess(r, mu, ReTau, mesh):
    # distance to the wall
    d = np.minimum(mesh.y, mesh.y[-1] - mesh.y)

    # Model constants
    kappa = 0.426
    A = 25.4
    ReTauArr = np.ones(mesh.nPoints) * ReTau
    # distance to the wall in wall units
    yplus = d * ReTauArr

    df = 1 - np.exp(-yplus / A)
    t1 = np.power(2 * d - d * d, 2)
    t2 = np.power(3 - 4 * d + 2 * d * d, 2)
    mut = 0.5 * np.power(1 + 1 / 9 * np.power(kappa * ReTauArr, 2) * (t1 * t2) * df * df, 0.5) - 0.5

    return mut * mu


def SA(u, nuSA, r, mu, mesh):
    n = mesh.nPoints

    # Model constants
    cv1_3 = np.power(7.1, 3.0)
    cb1 = 0.1355
    cb2 = 0.622
    cb3 = 2.0 / 3.0
    inv_cb3 = 1.0 / cb3
    kappa_2 = np.power(0.41, 2.0)
    cw1 = cb1 / kappa_2 + (1.0 + cb2) / cb3
    cw2 = 0.3
    cw3_6 = np.power(2.0, 6.0)

    # Model functions
    strMag = np.absolute(mesh.ddy @ u)  # VortRate = StrainRate in fully developed channel
    wallDist = np.minimum(mesh.y, mesh.y[-1] - mesh.y)
    wallDist = np.maximum(wallDist, 1.0e-8)
    inv_wallDist2 = 1 / np.power(wallDist, 2)

    chi = nuSA * r / mu
    fv1 = np.power(chi, 3) / (np.power(chi, 3) + cv1_3)
    fv2 = 1.0 - (chi / (1.0 + (chi * fv1)))
    inv_kappa2_d2 = inv_wallDist2 * (1.0 / kappa_2)
    Shat = strMag + inv_kappa2_d2 * fv2 * nuSA
    inv_Shat = 1.0 / Shat
    r_SA = np.minimum(nuSA * inv_kappa2_d2 * inv_Shat, 10.0)
    g = r_SA + cw2 * (np.power(r_SA, 6) - r_SA)
    g_6 = np.power(g, 6)
    fw_ = np.power(((1.0 + cw3_6) / (g_6 + cw3_6)), (1 / 6))
    fw = g * fw_

    # Eddy viscosity
    mut = np.zeros(n)
    mut[1:-1] = fv1[1:-1] * nuSA[1:-1] * r
    mut[1:-1] = np.minimum(np.maximum(mut[1:-1], 0.0), 100.0)

    nueff = (mu / r + nuSA)
    fs = np.ones(n)
    fd = np.ones(n)

    # ---------------------------------------------------------------------
    # nuSA-equation

    # diffusion matrix: mueff*d2()/dy2 + dmueff/dy d()/dy
    A = np.einsum('i,ij->ij', nueff * fd, mesh.d2dy2) \
        + np.einsum('i,ij->ij', (mesh.ddy @ nueff) * fd, mesh.ddy)
    A = inv_cb3 * A

    # implicitly treated source term
    np.fill_diagonal(A, A.diagonal() - cw1 * fw * nuSA * inv_wallDist2)

    # Right hand side
    dnudy = mesh.ddy @ (fs * nuSA)
    b = - cb1 * Shat[1:-1] * nuSA[1:-1] - cb2 * inv_cb3 * np.power(dnudy[1:-1], 2)

    # Wall boundary conditions
    nuSA[0] = nuSA[-1] = 0.0

    # Solve
    nuSA = solveEqn(nuSA, A, b, 0.75)
    nuSA[1:-1] = np.maximum(nuSA[1:-1], 1.e-12)

    return mut, nuSA


def MK(u, k, e, r, mu, ReTau, mesh):
    n = mesh.nPoints
    d = np.minimum(mesh.y, mesh.y[-1] - mesh.y)

    yplus = d * ReTau

    # Model constants
    cmu = 0.09
    sigk = 1.4
    sige = 1.3
    Ce1 = 1.4
    Ce2 = 1.8

    # Model functions
    ReTurb = r * np.power(k, 2) / (mu * e)
    f2 = (1 - 2 / 9 * np.exp(-np.power(ReTurb / 6, 2))) * np.power(1 - np.exp(-yplus / 5), 2)
    fmue = (1 - np.exp(-yplus / 70)) * (1.0 + 3.45 / np.power(ReTurb, 0.5))
    fmue[0] = fmue[-1] = 0.0

    # eddy viscosity
    mut = cmu * fmue * r / e * np.power(k, 2)
    mut[1:-1] = np.minimum(np.maximum(mut[1:-1], 1.0e-10), 100.0)

    # Turbulent production: Pk = mut*dudy^2
    Pk = mut * np.power(mesh.ddy @ u, 2)

    # ---------------------------------------------------------------------
    # e-equation

    # effective viscosity
    mueff = mu + mut / sige
    fs = fd = np.ones(n)

    # diffusion matrix: mueff*d2()/dy2 + dmueff/dy d()/dy
    A = np.einsum('i,ij->ij', mueff * fd, mesh.d2dy2) \
        + np.einsum('i,ij->ij', (mesh.ddy @ mueff) * fd, mesh.ddy)

    # Left-hand-side, implicitly treated source term
    np.fill_diagonal(A, A.diagonal() - Ce2 * f2 * r * e / k / fs)

    # Right-hand-side
    b = -e[1:-1] / k[1:-1] * Ce1 * Pk[1:-1]

    # Wall boundary conditions
    e[0] = mu / r * k[1] / np.power(d[1], 2)
    e[-1] = mu / r * k[-2] / np.power(d[-2], 2)

    # Solve eps equation
    e = solveEqn(e * fs, A, b, 0.8) / fs
    e[1:-1] = np.maximum(e[1:-1], 1.e-12)

    # ---------------------------------------------------------------------
    # k-equation

    mueff = mu + mut / sigk
    fs = fd = np.ones(n)

    # diffusion matrix: mueff*d2()/dy2 + dmueff/dy d()/dy
    A = np.einsum('i,ij->ij', mueff * fd, mesh.d2dy2) \
        + np.einsum('i,ij->ij', (mesh.ddy @ mueff) * fd, mesh.ddy)

    # implicitly treated source term
    np.fill_diagonal(A, A.diagonal() - r * e / k / fs)

    # Right-hand-side
    b = -Pk[1:-1]

    # Wall boundary conditions
    k[0] = k[-1] = 0.0

    # Solve TKE
    k = solveEqn(k * fs, A, b, 0.7) / fs
    k[1:-1] = np.maximum(k[1:-1], 1.e-12)

    return mut, k, e


def SST(u, k, om, r, mu, mesh):
    n = mesh.nPoints

    # model constants
    sigma_k1 = 0.85
    sigma_k2 = 1.0
    sigma_om1 = 0.5
    sigma_om2 = 0.856
    beta_1 = 0.075
    beta_2 = 0.0828
    betaStar = 0.09
    a1 = 0.31
    alfa_1 = beta_1 / betaStar - sigma_om1 * 0.41 ** 2.0 / betaStar ** 0.5
    alfa_2 = beta_2 / betaStar - sigma_om2 * 0.41 ** 2.0 / betaStar ** 0.5

    # Relaxation factors
    underrelaxK = 0.6
    underrelaxOm = 0.4

    # required gradients
    dkdy = mesh.ddy @ k
    domdy = mesh.ddy @ om

    wallDist = np.minimum(mesh.y, mesh.y[-1] - mesh.y)
    wallDist = np.maximum(wallDist, 1.0e-8)

    # VortRate = StrainRate in fully developed channel
    strMag = np.absolute(mesh.ddy @ u)

    # Blending functions
    CDkom = 2.0 * sigma_om2 * r / om * dkdy * domdy
    gamma1 = 500.0 * mu / (r * om * wallDist * wallDist)
    gamma2 = 4.0 * sigma_om2 * r * k / (wallDist * wallDist * np.maximum(CDkom, 1.0e-20))
    gamma3 = np.sqrt(k) / (betaStar * om * wallDist)
    gamma = np.minimum(np.maximum(gamma1, gamma3), gamma2)
    bF1 = np.tanh(np.power(gamma, 4.0))
    gamma = np.maximum(2.0 * gamma3, gamma1)
    bF2 = np.tanh(np.power(gamma, 2.0))

    # more model constants
    alfa = alfa_1 * bF1 + (1 - bF1) * alfa_2
    beta = beta_1 * bF1 + (1 - bF1) * beta_2
    sigma_k = sigma_k1 * bF1 + (1 - bF1) * sigma_k2
    sigma_om = sigma_om1 * bF1 + (1 - bF1) * sigma_om2

    # Eddy viscosity
    zeta = np.minimum(1.0 / om, a1 / (strMag * bF2))
    mut = r * k * zeta
    mut = np.minimum(np.maximum(mut, 0.0), 100.0)

    # ---------------------------------------------------------------------
    # om-equation

    # effective viscosity
    mueff = mu + sigma_om * mut
    fs = np.ones(n)

    # diffusion matrix: mueff*d2()/dy2 + dmueff/dy d()/dy
    A = np.einsum('i,ij->ij', mueff, mesh.d2dy2) \
        + np.einsum('i,ij->ij', mesh.ddy @ mueff, mesh.ddy)

    # implicitly treated source term
    np.fill_diagonal(A, A.diagonal() - beta * r * om / fs)

    # Right-hand-side
    b = -alfa[1:-1] * r * strMag[1:-1] * strMag[1:-1] - (1 - bF1[1:-1]) * CDkom[1:-1]

    # Wall boundary conditions
    om[0] = 60.0 * mu / beta_1 / r / wallDist[1] / wallDist[1]
    om[-1] = 60.0 * mu / beta_1 / r / wallDist[-2] / wallDist[-2]

    # Solve
    om = solveEqn(om * fs, A, b, underrelaxOm) / fs
    om[1:-1] = np.maximum(om[1:-1], 1.e-12)

    # ---------------------------------------------------------------------
    # k-equation

    # effective viscosity
    mueff = mu + sigma_k * mut
    fs = np.ones(n)
    fd = np.ones(n)

    # diffusion matrix: mueff*d2()/dy2 + dmueff/dy d()/dy
    A = np.einsum('i,ij->ij', mueff * fd, mesh.d2dy2) \
        + np.einsum('i,ij->ij', (mesh.ddy @ mueff) * fd, mesh.ddy)

    # implicitly treated source term
    np.fill_diagonal(A, A.diagonal() - betaStar * r * om / fs)

    # Right-hand-side
    Pk = np.minimum(mut * strMag * strMag, 20 * betaStar * k * r * om)
    b = -Pk[1:-1]

    # Wall boundary conditions
    k[0] = k[-1] = 0.0

    # Solve
    k = solveEqn(k * fs, A, b, underrelaxK) / fs
    k[1:-1] = np.maximum(k[1:-1], 1.e-12)

    return mut, k, om


def V2F(u, k, e, v2, r, mu, mesh):
    n = mesh.nPoints
    f = np.zeros(n)

    # Model constants
    cmu = 0.22
    sigk = 1.0
    sige = 1.3
    Ce2 = 1.9
    Ct = 6
    Cl = 0.23
    Ceta = 70
    C1 = 1.4
    C2 = 0.3

    # Relaxation factors
    underrelaxK = 0.6
    underrelaxE = 0.6
    underrelaxV2 = 0.6

    # Time and length scales, eddy viscosity and turbulent production
    Tt = np.maximum(k / e, Ct * np.power(mu / (r * e), 0.5))
    Lt = Cl * np.maximum(np.power(k, 1.5) / e, Ceta * np.power(np.power(mu / r, 3) / e, 0.25))
    mut = np.maximum(cmu * r * v2 * Tt, 0.0)
    Pk = mut * np.power(mesh.ddy @ u, 2.0)

    # ---------------------------------------------------------------------
    # f-equation

    # implicitly treated source term
    A = np.einsum('i,ij->ij', Lt * Lt, mesh.d2dy2)
    np.fill_diagonal(A, A.diagonal() - 1.0)

    # Right-hand-side
    vok = v2[1:-1] / k[1:-1]
    rhsf = ((C1 - 6) * vok - 2 / 3 * (C1 - 1)) / Tt[1:-1] - C2 * Pk[1:-1] / (r * k[1:-1])

    # Solve
    f = solveEqn(f, A, rhsf, 1)
    f[1:-1] = np.maximum(f[1:-1], 1.e-12)

    # ---------------------------------------------------------------------
    # v2-equation:

    # effective viscosity
    mueff = mu + mut
    fs = np.ones(n)
    fd = np.ones(n)

    # diffusion matrix: mueff*d2()/dy2 + dmueff/dy d()/dy
    A = np.einsum('i,ij->ij', mueff * fd, mesh.d2dy2) \
        + np.einsum('i,ij->ij', (mesh.ddy @ mueff) * fd, mesh.ddy)

    # implicitly treated source term
    np.fill_diagonal(A, A.diagonal() - 6.0 * r * e / k / fs)

    # Right-hand-side
    b = -r * k[1:-1] * f[1:-1]

    # Wall boundary conditions
    v2[0] = v2[-1] = 0.0

    # Solve
    v2 = solveEqn(v2 * fs, A, b, underrelaxV2) / fs
    v2[1:-1] = np.maximum(v2[1:-1], 1.e-12)

    # ---------------------------------------------------------------------
    # e-equation

    # effective viscosity
    mueff = mu + mut / sige
    fs = np.ones(n)
    fd = np.ones(n)

    # diffusion matrix: mueff*d2()/dy2 + dmueff/dy d()/dy
    A = np.einsum('i,ij->ij', mueff * fd, mesh.d2dy2) \
        + np.einsum('i,ij->ij', (mesh.ddy @ mueff) * fd, mesh.ddy)

    # implicitly treated source term
    np.fill_diagonal(A, A.diagonal() - Ce2 / Tt * r / fs)

    # Right-hand-side
    Ce1 = 1.4 * (1 + 0.045 * np.sqrt(k[1:-1] / v2[1:-1]))
    b = -1 / Tt[1:-1] * Ce1 * Pk[1:-1]

    # Wall boundary conditions
    e[0] = mu * k[1] / r / np.power(mesh.y[1] - mesh.y[0], 2)
    e[-1] = mu * k[-2] / r / np.power(mesh.y[-1] - mesh.y[-2], 2)

    # Solve
    e = solveEqn(e * fs, A, b, underrelaxE) / fs
    e[1:-1] = np.maximum(e[1:-1], 1.e-12)

    # ---------------------------------------------------------------------
    # k-equation

    # effective viscosity
    mueff = mu + mut / sigk
    fs = np.ones(n)
    fd = np.ones(n)

    # diffusion matrix: mueff*d2()/dy2 + dmueff/dy d()/dy
    A = np.einsum('i,ij->ij', mueff * fd, mesh.d2dy2) \
        + np.einsum('i,ij->ij', (mesh.ddy @ mueff) * fd, mesh.ddy)

    # implicitly treated source term
    np.fill_diagonal(A, A.diagonal() - r * e / k / fs)

    # Right-hand-side
    b = -Pk[1:-1]

    # Wall boundary conditions
    k[0] = k[-1] = 0.0

    # Solve
    k = solveEqn(k * fs, A, b, underrelaxK) / fs
    k[1:-1] = np.maximum(k[1:-1], 1.e-12)

    return mut, k, e, v2


def solveEqn(x, A, b, omega):
    import numpy as np

    n = np.size(x)
    x_new = x.copy()

    # add boundary conditions
    b = b - x[0] * A[1:n - 1, 0] - x[n - 1] * A[1:n - 1, n - 1]

    # perform under-relaxation
    b[:] = b[:] + (1 - omega) / omega * A.diagonal()[1:-1] * x[1:-1]
    np.fill_diagonal(A, A.diagonal() / omega)

    # solve linear system
    x_new[1:-1] = np.linalg.solve(A[1:-1, 1:-1], b)
    return x_new


def finiteDiffCoeff(x, k):
    n = np.size(x)
    A = np.ones((n, n))

    for i in range(1, n):
        A[i, :] = pow(x, i) / m.factorial(i)

    b = np.zeros((n, 1))  # b is right hand side,
    b[k] = 1  # so k'th derivative term remains
    sol = np.linalg.solve(A, b)  # solve system for coefficients
    return sol.transpose()


class Mesh:
    def __init__(self, n, H, fact, ns):

        self.nPoints = n

        di = 1.0 / (n - 1)
        i = (np.linspace(0, n - 1, n)) / (n - 1) - 0.5

        # y - coordinate: tanh clustering at the walls
        self.y = H * (1.0 + np.tanh(fact * i) / m.tanh(fact / 2)) / 2.0

        # coordinate transformation: derivative of y with respect to 'i'
        dydi = H * fact / 2.0 / np.tanh(fact / 2) / np.power(np.cosh(fact * i), 2.0)

        # coordinate transformation: second derivative of y with respect to 'i'
        d2ydi2 = -H * np.power(fact, 2.0) / np.tanh(fact / 2) * np.tanh(fact * i) / np.power(np.cosh(fact * i), 2.0)

        # -------------------------------------------------------------
        # coefficient matrix for d()/dy
        # du/dy = 1/(dy/di) * du/di
        ddy = np.zeros((n, n))

        ddy[0, 0:7] = finiteDiffCoeff(np.arange(0, 7), 1)
        ddy[1, 0:7] = finiteDiffCoeff(np.arange(-1, 6), 1)
        ddy[2, 0:7] = finiteDiffCoeff(np.arange(-2, 5), 1)
        ddy[n - 3, n - 7:n] = finiteDiffCoeff(np.arange(-4, 3), 1)
        ddy[n - 2, n - 7:n] = finiteDiffCoeff(np.arange(-5, 2), 1)
        ddy[n - 1, n - 7:n] = finiteDiffCoeff(np.arange(-6, 1), 1)

        for i in range(ns, n - ns):
            ddy[i, :] = 0.0
            ddy[i, i - ns:i + ns + 1] = finiteDiffCoeff(np.arange(-ns, ns + 1), 1)

        # multiply coordinate transformation
        for i in range(0, n):
            ddy[i, :] = ddy[i, :] * 1 / di / dydi[i];

        self.ddy = ddy

        # -------------------------------------------------------------
        # coefficient matrix for d2()/dy2 (second derivative)
        # d2u/dy2 = 1/(dy/di)^2*d2u/di2 - 1/(dy/di)^3*d2y/di2*du/di
        d2dy2 = np.zeros((n, n))

        d2dy2[0, 0:7] = finiteDiffCoeff(np.arange(0, 7), 2)
        d2dy2[1, 0:7] = finiteDiffCoeff(np.arange(-1, 6), 2)
        d2dy2[2, 0:7] = finiteDiffCoeff(np.arange(-2, 5), 2)
        d2dy2[n - 3, n - 7:n] = finiteDiffCoeff(np.arange(-4, 3), 2)
        d2dy2[n - 2, n - 7:n] = finiteDiffCoeff(np.arange(-5, 2), 2)
        d2dy2[n - 1, n - 7:n] = finiteDiffCoeff(np.arange(-6, 1), 2)

        for i in range(ns, n - ns):
            d2dy2[i, :] = 0.0
            d2dy2[i, i - ns:i + ns + 1] = finiteDiffCoeff(np.arange(-ns, ns + 1), 2)

        # multiply coordinate transformation
        for i in range(0, n):
            d2dy2[i, :] = d2dy2[i, :] / np.power(di * dydi[i], 2.0) - ddy[i, :] * d2ydi2[i] / np.power(dydi[i], 2)

        self.d2dy2 = d2dy2

import numpy as np
import math as m
import matplotlib
import matplotlib.pyplot as plt
matplotlib.rcParams.update({'font.size': 14})
matplotlib.rcParams['figure.figsize'] = [16,10]

## ---------------------------
##      User-defined inputs
## ---------------------------

ReTau  = 395
rho = 1.0
mu = 1.0/ReTau

## ---------------------------
##      Generate mesh
## ---------------------------

height = 2      # channel height
n = 100         # number of mesh points
fact = 6        # streching factor and stencil for finite difference discretization
mesh = Mesh(n, height, fact, 1)

## ---------------------------
##      Solve RANS equations
## ---------------------------

turbModel          = "SA"    # turbulence model
u1,mut1,k1,e1,om1 = solveRANS(rho,mu,mesh,turbModel,ReTau)

turbModel          = "SST"    # turbulence model
u2,mut1,k1,e1,om1 = solveRANS(rho,mu,mesh,turbModel,ReTau)

turbModel          = "V2F"    # turbulence model
u3,mut1,k1,e1,om1 = solveRANS(rho,mu,mesh,turbModel,ReTau)

turbModel          = "MK"    # turbulence model
u4,mut1,k1,e1,om1 = solveRANS(rho,mu,mesh,turbModel,ReTau)

turbModel          = "Cess"    # turbulence model
u5,mut1,k1,e1,om1 = solveRANS(rho,mu,mesh,turbModel,ReTau)

ypl1 = mesh.y * ReTau

# analytic results for viscous sub-layer
ypLam = np.linspace(0.2, 13, 100);

# semi-empirical result for log-layer
ypTurb = np.linspace(0.9, 3, 20)
upTurb = 1 / 0.41 * np.log(np.power(10, ypTurb)) + 5.2

fig, axs = plt.subplots(2, 2, figsize=(9, 6), dpi=100)

axs[0, 0].semilogx(ypLam, ypLam, 'k-.')
axs[0, 0].semilogx(np.power(10, ypTurb), upTurb, 'k-.')
axs[0, 0].semilogx(ypl1[1:n // 2], u1[1:n // 2], 'r-', linewidth=3, label='SA')
axs[0, 0].set_ylabel('$u^{+}$', fontsize=14)
axs[0, 0].legend()

axs[0, 1].semilogx(ypLam, ypLam, 'k-.')
axs[0, 1].semilogx(np.power(10, ypTurb), upTurb, 'k-.')
axs[0, 1].semilogx(ypl1[1:n // 2], u2[1:n // 2], 'r-', linewidth=3, label='SST')
axs[0, 1].set_ylabel('$u^{+}$', fontsize=14)
axs[0, 1].legend()

axs[1, 0].semilogx(ypLam, ypLam, 'k-.')
axs[1, 0].semilogx(np.power(10, ypTurb), upTurb, 'k-.')
axs[1, 0].semilogx(ypl1[1:n // 2], u3[1:n // 2], 'r-', linewidth=3, label='V2F')
axs[1, 0].set_xlabel('$y^+$', fontsize=14)
axs[1, 0].set_ylabel('$u^{+}$', fontsize=14)
axs[1, 0].legend()

axs[1, 1].semilogx(ypLam, ypLam, 'k-.')
axs[1, 1].semilogx(np.power(10, ypTurb), upTurb, 'k-.')
axs[1, 1].semilogx(ypl1[1:n // 2], u4[1:n // 2], 'r-', linewidth=3, label='KE')
axs[1, 1].set_xlabel('$y^+$', fontsize=14)
axs[1, 1].set_ylabel('$u^{+}$', fontsize=14)
axs[1, 1].legend()
fig.tight_layout(pad=1.5)

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
from scipy.interpolate import CubicSpline
y_uniform = np.linspace(0, height, 101)
np.save(os.path.join(OUTPUT_FOLDER, f"u_{python_filename}.npy"), CubicSpline(mesh.y, u5)(y_uniform))