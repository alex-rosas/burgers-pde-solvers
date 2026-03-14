"""
solvers/fem.py
==============
Finite Element solver for the viscous Burgers equation.

Spatial:  Galerkin P1 linear elements (2nd order)
Temporal: explicit convection + Crank-Nicolson diffusion

PDE:    u_t + u*u_x = nu * u_xx
Domain: x in [0, 2*pi], periodic
IC:     u(x,0) = sin(x)
"""

import numpy as np
from scipy.sparse import diags
from scipy.sparse.linalg import spsolve


def assemble_mass(N, h):
    diag_m = np.full(N,   4.0) * h / 6.0
    diag_o = np.full(N-1, 1.0) * h / 6.0
    M = diags([diag_o, diag_m, diag_o],
              [-1, 0, 1], shape=(N, N), format='lil')
    M[0, N-1] = h / 6.0
    M[N-1, 0] = h / 6.0
    return M.tocsr()


def assemble_stiffness(N, h):
    diag_m = np.full(N,    2.0) / h
    diag_o = np.full(N-1, -1.0) / h
    K = diags([diag_o, diag_m, diag_o],
              [-1, 0, 1], shape=(N, N), format='lil')
    K[0, N-1] = -1.0 / h
    K[N-1, 0] = -1.0 / h
    return K.tocsr()


def convection_vector(U, N, h):
    C  = np.zeros(N)
    xi = np.array([-1.0/np.sqrt(3), 1.0/np.sqrt(3)])
    w  = np.array([1.0, 1.0])
    for e in range(N):
        j0 = e
        j1 = (e + 1) % N
        dU   = (U[j1] - U[j0]) / h
        s    = h * (1.0 + xi) / 2.0
        phi0 = 1.0 - s / h
        phi1 = s / h
        u_g  = U[j0]*phi0 + U[j1]*phi1
        jac       = h / 2.0
        C[j0]    += jac * np.sum(w * u_g * dU * phi0)
        C[j1]    += jac * np.sum(w * u_g * dU * phi1)
    return C


def solve_fem(u0, N, T, nu, cfl=0.5):
    h  = 2.0 * np.pi / N
    dx = h
    M  = assemble_mass(N, h)
    K  = assemble_stiffness(N, h)
    U  = u0.copy()
    t  = 0.0
    while t < T:
        max_u = np.max(np.abs(U)) + 1e-8
        dt    = cfl * dx / max_u
        if t + dt > T:
            dt = T - t
        C      = convection_vector(U, N, h)
        delta  = spsolve(M, C)
        U_star = U - dt * delta
        A_fem  = M + (nu * dt / 2.0) * K
        B_fem  = M - (nu * dt / 2.0) * K
        U      = spsolve(A_fem, B_fem.dot(U_star))
        t += dt
    return U, t
