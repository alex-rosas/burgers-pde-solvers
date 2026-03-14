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
    """
    Assemble the global mass matrix M from linear hat functions.

    M_ij = integral( phi_i * phi_j ) dx

    For uniform linear elements this gives a circulant tridiagonal:
        M = (h/6) * tridiag(1, 4, 1)
    with periodic corner entries M[0,N-1] = M[N-1,0] = h/6.

    The mass matrix appears in the time-derivative term of the
    weak form. It is symmetric positive definite and depends only
    on the mesh geometry -- assembled once before the time loop.

    Parameters
    ----------
    N : int   -- number of nodes (= number of elements, periodic mesh)
    h : float -- element width (2*pi / N)

    Returns
    -------
    M : scipy sparse CSR matrix, shape (N, N)
    """
    diag_m = np.full(N,   4.0) * h / 6.0
    diag_o = np.full(N-1, 1.0) * h / 6.0

    M = diags([diag_o, diag_m, diag_o],
              [-1, 0, 1], shape=(N, N), format='lil')

    # Periodic corners: node 0 neighbours node N-1
    M[0, N-1] = h / 6.0
    M[N-1, 0] = h / 6.0

    return M.tocsr()


def assemble_stiffness(N, h):
    """
    Assemble the global stiffness matrix K from linear hat functions.

    K_ij = integral( phi_i' * phi_j' ) dx

    For uniform linear elements:
        K = (1/h) * tridiag(-1, 2, -1)
    with periodic corner entries K[0,N-1] = K[N-1,0] = -1/h.

    The stiffness matrix appears in the diffusion term of the
    weak form. It encodes the discrete Laplacian and is combined
    with the mass matrix in the Crank-Nicolson system.

    Parameters
    ----------
    N : int   -- number of nodes
    h : float -- element width

    Returns
    -------
    K : scipy sparse CSR matrix, shape (N, N)
    """
    diag_m = np.full(N,    2.0) / h
    diag_o = np.full(N-1, -1.0) / h

    K = diags([diag_o, diag_m, diag_o],
              [-1, 0, 1], shape=(N, N), format='lil')

    # Periodic corners
    K[0, N-1] = -1.0 / h
    K[N-1, 0] = -1.0 / h

    return K.tocsr()


def convection_vector(U, N, h):
    """
    Compute the nonlinear convection vector C(U) by element-wise
    integration using 2-point Gauss-Legendre quadrature.

    C_i(U) = integral( u_h * du_h/dx * phi_i ) dx

    On each element e = [x_j, x_{j+1}]:
      u_h(x)    = U[j]*phi0(x) + U[j+1]*phi1(x)  (linear interpolation)
      du_h/dx   = (U[j+1] - U[j]) / h             (constant on element)
      phi0(s)   = 1 - s/h                          (left hat function)
      phi1(s)   = s/h                              (right hat function)

    2-point Gauss-Legendre on [-1,1]: xi = +/-1/sqrt(3), weights w = 1.
    Mapped to [0,h]: s = h*(1+xi)/2, Jacobian = h/2.

    This quadrature is exact for cubic integrands -- no quadrature
    error for the P1 FEM nonlinear term.

    Parameters
    ----------
    U : np.ndarray -- nodal values, shape (N,)
    N : int        -- number of nodes
    h : float      -- element width

    Returns
    -------
    C : np.ndarray -- convection vector, shape (N,)
    """
    C  = np.zeros(N)
    xi = np.array([-1.0/np.sqrt(3), 1.0/np.sqrt(3)])
    w  = np.array([1.0, 1.0])

    for e in range(N):
        j0 = e
        j1 = (e + 1) % N        # periodic: last element wraps to node 0

        dU   = (U[j1] - U[j0]) / h       # constant derivative on element
        s    = h * (1.0 + xi) / 2.0      # Gauss points mapped to [0, h]
        phi0 = 1.0 - s / h               # left hat function at Gauss pts
        phi1 = s / h                     # right hat function at Gauss pts
        u_g  = U[j0]*phi0 + U[j1]*phi1  # u_h at Gauss points

        # Integrate with Gauss quadrature; jac = h/2 from change of vars
        jac    = h / 2.0
        C[j0] += jac * np.sum(w * u_g * dU * phi0)
        C[j1] += jac * np.sum(w * u_g * dU * phi1)

    return C


def solve_fem(u0, N, T, nu, cfl=0.5):
    """
    Advance the FEM solution from t=0 to t=T using operator splitting:

    Each time step t^n -> t^{n+1}:
      1. Adaptive dt from CFL condition: dt = cfl * h / max|U|
      2. Substep 1 (explicit convection):
            Solve M * delta = C(U^n)
            U* = U^n - dt * delta
      3. Build A_fem = M + (nu*dt/2)*K
             B_fem = M - (nu*dt/2)*K
      4. Substep 2 (implicit diffusion, Crank-Nicolson):
            Solve A_fem * U^{n+1} = B_fem * U*

    M and K are assembled once before the loop since they depend
    only on mesh geometry, not on dt or U.

    Parameters
    ----------
    u0  : np.ndarray -- initial condition at nodes, shape (N,)
    N   : int        -- number of nodes
    T   : float      -- final time
    nu  : float      -- kinematic viscosity
    cfl : float      -- CFL number (must be <= 1 for stability)

    Returns
    -------
    U : np.ndarray -- solution at time T, shape (N,)
    t : float      -- actual final time reached
    """
    h  = 2.0 * np.pi / N
    dx = h

    # Assemble M and K once -- depend only on mesh, not on time or solution
    M = assemble_mass(N, h)
    K = assemble_stiffness(N, h)

    U = u0.copy()
    t = 0.0

    while t < T:
        # Adaptive dt: CFL = max|U| * dt / dx
        max_u = np.max(np.abs(U)) + 1e-8   # 1e-8 avoids division by zero
        dt    = cfl * dx / max_u
        if t + dt > T:
            dt = T - t

        # Substep 1: explicit convection
        # Solve M*delta = C(U^n), then U* = U^n - dt*delta
        C      = convection_vector(U, N, h)
        delta  = spsolve(M, C)
        U_star = U - dt * delta

        # Substep 2: implicit diffusion (Crank-Nicolson)
        # A_fem * U^{n+1} = B_fem * U*
        A_fem = M + (nu * dt / 2.0) * K
        B_fem = M - (nu * dt / 2.0) * K
        U     = spsolve(A_fem, B_fem.dot(U_star))

        t += dt

    return U, t
