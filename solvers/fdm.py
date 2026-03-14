"""
solvers/fdm.py
==============
Finite Difference solver for the viscous Burgers equation.

Spatial:  upwind scheme for advection (1st order)
          centred differences for diffusion (2nd order)
Temporal: explicit advection + Crank-Nicolson diffusion (operator splitting)

PDE:    u_t + u*u_x = nu * u_xx
Domain: x in [0, 2*pi], periodic
IC:     u(x,0) = sin(x)
"""

import numpy as np
from scipy.sparse import diags
from scipy.sparse.linalg import spsolve


def build_cn_matrices(N, dx, nu, dt):
    """
    Build the Crank-Nicolson matrices A and B (equation eq:matA, eq:matB).

    A * u^{n+1} = B * u^*

    alpha = nu * dt / (2 * dx^2)

    A: diagonal = 1 + 2*alpha, off-diagonals = -alpha
    B: diagonal = 1 - 2*alpha, off-diagonals = +alpha

    Both are circulant tridiagonal with periodic corner entries.

    Parameters
    ----------
    N  : int   -- number of grid points
    dx : float -- grid spacing
    nu : float -- kinematic viscosity
    dt : float -- time step (used only to compute alpha here)

    Returns
    -------
    A, B : scipy sparse CSR matrices of shape (N, N)
    """
    alpha = nu * dt / (2.0 * dx**2)

    # Main diagonal and off-diagonals
    diag_A  = np.full(N,  1 + 2*alpha)
    diag_B  = np.full(N,  1 - 2*alpha)
    off_A   = np.full(N-1, -alpha)
    off_B   = np.full(N-1,  alpha)

    # Build sparse tridiagonal matrices
    A = diags([off_A, diag_A, off_A], [-1, 0, 1],
              shape=(N, N), format='lil')
    B = diags([off_B, diag_B, off_B], [-1, 0, 1],
              shape=(N, N), format='lil')

    # Periodic boundary conditions: connect last point to first
    A[0, N-1] = -alpha
    A[N-1, 0] = -alpha
    B[0, N-1] =  alpha
    B[N-1, 0] =  alpha

    return A.tocsr(), B.tocsr()


def upwind_advection(u, dx):
    """
    Compute the upwind approximation of u * u_x (equation eq:upwind2).

    For each grid point j:
      if u[j] >= 0: u_x ~ (u[j] - u[j-1]) / dx  (backward difference)
      if u[j] <  0: u_x ~ (u[j+1] - u[j]) / dx  (forward difference)

    Uses np.roll for periodic neighbour access:
      np.roll(u,  1) shifts array right -> u[j-1]
      np.roll(u, -1) shifts array left  -> u[j+1]

    Parameters
    ----------
    u  : np.ndarray -- current solution vector, shape (N,)
    dx : float      -- grid spacing

    Returns
    -------
    f : np.ndarray -- u * u_x at each grid point, shape (N,)
    """
    u_left  = np.roll(u,  1)   # u[j-1] with periodic wrap
    u_right = np.roll(u, -1)   # u[j+1] with periodic wrap

    # Backward difference (used when u >= 0)
    du_backward = (u - u_left)  / dx

    # Forward difference (used when u < 0)
    du_forward  = (u_right - u) / dx

    # Select based on sign of u (upwinding)
    du = np.where(u >= 0, du_backward, du_forward)

    return u * du


def solve_fdm(u0, N, T, nu, cfl=0.5):
    """
    Main FDM time-stepping loop.

    Each step:
      1. Compute adaptive dt from CFL condition
      2. Substep 1: u* = u^n - dt * f(u^n)        (explicit advection)
      3. Rebuild A, B with new dt
      4. Substep 2: A * u^{n+1} = B * u*           (implicit diffusion)

    Parameters
    ----------
    u0  : np.ndarray -- initial condition, shape (N,)
    N   : int        -- number of grid points
    T   : float      -- final time
    nu  : float      -- kinematic viscosity
    cfl : float      -- CFL number (default 0.5, must be <= 1 for stability)

    Returns
    -------
    u   : np.ndarray -- solution at time T, shape (N,)
    t   : float      -- actual final time reached
    """
    dx = 2.0 * np.pi / N
    u  = u0.copy()
    t  = 0.0

    while t < T:
        # -- Compute adaptive time step from CFL condition
        max_u = np.max(np.abs(u)) + 1e-8   # 1e-8 avoids division by zero
        dt    = cfl * dx / max_u

        # Don't overshoot final time
        if t + dt > T:
            dt = T - t

        # -- Build Crank-Nicolson matrices for this dt
        A, B = build_cn_matrices(N, dx, nu, dt)

        # -- Substep 1: explicit advection
        # u* = u^n - dt * f(u^n)
        f     = upwind_advection(u, dx)
        u_star = u - dt * f

        # -- Substep 2: implicit diffusion
        # Solve A * u^{n+1} = B * u*
        rhs = B.dot(u_star)
        u   = spsolve(A, rhs)

        t += dt

    return u, t
