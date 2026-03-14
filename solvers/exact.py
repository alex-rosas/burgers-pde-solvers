"""
solvers/exact.py
================
Exact solution to the viscous Burgers' equation via the Cole-Hopf
transformation. This module is the ground truth against which all
numerical solvers are benchmarked.

PDE:  u_t + u * u_x = nu * u_xx
Domain: x in [0, 2*pi], periodic boundary conditions
Initial condition: u(x, 0) = sin(x)
"""

import numpy as np
from scipy.integrate import cumulative_trapezoid


# ── Step 1: Compute phi_0 from u_0 ───────────────────────────────────────────

def compute_phi0(x, nu):
    """
    Compute the initial condition for phi (the heat equation variable)
    from u_0(x) = sin(x) via the Cole-Hopf relation:

        phi_0(x) = exp( -1/(2*nu) * integral_0^x u_0(s) ds )

    For u_0 = sin(x), the integral is -cos(x) + cos(0) = 1 - cos(x).
    So phi_0(x) = exp( (cos(x) - 1) / (2*nu) )

    Parameters
    ----------
    x  : np.ndarray  - grid points in [0, 2*pi]
    nu : float       - kinematic viscosity (must be > 0)

    Returns
    -------
    phi0 : np.ndarray - values of phi_0 at grid points
    """
    # u_0(s) = sin(s), integral from 0 to x is (1 - cos(x))
    integral = 1.0 - np.cos(x)

    # phi_0 = exp( -integral / (2*nu) )
    phi0 = np.exp(-integral / (2.0 * nu))

    return phi0


# ── Step 2: Evolve phi in time via Fourier series ────────────────────────────

def compute_phi(x, t, nu, N_modes=200):
    """
    Evaluate phi(x, t) by solving the heat equation exactly via Fourier series:

        phi(x, t) = sum_{k=-N_modes}^{N_modes} phi_hat_k(0) * exp(-nu*k^2*t) * exp(i*k*x)

    Steps:
      1. Compute phi_0 on the grid
      2. Take FFT to get Fourier coefficients phi_hat_k(0)
      3. Multiply each coefficient by exp(-nu * k^2 * t)  <- heat equation solution
      4. Take IFFT to get phi(x, t) back in physical space

    Parameters
    ----------
    x       : np.ndarray - grid points in [0, 2*pi]
    t       : float      - time at which to evaluate phi
    nu      : float      - kinematic viscosity
    N_modes : int        - number of Fourier modes to retain (200 is very accurate)

    Returns
    -------
    phi : np.ndarray - values of phi(x, t), real-valued and positive
    """
    N = len(x)

    # Step 2a: compute phi_0
    phi0 = compute_phi0(x, nu)

    # Step 2b: FFT of phi_0
    # np.fft.fft returns coefficients in the order: [k=-N/2, ..., -1, 0, 1, ..., N/2-1]
    phi0_hat = np.fft.fft(phi0)

    # Step 2c: wavenumbers in the same order as np.fft.fft output
    # fftfreq(N) returns [0, 1/N, 2/N, ..., -1/2, ...] so multiply by N to get integers
    k = np.fft.fftfreq(N) * N   # shape: (N,), values: [0, 1, ..., N/2-1, -N/2, ..., -1]

    # Step 2d: multiply each mode by its exact heat equation decay factor
    # Each mode k decays as exp(-nu * k^2 * t) — higher modes decay faster
    decay = np.exp(-nu * k**2 * t)
    phi_hat = phi0_hat * decay

    # Step 2e: inverse FFT to return to physical space
    # np.real() discards tiny imaginary parts from floating point arithmetic
    phi = np.real(np.fft.ifft(phi_hat))

    return phi


# ── Step 3: Recover u from phi via Cole-Hopf formula ─────────────────────────

def u_exact(x, t, nu):
    """
    Compute the exact solution u(x, t) of the viscous Burgers' equation
    using the Cole-Hopf formula:

        u(x, t) = -2 * nu * phi_x / phi

    where phi_x is the spatial derivative of phi, computed via FFT
    differentiation (exact for periodic functions).

    Parameters
    ----------
    x  : np.ndarray - grid points in [0, 2*pi]
    t  : float      - evaluation time
    nu : float      - kinematic viscosity

    Returns
    -------
    u : np.ndarray - exact solution values at grid points
    """
    N = len(x)

    # Step 3a: compute phi(x, t)
    phi = compute_phi(x, t, nu)

    # Step 3b: compute phi_x via spectral differentiation
    # In Fourier space: d/dx corresponds to multiplication by (i*k)
    # So phi_x = IFFT(i * k * FFT(phi))
    phi_hat = np.fft.fft(phi)
    k = np.fft.fftfreq(N) * N
    phi_x = np.real(np.fft.ifft(1j * k * phi_hat))

    # Step 3c: apply Cole-Hopf formula
    # Guard against division by zero (phi > 0 always for nu > 0, but just in case)
    u = -2.0 * nu * phi_x / (phi + 1e-300)

    return u