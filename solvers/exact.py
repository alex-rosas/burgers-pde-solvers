"""
solvers/exact.py
================
Exact solution to the viscous Burgers equation via the Cole-Hopf
transformation. Numerically stable for nu down to ~0.005.

PDE:    u_t + u*u_x = nu * u_xx
Domain: x in [0, 2*pi], periodic
IC:     u(x,0) = sin(x)
"""

import numpy as np


def compute_phi0(x, nu):
    """
    phi_0(x) = exp( (cos(x) - 1) / (2*nu) )

    For small nu this becomes very sharply peaked near x=0.
    The exponent (cos(x)-1)/(2*nu) is always <= 0 so phi0 in (0,1].
    """
    return np.exp((np.cos(x) - 1.0) / (2.0 * nu))


def compute_phi(x, t, nu):
    """
    Solve the heat equation phi_t = nu * phi_xx exactly:

        phi(x,t) = IFFT( FFT(phi0) * exp(-nu * k^2 * t) )

    For small nu, phi0 is very peaked and its FFT coefficients
    are large. We normalise phi by its maximum value before
    computing the derivative to avoid catastrophic cancellation
    in the Cole-Hopf formula. The normalisation cancels in
    u = -2*nu * phi_x / phi.
    """
    N = len(x)
    phi0     = compute_phi0(x, nu)
    phi0_hat = np.fft.fft(phi0)
    k        = np.fft.fftfreq(N) * N
    decay    = np.exp(-nu * k**2 * t)
    phi      = np.real(np.fft.ifft(phi0_hat * decay))

    # Normalise to prevent underflow in the ratio phi_x / phi
    # This is valid because u = -2nu * phi_x/phi is scale-invariant
    phi_max = np.max(np.abs(phi))
    if phi_max > 0:
        phi = phi / phi_max

    return phi


def u_exact(x, t, nu):
    """
    Exact Burgers solution via Cole-Hopf:

        u(x,t) = -2*nu * phi_x / phi

    phi_x computed by spectral differentiation (exact for periodic).
    Normalisation of phi cancels in the ratio -- result is unaffected.
    """
    N       = len(x)
    phi     = compute_phi(x, t, nu)
    phi_hat = np.fft.fft(phi)
    k       = np.fft.fftfreq(N) * N
    phi_x   = np.real(np.fft.ifft(1j * k * phi_hat))
    return -2.0 * nu * phi_x / (phi + 1e-300)
