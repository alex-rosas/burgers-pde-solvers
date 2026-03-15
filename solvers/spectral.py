"""
solvers/spectral.py
===================
Fourier pseudospectral solver for the viscous Burgers equation.

Spatial:  Fourier pseudospectral with dealiasing (2/3 rule)
Temporal: RK4 with integrating factor (removes diffusion stiffness)

PDE:    u_t + u*u_x = nu * u_xx
Domain: x in [0, 2*pi], periodic
IC:     u(x,0) = sin(x)
"""

import numpy as np


def wavenumbers(N):
    """
    Return the wavenumber array in FFT order.

    np.fft.fftfreq(N) returns [0, 1/N, ..., (N/2-1)/N, -N/2/N, ..., -1/N].
    Multiplying by N gives integer wavenumbers:
        [0, 1, ..., N/2-1, -N/2, ..., -1]

    This ordering matches the output of np.fft.fft exactly, so
    multiplying u_hat * k gives the correct spectral derivative
    coefficients without any reordering.

    Parameters
    ----------
    N : int -- number of grid points

    Returns
    -------
    k : np.ndarray -- integer wavenumbers, shape (N,)
    """
    return np.fft.fftfreq(N) * N


def dealias(u_hat, N):
    """
    Zero out modes |k| > N/3 to eliminate aliasing (2/3 rule).

    For a quadratic nonlinearity, the product of two functions each
    with modes up to k_max = N/3 has modes up to 2*N/3. Zeroing
    modes beyond N/3 after each FFT prevents aliased high-frequency
    energy from accumulating and destabilising the simulation.

    In FFT order the positive modes are indices [0..N/2-1] and
    negative modes are [N/2..N-1]. The slice [k_max : N-k_max]
    removes all modes with |k| > k_max = N//3.

    Parameters
    ----------
    u_hat : np.ndarray -- Fourier coefficients, shape (N,), modified in place
    N     : int        -- number of grid points

    Returns
    -------
    u_hat : np.ndarray -- dealiased coefficients, shape (N,)
    """
    k_max = N // 3
    u_hat[k_max : N - k_max] = 0.0
    return u_hat


def nonlinear_term(u_hat, k, N):
    """
    Compute the pseudospectral approximation of u*u_x in Fourier space.

    Uses the pseudospectral approach to avoid O(N^2) convolution:
      Step 1: Dealias u_hat (zero |k| > N/3)
      Step 2: u_j     = IFFT(u_hat)         -- physical space values
      Step 3: ux_j    = IFFT(i*k * u_hat)   -- spectral derivative (exact)
      Step 4: prod_j  = u_j * ux_j          -- pointwise product, O(N)
      Step 5: nl_hat  = FFT(prod_j)         -- back to Fourier space
      Step 6: Dealias nl_hat

    Total cost: 3 FFTs at O(N log N), vs O(N^2) for direct convolution.
    np.real() discards floating-point imaginary residuals (~1e-15).

    Parameters
    ----------
    u_hat : np.ndarray -- Fourier coefficients of u, shape (N,)
    k     : np.ndarray -- wavenumber array from wavenumbers(N), shape (N,)
    N     : int        -- number of grid points

    Returns
    -------
    nl_hat : np.ndarray -- Fourier coefficients of u*u_x, shape (N,)
    """
    # Copy before dealiasing to avoid modifying the original
    u_d = dealias(u_hat.copy(), N)

    # Physical space values and spectral derivative
    u_phys  = np.real(np.fft.ifft(u_d))
    ux_phys = np.real(np.fft.ifft(1j * k * u_d))

    # Pointwise product and transform back
    nl_hat = np.fft.fft(u_phys * ux_phys)
    nl_hat = dealias(nl_hat, N)

    return nl_hat


def rhs(v_hat, t, k, N, nu):
    """
    Right-hand side of the integrating-factor ODE:

        dv_hat/dt = exp(+nu*k^2*t) * ( -nonlinear_term(u_hat) )

    where the integrating factor variable v_hat = exp(+nu*k^2*t) * u_hat
    and therefore u_hat = exp(-nu*k^2*t) * v_hat.

    By working with v_hat instead of u_hat, the stiff linear diffusion
    term -nu*k^2*u_hat is absorbed exactly into the exponential factor,
    leaving only the non-stiff nonlinear term. This allows RK4 to use
    time steps controlled only by the CFL condition, not by nu*k^2.

    Parameters
    ----------
    v_hat : np.ndarray -- integrating-factor variable, shape (N,)
    t     : float      -- current time (needed for the IF factors)
    k     : np.ndarray -- wavenumber array, shape (N,)
    N     : int        -- number of grid points
    nu    : float      -- kinematic viscosity

    Returns
    -------
    dv_hat : np.ndarray -- time derivative of v_hat, shape (N,)
    """
    # Only compute IF for active modes (|k| <= N/3)
    # For inactive modes nl=0 after dealiasing, so RHS=0 regardless
    # This avoids exp(nu*k^2*t) overflow for large k and large nu
    k_max  = N // 3
    active = np.abs(k) <= k_max

    IF_inv = np.exp(-nu * k**2 * t)
    u_hat  = IF_inv * v_hat
    nl     = nonlinear_term(u_hat, k, N)

    result           = np.zeros_like(v_hat)
    result[active]   = np.exp(nu * k[active]**2 * t) * (-nl[active])
    return result


def solve_spectral(u0, N, T, nu, cfl=0.5):
    """
    Advance the spectral solution from t=0 to t=T using RK4
    with integrating factor.

    Algorithm:
      1. Transform u0 to Fourier space and dealias
      2. Set v_hat = u_hat (at t=0, exp(0)=1 so v_hat = u_hat)
      3. For each step:
         a. Recover u_phys to estimate max|u| for adaptive dt
         b. Compute dt = cfl * dx / max|u|
         c. Advance v_hat one RK4 step using rhs()
      4. Recover final u in physical space:
            u_hat = exp(-nu*k^2*T) * v_hat
            u     = IFFT(u_hat)

    Parameters
    ----------
    u0  : np.ndarray -- initial condition at grid points, shape (N,)
    N   : int        -- number of grid points
    T   : float      -- final time
    nu  : float      -- kinematic viscosity
    cfl : float      -- CFL number for adaptive time step

    Returns
    -------
    u : np.ndarray -- solution at time T in physical space, shape (N,)
    t : float      -- actual final time reached
    """
    dx    = 2.0 * np.pi / N
    k     = wavenumbers(N)

    # Initial condition in Fourier space, dealiased
    u_hat = dealias(np.fft.fft(u0), N)

    # At t=0: v_hat = exp(0)*u_hat = u_hat
    v_hat = u_hat.copy()
    t     = 0.0

    while t < T:
        # Recover u in physical space to compute adaptive dt
        IF_inv = np.exp(-nu * k**2 * t)
        u_phys = np.real(np.fft.ifft(IF_inv * v_hat))

        # Adaptive dt from CFL condition
        max_u = np.max(np.abs(u_phys)) + 1e-8
        dt    = cfl * dx / max_u
        if t + dt > T:
            dt = T - t

        # RK4 stages on v_hat (4th-order accurate in time)
        k1 = dt * rhs(v_hat,           t,          k, N, nu)
        k2 = dt * rhs(v_hat + 0.5*k1,  t + 0.5*dt, k, N, nu)
        k3 = dt * rhs(v_hat + 0.5*k2,  t + 0.5*dt, k, N, nu)
        k4 = dt * rhs(v_hat + k3,      t + dt,     k, N, nu)

        v_hat = v_hat + (k1 + 2*k2 + 2*k3 + k4) / 6.0
        t    += dt

    # Recover final solution in physical space
    IF_inv = np.exp(-nu * k**2 * t)
    u      = np.real(np.fft.ifft(IF_inv * v_hat))

    return u, t
