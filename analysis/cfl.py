"""
analysis/cfl.py
===============
Contrast Crank-Nicolson (unconditionally stable) vs fully explicit
Euler (stable only if r = nu*dt/dx^2 <= 0.5).

Same dt in each column, different scheme in each row.
Top row: Crank-Nicolson -- stays stable at any dt.
Bottom row: Explicit Euler -- blows up when r > 0.5.

No changes to existing solver files. CN stepping is reproduced
inline using build_cn_matrices + spsolve from fdm.py.

Run from project root: python analysis/cfl.py
"""

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import numpy as np
import matplotlib.pyplot as plt
from scipy.sparse.linalg import spsolve
from solvers.exact import u_exact
from solvers.fdm   import build_cn_matrices, upwind_advection

N      = 64
NU     = 0.1
NSTEPS = 100
YLIM   = (-5, 5)

x  = np.linspace(0, 2*np.pi, N, endpoint=False)
dx = 2*np.pi / N
u0 = np.sin(x)

# Two r values: at the explicit stability limit, and 4x beyond it
R_LIST = [0.5, 2.0]


def diffusion_explicit(U, dx, nu):
    """Explicit central difference for nu*u_xx (periodic)."""
    return nu * (np.roll(U, -1) - 2*U + np.roll(U, 1)) / dx**2


def run_cn(u0, dt, nsteps):
    """
    Crank-Nicolson: implicit diffusion + explicit upwind advection.
    Unconditionally stable -- stays clean at any dt.
    """
    A, B = build_cn_matrices(N, dx, NU, dt)
    U    = u0.copy()
    for _ in range(nsteps):
        U_star = U - dt * upwind_advection(U, dx)
        U      = spsolve(A, B.dot(U_star))
        if np.max(np.abs(U)) > 1e4:
            return U, True
    return U, False


def run_explicit(u0, dt, nsteps):
    """
    Fully explicit Euler: explicit diffusion + explicit advection.
    Stable only if r = nu*dt/dx^2 <= 0.5 (Von Neumann condition).
    """
    U = u0.copy()
    for step in range(nsteps):
        adv  = upwind_advection(U, dx)
        diff = diffusion_explicit(U, dx, NU)
        U    = U - dt * adv + dt * diff
        if np.max(np.abs(U)) > 1e4:
            return U, True, step + 1
    return U, False, nsteps


fig, axes = plt.subplots(2, 2, figsize=(10, 7))

for col, r in enumerate(R_LIST):
    dt = r * dx**2 / NU
    T  = NSTEPS * dt

    # Top row: Crank-Nicolson
    ax = axes[0][col]
    U, blowup = run_cn(u0, dt, NSTEPS)
    u_ex = u_exact(x, T, NU)
    ax.plot(x, u_ex, 'k--', lw=1.5, label='Exact')
    ax.plot(x, np.clip(U, YLIM[0], YLIM[1]),
            color='steelblue', lw=1.5, label='Crank-Nicolson')
    ax.set_title(f'CN  |  r={r}  |  STABLE', fontsize=10)
    ax.set_ylim(YLIM); ax.legend(fontsize=9); ax.grid(alpha=0.3)
    err = np.sqrt(dx * np.sum((U - u_ex)**2))
    ax.text(0.97, 0.05, f'L2={err:.1e}',
            transform=ax.transAxes, fontsize=8,
            ha='right', color='dimgray')

    # Bottom row: Explicit Euler
    ax = axes[1][col]
    U, blowup, blow_step = run_explicit(u0, dt, NSTEPS)
    ax.plot(x, u_ex, 'k--', lw=1.5, label='Exact')
    ax.plot(x, np.clip(U, YLIM[0], YLIM[1]),
            color='tomato', lw=1.5, label='Explicit Euler')
    status = 'UNSTABLE' if blowup else 'STABLE'
    ax.set_title(f'Explicit  |  r={r}  |  {status}', fontsize=10)
    ax.set_ylim(YLIM); ax.legend(fontsize=9); ax.grid(alpha=0.3)
    if blowup:
        ax.text(0.97, 0.05, f'blew up at step {blow_step}',
                transform=ax.transAxes, fontsize=8,
                ha='right', color='red')
    else:
        err = np.sqrt(dx * np.sum((U - u_ex)**2))
        ax.text(0.97, 0.05, f'L2={err:.1e}',
                transform=ax.transAxes, fontsize=8,
                ha='right', color='dimgray')

for ax in axes.flatten():
    ax.set_xlabel('x')

plt.suptitle(
    f'Crank-Nicolson vs Explicit Euler  |  N={N}, nu={NU}, {NSTEPS} steps\n'
    r'Same $\Delta t$ per column -- only the scheme changes',
    fontsize=11
)
plt.tight_layout()
plt.savefig(ROOT / 'figures' / 'cfl_blowup.png', dpi=150)
print("Saved figures/cfl_blowup.png")
plt.close()