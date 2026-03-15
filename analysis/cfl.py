"""
analysis/cfl.py
===============
Demonstrate numerical instability using a fully explicit FDM scheme.
Runs a fixed number of steps so unstable cases have time to blow up.

Run from project root: python analysis/cfl.py
"""

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import numpy as np
import matplotlib.pyplot as plt
from solvers.exact import u_exact
from solvers.fdm   import upwind_advection

N       = 64
NU      = 0.1
NSTEPS  = 100     # fixed number of steps for all cases
YLIM    = (-5, 5)

x  = np.linspace(0, 2*np.pi, N, endpoint=False)
dx = 2*np.pi / N
u0 = np.sin(x)

# Diffusion stability threshold
dt_stable = 0.4 * dx**2 / NU      # r=0.4, safely stable
dt_list   = [
    0.4  * dx**2 / NU,   # r=0.4  stable
    0.5  * dx**2 / NU,   # r=0.5  marginal
    1.0  * dx**2 / NU,   # r=1.0  unstable
    2.0  * dx**2 / NU,   # r=2.0  strongly unstable
]
labels = ['r=0.4 (stable)', 'r=0.5 (marginal)', 'r=1.0 (unstable)', 'r=2.0 (unstable)']

print(f"dx={dx:.4f}  dt_stable={dt_stable:.5f}")
for dt, lab in zip(dt_list, labels):
    print(f"  {lab}: dt={dt:.5f}  T={NSTEPS*dt:.4f}")


def diffusion_explicit(U, dx, nu):
    return nu * (np.roll(U, -1) - 2*U + np.roll(U, 1)) / dx**2


fig, axes = plt.subplots(2, 2, figsize=(10, 7))
axes = axes.flatten()

for ax, (dt, label) in enumerate(zip(dt_list, labels)):
    ax = axes[ax]
    r   = NU * dt / dx**2
    T   = NSTEPS * dt
    U   = u0.copy()
    blowup = False
    blowup_step = 0

    for step in range(NSTEPS):
        adv  = upwind_advection(U, dx)
        diff = diffusion_explicit(U, dx, NU)
        U    = U - dt * adv + dt * diff
        if np.max(np.abs(U)) > 1e4:
            blowup = True
            blowup_step = step + 1
            break

    u_ex   = u_exact(x, T, NU)
    U_plot = np.clip(U, YLIM[0], YLIM[1])

    ax.plot(x, u_ex,   'k--', lw=1.5, label='Exact')
    ax.plot(x, U_plot, color='tomato', lw=1.5, label='Explicit FDM')

    status = 'UNSTABLE' if blowup else 'STABLE'
    ax.set_title(f'{label}  |  {status}', fontsize=10)
    ax.set_xlabel('x')
    ax.set_ylim(YLIM)
    ax.legend(fontsize=9)
    ax.grid(alpha=0.3)

    if not blowup:
        err = np.sqrt(dx * np.sum((U - u_ex)**2))
        ax.text(0.97, 0.05, f'L2={err:.1e}  T={T:.3f}',
                transform=ax.transAxes, fontsize=8,
                ha='right', color='dimgray')
    else:
        ax.text(0.97, 0.05, f'blew up at step {blowup_step}',
                transform=ax.transAxes, fontsize=8,
                ha='right', color='red')

plt.suptitle(
    f'Explicit FDM stability  |  N={N}, nu={NU}, {NSTEPS} steps\n'
    f'r = nu*dt/dx^2  -- stable only if r <= 0.5',
    fontsize=11
)
plt.tight_layout()
plt.savefig(ROOT / 'figures' / 'cfl_blowup.png', dpi=150)
print("Saved figures/cfl_blowup.png")
plt.close()
