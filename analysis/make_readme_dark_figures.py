"""
analysis/make_readme_dark_figures.py
=====================================
Generates dark-background (GitHub-dark compatible) versions of two README
figures.  The original light-background figures are kept unchanged.

Outputs
-------
  figures/readme_cfl_dark.png            dark version of cfl_blowup.png
  figures/readme_shock_zoom_dark.png     dark version of shock_zoom.png

Run from project root:
  python analysis/make_readme_dark_figures.py
"""

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import numpy as np
import matplotlib.pyplot as plt
from scipy.sparse.linalg import spsolve

from solvers.exact    import u_exact
from solvers.fdm      import solve_fdm, build_cn_matrices, upwind_advection
from solvers.fem      import solve_fem
from solvers.spectral import solve_spectral

# ---------------------------------------------------------------------------
# GitHub-dark palette (matches readme_convergence_banner.png)
# ---------------------------------------------------------------------------
BG      = '#0d1117'
SPINE   = '#30363d'
TICK    = '#8b949e'
LABEL   = '#c9d1d9'
TITLE   = '#e6edf3'
LEG_BG  = '#161b22'
GRID_C  = '#8b949e'

C_FDM   = '#e05c5c'   # red
C_FEM   = '#5b9bd5'   # blue
C_SPEC  = '#4dbb6e'   # green
C_CN    = '#5b9bd5'
C_EULER = '#e05c5c'
C_EXACT = '#8b949e'   # muted grey


def apply_dark(ax):
    """Apply GitHub-dark styling to an Axes instance."""
    ax.set_facecolor(BG)
    ax.tick_params(colors=TICK, labelsize=9)
    for spine in ax.spines.values():
        spine.set_edgecolor(SPINE)
    ax.xaxis.label.set_color(LABEL)
    ax.yaxis.label.set_color(LABEL)
    ax.title.set_color(TITLE)
    ax.grid(alpha=0.15, color=GRID_C, which='both')


def dark_legend(ax, **kwargs):
    """Attach a legend styled for the dark theme."""
    return ax.legend(
        fontsize=9, framealpha=0.2, labelcolor=TITLE,
        facecolor=LEG_BG, edgecolor=SPINE, **kwargs
    )


# ===========================================================================
# Figure 1 — CFL / stability (dark)
# ===========================================================================
print("Generating readme_cfl_dark.png …")

N_CFL  = 64
NU_CFL = 0.1
NSTEPS = 100
YLIM   = (-5, 5)
R_LIST = [0.5, 2.0]

x_cfl  = np.linspace(0, 2 * np.pi, N_CFL, endpoint=False)
dx_cfl = 2 * np.pi / N_CFL
u0_cfl = np.sin(x_cfl)


def _diffusion_explicit(U, dx, nu):
    return nu * (np.roll(U, -1) - 2 * U + np.roll(U, 1)) / dx**2


def _run_cn(u0, dt):
    A, B = build_cn_matrices(N_CFL, dx_cfl, NU_CFL, dt)
    U = u0.copy()
    for _ in range(NSTEPS):
        U_star = U - dt * upwind_advection(U, dx_cfl)
        U = spsolve(A, B.dot(U_star))
        if np.max(np.abs(U)) > 1e4:
            return U, True
    return U, False


def _run_explicit(u0, dt):
    U = u0.copy()
    for step in range(NSTEPS):
        U = U - dt * upwind_advection(U, dx_cfl) + dt * _diffusion_explicit(U, dx_cfl, NU_CFL)
        if np.max(np.abs(U)) > 1e4:
            return U, True, step + 1
    return U, False, NSTEPS


fig, axes = plt.subplots(2, 2, figsize=(10, 7))
fig.patch.set_facecolor(BG)

for col, r in enumerate(R_LIST):
    dt   = r * dx_cfl**2 / NU_CFL
    T    = NSTEPS * dt
    u_ex = u_exact(x_cfl, T, NU_CFL)

    # Top row — Crank-Nicolson
    ax = axes[0][col]
    U, _ = _run_cn(u0_cfl, dt)
    ax.plot(x_cfl, u_ex, '--', color=C_EXACT, lw=1.5, label='Exact')
    ax.plot(x_cfl, np.clip(U, *YLIM), color=C_CN, lw=1.5, label='Crank-Nicolson')
    ax.set_title(f'CN  |  r={r}  |  STABLE', fontsize=10)
    ax.set_ylim(YLIM)
    err = np.sqrt(dx_cfl * np.sum((U - u_ex)**2))
    ax.text(0.97, 0.05, f'L2={err:.1e}', transform=ax.transAxes,
            fontsize=8, ha='right', color=TICK)
    apply_dark(ax)
    dark_legend(ax)

    # Bottom row — Explicit Euler
    ax = axes[1][col]
    U, blowup, blow_step = _run_explicit(u0_cfl, dt)
    ax.plot(x_cfl, u_ex, '--', color=C_EXACT, lw=1.5, label='Exact')
    ax.plot(x_cfl, np.clip(U, *YLIM), color=C_EULER, lw=1.5, label='Explicit Euler')
    status = 'UNSTABLE' if blowup else 'STABLE'
    ax.set_title(f'Explicit  |  r={r}  |  {status}', fontsize=10)
    ax.set_ylim(YLIM)
    apply_dark(ax)
    dark_legend(ax)
    if blowup:
        ax.text(0.97, 0.05, f'blew up at step {blow_step}',
                transform=ax.transAxes, fontsize=8, ha='right', color=C_EULER)
    else:
        err = np.sqrt(dx_cfl * np.sum((U - u_ex)**2))
        ax.text(0.97, 0.05, f'L2={err:.1e}', transform=ax.transAxes,
                fontsize=8, ha='right', color=TICK)

for ax in axes.flatten():
    ax.set_xlabel('x', color=LABEL)

fig.suptitle(
    f'Crank-Nicolson vs Explicit Euler  |  N={N_CFL}, ν={NU_CFL}, {NSTEPS} steps\n'
    r'Same $\Delta t$ per column — only the scheme changes',
    fontsize=11, color=TITLE
)
plt.tight_layout()
out = ROOT / 'figures' / 'readme_cfl_dark.png'
plt.savefig(out, dpi=150, facecolor=fig.get_facecolor())
print(f"  Saved {out.relative_to(ROOT)}")
plt.close()


# ===========================================================================
# Figure 2 — Shock layer zoom (dark)
# ===========================================================================
print("Generating readme_shock_zoom_dark.png …")

N_S   = 256
T_S   = 1.0
NU_S  = 0.005
CFL_S = 0.4

x_s  = np.linspace(0, 2 * np.pi, N_S, endpoint=False)
u0_s = np.sin(x_s)
mask = (x_s >= 2.5) & (x_s <= 4.5)

print("  Running FDM …")
u_fdm,  _ = solve_fdm(u0_s, N_S, T_S, NU_S, cfl=CFL_S)
print("  Running FEM …")
u_fem,  _ = solve_fem(u0_s, N_S, T_S, NU_S, cfl=CFL_S)
print("  Running Spectral …")
u_spec, _ = solve_spectral(u0_s, N_S, T_S, NU_S, cfl=CFL_S)

fig, ax = plt.subplots(figsize=(8, 5))
fig.patch.set_facecolor(BG)

ax.plot(x_s[mask], u_fdm[mask],  '-', color=C_FDM,  lw=2.2, label='FDM')
ax.plot(x_s[mask], u_fem[mask],  '-', color=C_FEM,  lw=2.2, label='FEM')
ax.plot(x_s[mask], u_spec[mask], '-', color=C_SPEC, lw=2.2, label='Spectral')
ax.axvline(np.pi, color=TICK, lw=0.9, linestyle=':', alpha=0.6,
           label=r'Shock centre $x=\pi$')

ax.set_xlabel('x', fontsize=12)
ax.set_ylabel('u(x, T)', fontsize=12)
ax.set_title(
    rf'Shock layer zoom  |  $\nu$={NU_S}, N={N_S}, T={T_S}',
    fontsize=12
)
apply_dark(ax)
dark_legend(ax)
plt.tight_layout()

out = ROOT / 'figures' / 'readme_shock_zoom_dark.png'
plt.savefig(out, dpi=150, facecolor=fig.get_facecolor())
print(f"  Saved {out.relative_to(ROOT)}")
plt.close()

print("\nDone. README already updated to reference the dark figures.")
