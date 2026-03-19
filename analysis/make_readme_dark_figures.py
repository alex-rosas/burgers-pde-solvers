"""
analysis/make_readme_dark_figures.py
=====================================
Single source of truth for every dark-themed figure that appears in the
README.  All four outputs share the same GitHub-dark palette so the README
looks uniform on dark mode.

Outputs (all saved to figures/)
--------------------------------
  readme_convergence_banner.png   convergence L² error vs N  (Fig 1)
  readme_cfl_dark.png             CN vs Explicit Euler       (Fig 2)
  readme_shock_zoom_dark.png      shock-layer zoom           (Fig 3)
  readme_formulation_l2diff_dark.png  formulation divergence (Fig 4)

Originals produced by the analysis scripts (white background) are kept
unchanged; only the README links point to the dark versions.

Run from project root:
  python analysis/make_readme_dark_figures.py
"""

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from scipy.sparse.linalg import spsolve

from solvers.exact    import u_exact
from solvers.fdm      import solve_fdm, build_cn_matrices, upwind_advection
from solvers.fem      import solve_fem
from solvers.spectral import solve_spectral

# ---------------------------------------------------------------------------
# GitHub-dark palette
# ---------------------------------------------------------------------------
BG     = '#0d1117'
SPINE  = '#30363d'
TICK   = '#8b949e'
LABEL  = '#c9d1d9'
TITLE  = '#e6edf3'
LEG_BG = '#161b22'
GRID_C = '#8b949e'

COLORS = {
    'FDM':      '#e05c5c',
    'FEM':      '#5b9bd5',
    'Spectral': '#4dbb6e',
    'CN':       '#5b9bd5',
    'Euler':    '#e05c5c',
    'Exact':    '#8b949e',
}


def apply_dark(ax, axis='both'):
    """Apply GitHub-dark styling to an Axes instance."""
    ax.set_facecolor(BG)
    ax.tick_params(colors=TICK, labelsize=9)
    for spine in ax.spines.values():
        spine.set_edgecolor(SPINE)
    ax.xaxis.label.set_color(LABEL)
    ax.yaxis.label.set_color(LABEL)
    ax.title.set_color(TITLE)
    ax.grid(alpha=0.15, color=GRID_C, axis=axis)


def dark_legend(ax, **kwargs):
    """Attach a dark-themed legend to an Axes."""
    return ax.legend(
        fontsize=9, framealpha=0.2, labelcolor=TITLE,
        facecolor=LEG_BG, edgecolor=SPINE, **kwargs
    )


def new_dark_fig(*args, **kwargs):
    """Create a Figure with the dark background already set."""
    fig = plt.figure(*args, **kwargs)
    fig.patch.set_facecolor(BG)
    return fig


def save(fig, name):
    out = ROOT / 'figures' / name
    fig.savefig(out, dpi=160, facecolor=fig.get_facecolor())
    print(f"  Saved figures/{name}")
    plt.close(fig)


# ===========================================================================
# Figure 1 — Convergence banner
# ===========================================================================
print("[ 1/4 ] readme_convergence_banner.png")

csv_path = ROOT / 'results' / 'convergence.csv'
df = pd.read_csv(csv_path)

fig = new_dark_fig(figsize=(9, 4.5))
ax  = fig.add_subplot(111)
ax.set_facecolor(BG)

for method, marker in [('FDM', 'o'), ('FEM', 's'), ('Spectral', 'D')]:
    sub = df[df['method'] == method]
    ax.semilogy(sub['N'], sub['error'], f'{marker}-',
                color=COLORS[method], lw=2.2, ms=7, label=method, zorder=3)

# Annotations
ax.annotate('slope $-1$  (1st order)',
            xy=(df[df['method'] == 'FDM']['N'].values[4],
                df[df['method'] == 'FDM']['error'].values[4]),
            xytext=(620, 10e-6), color=COLORS['FDM'], fontsize=9,
            arrowprops=dict(arrowstyle='->', color=COLORS['FDM'], lw=1.2))
ax.annotate('slope $-2$  (2nd order)',
            xy=(df[df['method'] == 'FEM']['N'].values[4],
                df[df['method'] == 'FEM']['error'].values[4]),
            xytext=(500, 10e-9), color=COLORS['FEM'], fontsize=9,
            arrowprops=dict(arrowstyle='->', color=COLORS['FEM'], lw=1.2))
ax.annotate('exponential decay\n(spectral)',
            xy=(df[df['method'] == 'Spectral']['N'].values[2],
                df[df['method'] == 'Spectral']['error'].values[2]),
            xytext=(350, 10e-11), color=COLORS['Spectral'], fontsize=9,
            arrowprops=dict(arrowstyle='->', color=COLORS['Spectral'], lw=1.2))

ax.axhline(1e-14, color='gray', lw=0.8, ls=':', alpha=0.6)
ax.text(18, 3e-14, 'machine precision', color='gray', fontsize=8, alpha=0.8)

ax.set_xlabel('Grid points $N$', fontsize=11)
ax.set_ylabel('$L^2$ error', fontsize=11)
ax.set_title('Convergence: $L^2$ error vs grid size', fontsize=13, pad=12)
ax.set_xlim(12, 1400)
apply_dark(ax)
ax.legend(fontsize=10, framealpha=0.15, labelcolor=TITLE,
          facecolor=LEG_BG, edgecolor=SPINE, loc='upper right')

plt.tight_layout(pad=1.2)
save(fig, 'readme_convergence_banner.png')


# ===========================================================================
# Figure 2 — CFL / stability
# ===========================================================================
print("[ 2/4 ] readme_cfl_dark.png")

N_CFL  = 64
NU_CFL = 0.1
NSTEPS = 100
YLIM   = (-5, 5)
R_LIST = [0.5, 2.0]

x_cfl  = np.linspace(0, 2 * np.pi, N_CFL, endpoint=False)
dx_cfl = 2 * np.pi / N_CFL
u0_cfl = np.sin(x_cfl)


def _diff_explicit(U, dx, nu):
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
        U = (U - dt * upwind_advection(U, dx_cfl)
               + dt * _diff_explicit(U, dx_cfl, NU_CFL))
        if np.max(np.abs(U)) > 1e4:
            return U, True, step + 1
    return U, False, NSTEPS


fig, axes = plt.subplots(2, 2, figsize=(10, 7))
fig.patch.set_facecolor(BG)

for col, r in enumerate(R_LIST):
    dt   = r * dx_cfl**2 / NU_CFL
    T    = NSTEPS * dt
    u_ex = u_exact(x_cfl, T, NU_CFL)

    # Top — Crank-Nicolson
    ax = axes[0][col]
    U, _ = _run_cn(u0_cfl, dt)
    ax.plot(x_cfl, u_ex,               '--', color=COLORS['Exact'], lw=1.5, label='Exact')
    ax.plot(x_cfl, np.clip(U, *YLIM),  '-',  color=COLORS['CN'],    lw=1.5, label='Crank-Nicolson')
    ax.set_title(f'CN  |  r={r}  |  STABLE', fontsize=10)
    ax.set_ylim(YLIM)
    err = np.sqrt(dx_cfl * np.sum((U - u_ex)**2))
    ax.text(0.97, 0.05, f'L2={err:.1e}', transform=ax.transAxes,
            fontsize=8, ha='right', color=TICK)
    apply_dark(ax)
    dark_legend(ax)

    # Bottom — Explicit Euler
    ax = axes[1][col]
    U, blowup, blow_step = _run_explicit(u0_cfl, dt)
    ax.plot(x_cfl, u_ex,               '--', color=COLORS['Exact'], lw=1.5, label='Exact')
    ax.plot(x_cfl, np.clip(U, *YLIM),  '-',  color=COLORS['Euler'], lw=1.5, label='Explicit Euler')
    status = 'UNSTABLE' if blowup else 'STABLE'
    ax.set_title(f'Explicit  |  r={r}  |  {status}', fontsize=10)
    ax.set_ylim(YLIM)
    apply_dark(ax)
    dark_legend(ax)
    if blowup:
        ax.text(0.97, 0.05, f'blew up at step {blow_step}',
                transform=ax.transAxes, fontsize=8, ha='right', color=COLORS['Euler'])
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
save(fig, 'readme_cfl_dark.png')


# ===========================================================================
# Figure 3 — Shock layer zoom
# ===========================================================================
print("[ 3/4 ] readme_shock_zoom_dark.png")

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

ax.plot(x_s[mask], u_fdm[mask],  '-', color=COLORS['FDM'],  lw=2.2, label='FDM')
ax.plot(x_s[mask], u_fem[mask],  '-', color=COLORS['FEM'],  lw=2.2, label='FEM')
ax.plot(x_s[mask], u_spec[mask], '-', color=COLORS['Spectral'], lw=2.2, label='Spectral')
ax.axvline(np.pi, color=TICK, lw=0.9, linestyle=':', alpha=0.6,
           label=r'Shock centre $x=\pi$')

ax.set_xlabel('x', fontsize=12)
ax.set_ylabel('u(x, T)', fontsize=12)
ax.set_title(rf'Shock layer zoom  |  $\nu$={NU_S}, N={N_S}, T={T_S}', fontsize=12)
apply_dark(ax)
dark_legend(ax)
plt.tight_layout()
save(fig, 'readme_shock_zoom_dark.png')


# ===========================================================================
# Figure 4 — Formulation L² divergence
# ===========================================================================
print("[ 4/4 ] readme_formulation_l2diff_dark.png")

NU_VALUES = [0.05, 0.02, 0.01, 0.005]
dx_f = 2 * np.pi / N_S
x_f  = np.linspace(0, 2 * np.pi, N_S, endpoint=False)
u0_f = np.sin(x_f)

l2_diffs = []
for nu in NU_VALUES:
    print(f"  nu={nu} …")
    u_adv, _ = solve_fdm(u0_f, N_S, T_S, nu, cfl=CFL_S, formulation='advective')
    u_con, _ = solve_fdm(u0_f, N_S, T_S, nu, cfl=CFL_S, formulation='conservative')
    l2_diffs.append(np.sqrt(dx_f * np.sum((u_adv - u_con)**2)))

# Blue gradient matching the project palette
BAR_COLORS = ['#c9d9e8', '#7fb3d3', '#2e7bbf', '#1a4d80']

fig, ax = plt.subplots(figsize=(7, 5))
fig.patch.set_facecolor(BG)

bars = ax.bar([str(nu) for nu in NU_VALUES], l2_diffs,
              color=BAR_COLORS, edgecolor=SPINE, linewidth=0.8)

for bar, val in zip(bars, l2_diffs):
    ax.text(bar.get_x() + bar.get_width() / 2,
            bar.get_height() + max(l2_diffs) * 0.015,
            f'{val:.2e}', ha='center', va='bottom', fontsize=9, color=LABEL)

ax.set_xlabel('Viscosity $\\nu$', fontsize=12)
ax.set_ylabel('$\\|u_{adv} - u_{con}\\|_2$', fontsize=12)
ax.set_title(
    f'Formulation divergence vs viscosity  |  N={N_S}, T={T_S}\n'
    'Difference grows as $\\nu \\to 0$: formulation matters near shocks',
    fontsize=11
)
apply_dark(ax, axis='y')
plt.tight_layout()
save(fig, 'readme_formulation_l2diff_dark.png')

print("\nAll four README dark figures generated successfully.")
