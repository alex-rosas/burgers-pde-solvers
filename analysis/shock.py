"""
analysis/shock.py
=================
Shock resolution study: compare FDM, FEM, spectral near quasi-shocks.
Run from project root: python analysis/shock.py
"""

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from solvers.exact    import u_exact
from solvers.fdm      import solve_fdm
from solvers.fem      import solve_fem
from solvers.spectral import solve_spectral

N         = 256
T         = 1.0
CFL       = 0.4
NU_VALUES = [0.05, 0.02, 0.01, 0.005]

# Exact solution is reliable only for nu >= 0.02 at N=256
# For smaller nu, phi0 requires too many Fourier modes to represent
EXACT_RELIABLE = {0.05: True, 0.02: False, 0.01: False, 0.005: False}

METHODS = {
    'FDM':      (solve_fdm,      'tomato'),
    'FEM':      (solve_fem,      'steelblue'),
    'Spectral': (solve_spectral, 'seagreen'),
}

x  = np.linspace(0, 2*np.pi, N, endpoint=False)
u0 = np.sin(x)

# Run all solvers
solutions = {}
for nu in NU_VALUES:
    print(f"Running nu={nu}...")
    solutions[nu] = {}
    for method_name, (solver, _) in METHODS.items():
        u_num, _ = solver(u0, N, T, nu, cfl=CFL)
        solutions[nu][method_name] = u_num

# Figure 1: 4x3 panel
fig = plt.figure(figsize=(14, 12))
gs  = gridspec.GridSpec(4, 3, hspace=0.45, wspace=0.35)

for i, nu in enumerate(NU_VALUES):
    reliable = EXACT_RELIABLE[nu]
    if reliable:
        u_ex = u_exact(x, T, nu)

    for j, (method_name, (_, color)) in enumerate(METHODS.items()):
        ax    = fig.add_subplot(gs[i, j])
        u_num = solutions[nu][method_name]

        # Only plot exact when it is numerically reliable
        if reliable:
            ax.plot(x, u_ex, 'k--', lw=1.2, alpha=0.7, label='Exact')
            dx  = 2*np.pi / N
            err = np.sqrt(dx * np.sum((u_num - u_ex)**2))
            err_str = f'L2={err:.1e}'
        else:
            err_str = 'exact unreliable'

        ax.plot(x, u_num, '-', lw=1.5, color=color, label=method_name)

        # Always clip y-axis so solver curves are visible
        ax.set_ylim(-1.5, 1.5)
        ax.set_xlim(0, 2*np.pi)
        ax.set_title(f'{method_name} | nu={nu}', fontsize=9)
        ax.set_xlabel('x', fontsize=8)
        ax.tick_params(labelsize=7)
        ax.text(0.97, 0.05, err_str,
                transform=ax.transAxes, fontsize=7,
                ha='right', color='dimgray')
        if j == 0:
            ax.set_ylabel('u(x, T)', fontsize=8)

plt.suptitle(f'Shock resolution: N={N}, T={T}', fontsize=13, y=1.01)
plt.savefig(ROOT / 'figures' / 'shock_panel.png', dpi=150, bbox_inches='tight')
print("Saved figures/shock_panel.png")
plt.close()

# Figure 2: zoom at nu=0.005 -- three methods side by side
nu_zoom = 0.005
mask    = (x >= 2.5) & (x <= 4.5)

fig, ax = plt.subplots(figsize=(8, 5))
for method_name, (_, color) in METHODS.items():
    u_num = solutions[nu_zoom][method_name]
    ax.plot(x[mask], u_num[mask], '-', lw=2, color=color, label=method_name)

ax.set_xlabel('x', fontsize=12)
ax.set_ylabel('u(x, T)', fontsize=12)
ax.set_title(f'Shock layer zoom | nu={nu_zoom}, N={N}, T={T}', fontsize=12)
ax.legend(fontsize=10)
ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig(ROOT / 'figures' / 'shock_zoom.png', dpi=150)
print("Saved figures/shock_zoom.png")
plt.close()
