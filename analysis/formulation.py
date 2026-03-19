# analysis/formulation.py
# ========================
# Conservative vs advective formulation of the Burgers equation.
#
# The viscous Burgers equation can be written in two equivalent continuous forms:
#
#   Advective (non-conservative):  u_t + u * u_x      = nu * u_xx
#   Conservative:                  u_t + d(u^2/2)/dx  = nu * u_xx
#
# For smooth solutions these are identical.  Near shocks they differ because
# their numerical discretisations propagate the shock at different speeds.
# The conservative form encodes the Rankine-Hugoniot condition directly in the
# flux; the advective form does not, and can misplace the shock by O(dx/nu).
#
# This script runs both formulations across a range of viscosities on the same
# grid (N=256, T=1, CFL=0.4) and produces three figures:
#
#   figures/formulation_profiles.png  -- solution overlays across nu values
#   figures/formulation_zoom.png      -- shock-layer zoom at nu=0.005
#   figures/formulation_l2diff.png    -- L2 difference between formulations vs nu
#
# Run from project root: python analysis/formulation.py

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import numpy as np
import matplotlib.pyplot as plt
from solvers.fdm   import solve_fdm
from solvers.exact import u_exact

# ---------------------------------------------------------------------------
# Parameters
# ---------------------------------------------------------------------------
N         = 256
T         = 1.0
CFL       = 0.4
NU_VALUES = [0.05, 0.02, 0.01, 0.005]

# Exact solution is numerically reliable only for nu >= 0.05 at N=256, T=1
EXACT_RELIABLE = {0.05: True, 0.02: False, 0.01: False, 0.005: False}

COLOR_ADV  = 'tomato'
COLOR_CON  = 'steelblue'

x  = np.linspace(0, 2 * np.pi, N, endpoint=False)
dx = 2 * np.pi / N
u0 = np.sin(x)

# ---------------------------------------------------------------------------
# Run both formulations for every nu
# ---------------------------------------------------------------------------
print("Running solvers...")
solutions = {}
for nu in NU_VALUES:
    print(f"  nu={nu}")
    u_adv, _ = solve_fdm(u0, N, T, nu, cfl=CFL, formulation='advective')
    u_con, _ = solve_fdm(u0, N, T, nu, cfl=CFL, formulation='conservative')
    solutions[nu] = {'advective': u_adv, 'conservative': u_con}

# ---------------------------------------------------------------------------
# Figure 1: solution overlays across viscosities (2x2 grid)
# ---------------------------------------------------------------------------
fig, axes = plt.subplots(2, 2, figsize=(12, 9))
axes = axes.flatten()

for idx, nu in enumerate(NU_VALUES):
    ax       = axes[idx]
    u_adv    = solutions[nu]['advective']
    u_con    = solutions[nu]['conservative']
    reliable = EXACT_RELIABLE[nu]

    if reliable:
        u_ex = u_exact(x, T, nu)
        ax.plot(x, u_ex, 'k--', lw=1.2, alpha=0.6, label='Exact (Cole-Hopf)')

    ax.plot(x, u_adv, '-',  color=COLOR_ADV, lw=2.0, label=r'Advective  $u\,u_x$')
    ax.plot(x, u_con, '--', color=COLOR_CON, lw=2.0, label=r'Conservative  $\partial(u^2/2)/\partial x$')

    # L2 difference between the two formulations
    l2_diff = np.sqrt(dx * np.sum((u_adv - u_con)**2))

    ax.set_xlim(0, 2 * np.pi)
    ax.set_ylim(-1.5, 1.5)
    ax.set_title(f'$\\nu$ = {nu}', fontsize=11)
    ax.set_xlabel('x', fontsize=10)
    ax.set_ylabel('u(x, T)', fontsize=10)
    ax.legend(fontsize=8, loc='upper right')
    ax.grid(True, alpha=0.3)
    ax.text(0.03, 0.05,
            f'$\\|u_{{adv}} - u_{{con}}\\|_2$ = {l2_diff:.2e}',
            transform=ax.transAxes, fontsize=8, color='dimgray')

    if reliable:
        err_adv = np.sqrt(dx * np.sum((u_adv - u_ex)**2))
        err_con = np.sqrt(dx * np.sum((u_con - u_ex)**2))
        ax.text(0.03, 0.14,
                f'L2 vs exact: adv={err_adv:.1e}  con={err_con:.1e}',
                transform=ax.transAxes, fontsize=8, color='dimgray')

plt.suptitle(
    f'Conservative vs advective formulation  |  N={N}, T={T}, CFL={CFL}\n'
    'Formulations agree for smooth solutions; diverge near the shock layer',
    fontsize=12
)
plt.tight_layout()
out = ROOT / 'figures' / 'formulation_profiles.png'
plt.savefig(out, dpi=150, bbox_inches='tight')
print(f"Saved {out.relative_to(ROOT)}")
plt.close()

# ---------------------------------------------------------------------------
# Figure 2: zoom near the shock layer at nu=0.005
# ---------------------------------------------------------------------------
nu_zoom = 0.005
mask    = (x >= 2.5) & (x <= 4.5)

u_adv = solutions[nu_zoom]['advective']
u_con = solutions[nu_zoom]['conservative']

fig, ax = plt.subplots(figsize=(8, 5))
ax.plot(x[mask], u_adv[mask], '-',  color=COLOR_ADV, lw=2.5,
        label=r'Advective  $u\,u_x$')
ax.plot(x[mask], u_con[mask], '--', color=COLOR_CON, lw=2.5,
        label=r'Conservative  $\partial(u^2/2)/\partial x$')

# Fill between the two curves to highlight where profiles diverge
ax.fill_between(x[mask], u_adv[mask], u_con[mask],
                alpha=0.15, color='purple', label='Pointwise difference')

# Annotate the L2 difference in the shock layer
l2_zoom = np.sqrt(dx * np.sum((u_adv[mask] - u_con[mask])**2))
ax.text(0.03, 0.06,
        f'$L^2$ diff (zoom region): {l2_zoom:.2e}',
        transform=ax.transAxes, fontsize=9, color='dimgray')

ax.axvline(np.pi, color='k', lw=1.0, linestyle=':', alpha=0.5, label=r'Shock centre $x=\pi$')

ax.set_xlabel('x', fontsize=12)
ax.set_ylabel('u(x, T)', fontsize=12)
ax.set_title(
    rf'Shock-layer zoom  |  $\nu$={nu_zoom}, N={N}, T={T}' + '\n'
    r'Both formulations place the shock at $x=\pi$ (s=0 by symmetry);'
    '\nprofile shape and smearing width differ',
    fontsize=11
)
ax.legend(fontsize=10)
ax.grid(True, alpha=0.3)
plt.tight_layout()
out = ROOT / 'figures' / 'formulation_zoom.png'
plt.savefig(out, dpi=150)
print(f"Saved {out.relative_to(ROOT)}")
plt.close()

# ---------------------------------------------------------------------------
# Figure 3: L2 difference between formulations as a function of nu
# ---------------------------------------------------------------------------
l2_diffs = []
for nu in NU_VALUES:
    u_adv = solutions[nu]['advective']
    u_con = solutions[nu]['conservative']
    l2_diffs.append(np.sqrt(dx * np.sum((u_adv - u_con)**2)))

fig, ax = plt.subplots(figsize=(7, 5))
bars = ax.bar([str(nu) for nu in NU_VALUES], l2_diffs,
              color=['#d6e4f0', '#7fb3d3', '#2e86c1', '#1a5276'],
              edgecolor='k', linewidth=0.8)

for bar, val in zip(bars, l2_diffs):
    ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.0003,
            f'{val:.2e}', ha='center', va='bottom', fontsize=9)

ax.set_xlabel('Viscosity $\\nu$', fontsize=12)
ax.set_ylabel('$\\|u_{adv} - u_{con}\\|_2$', fontsize=12)
ax.set_title(
    f'Formulation divergence vs viscosity  |  N={N}, T={T}\n'
    'Difference grows as $\\nu \\to 0$: formulation matters near shocks',
    fontsize=11
)
ax.grid(True, axis='y', alpha=0.3)
plt.tight_layout()
out = ROOT / 'figures' / 'formulation_l2diff.png'
plt.savefig(out, dpi=150)
print(f"Saved {out.relative_to(ROOT)}")
plt.close()

# ---------------------------------------------------------------------------
# Console summary
# ---------------------------------------------------------------------------
print("\nFormulation divergence summary:")
print(f"  {'nu':>8}  {'L2(adv - con)':>16}")
print(f"  {'-'*8}  {'-'*16}")
for nu, diff in zip(NU_VALUES, l2_diffs):
    print(f"  {nu:>8.4f}  {diff:>16.4e}")
print()
print("Interpretation:")
print("  - Large nu (smooth regime): difference is small (~O(dx)) -> formulation is irrelevant.")
print("  - Small nu (shock regime):  difference grows -> the two formulations resolve the")
print("    shock layer with different profiles. The conservative form discretises the flux")
print("    directly and satisfies the Rankine-Hugoniot condition at the discrete level;")
print("    for this symmetric IC (sin x, shock at x=pi, speed s=0) the shock position")
print("    is the same, but the profile shape and smearing differ and diverge as nu->0.")
