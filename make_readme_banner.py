"""
make_readme_banner.py
=====================
Generates figures/readme_convergence_banner.png
for the README. Loads from results/convergence.csv if available,
otherwise uses representative values matching the project results.

Run from project root: python make_readme_banner.py
"""

from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import pandas as pd

ROOT = Path(__file__).resolve().parent

# ---- Load or construct data ---------------------------------------
csv_path = ROOT / 'results' / 'convergence.csv'

df = pd.read_csv(csv_path)
N_fdm      = df[df['method'] == 'FDM']['N'].values
err_fdm    = df[df['method'] == 'FDM']['error'].values
N_fem      = df[df['method'] == 'FEM']['N'].values
err_fem    = df[df['method'] == 'FEM']['error'].values
N_spec     = df[df['method'] == 'Spectral']['N'].values
err_spec   = df[df['method'] == 'Spectral']['error'].values

# ---- Figure -------------------------------------------------------
fig, ax = plt.subplots(figsize=(9, 4.5))
fig.patch.set_facecolor('#0d1117')   # GitHub dark background
ax.set_facecolor('#0d1117')

COLORS = {'FDM': '#e05c5c', 'FEM': '#5b9bd5', 'Spectral': '#4dbb6e'}

ax.semilogy(N_fdm,  err_fdm,  'o-', color=COLORS['FDM'],
            lw=2.2, ms=7, label='FDM', zorder=3)
ax.semilogy(N_fem,  err_fem,  's-', color=COLORS['FEM'],
            lw=2.2, ms=7, label='FEM', zorder=3)
ax.semilogy(N_spec, err_spec, 'D-', color=COLORS['Spectral'],
            lw=2.2, ms=7, label='Spectral', zorder=3)

# Annotation: FDM slope -- below FEM curve, above green annotation
ax.annotate('slope $-1$  (1st order)',
            xy=(N_fdm[4], err_fdm[4]),
            xytext=(620, 10e-6),
            color=COLORS['FDM'], fontsize=9,
            arrowprops=dict(arrowstyle='->', color=COLORS['FDM'], lw=1.2))

# Annotation: FEM slope -- between title and red curve
ax.annotate('slope $-2$  (2nd order)',
            xy=(N_fem[4], err_fem[4]),
            xytext=(500, 10e-9),
            color=COLORS['FEM'], fontsize=9,
            arrowprops=dict(arrowstyle='->', color=COLORS['FEM'], lw=1.2))

# Annotation: Spectral -- lower left, pointing to steep drop
ax.annotate('exponential decay\n(spectral)',
            xy=(N_spec[2], err_spec[2]),
            xytext=(350, 10e-11),
            color=COLORS['Spectral'], fontsize=9,
            arrowprops=dict(arrowstyle='->', color=COLORS['Spectral'], lw=1.2))

# Machine precision line
ax.axhline(1e-14, color='gray', lw=0.8, ls=':', alpha=0.6)
ax.text(18, 3e-14, 'machine precision', color='gray', fontsize=8, alpha=0.8)

# Axes styling
ax.set_xlabel('Grid points $N$', color='#c9d1d9', fontsize=11)
ax.set_ylabel('$L^2$ error', color='#c9d1d9', fontsize=11)
ax.set_title('Convergence: $L^2$ error vs grid size',
             color='#e6edf3', fontsize=13, pad=12)

ax.tick_params(colors='#8b949e', labelsize=9)
for spine in ax.spines.values():
    spine.set_edgecolor('#30363d')
ax.grid(alpha=0.15, color='#8b949e', which='both')
ax.set_xlim(12, 1400)

legend = ax.legend(fontsize=10, framealpha=0.15,
                   labelcolor='#e6edf3',
                   facecolor='#161b22', edgecolor='#30363d',
                   loc='upper right')

plt.tight_layout(pad=1.2)
out = ROOT / 'figures' / 'readme_convergence_banner.png'
plt.savefig(out, dpi=160, facecolor=fig.get_facecolor())
print(f'Saved {out}')
plt.close()