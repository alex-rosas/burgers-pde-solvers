"""
analysis/convergence.py
=======================
Convergence study: L2 error vs grid size N for FDM, FEM, spectral.
Run from project root: python analysis/convergence.py
"""

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tqdm import tqdm

from solvers.exact    import u_exact
from solvers.fdm      import solve_fdm
from solvers.fem      import solve_fem
from solvers.spectral import solve_spectral

NU       = 0.05
T        = 1.0
CFL      = 0.4
N_VALUES = [2**p for p in range(4, 11)]
METHODS  = {'FDM': solve_fdm, 'FEM': solve_fem, 'Spectral': solve_spectral}
COLORS   = {'FDM': 'tomato', 'FEM': 'steelblue', 'Spectral': 'seagreen'}

def l2_error(u_num, u_ex, dx):
    return np.sqrt(dx * np.sum((u_num - u_ex)**2))

results = []

for method_name, solver in METHODS.items():
    print(f"\nRunning {method_name}...")
    for N in tqdm(N_VALUES, desc=f"  N values"):
        x    = np.linspace(0, 2*np.pi, N, endpoint=False)
        dx   = 2*np.pi / N
        u0   = np.sin(x)
        u_num, _ = solver(u0, N, T, NU, cfl=CFL)
        u_ex     = u_exact(x, T, NU)
        err      = l2_error(u_num, u_ex, dx)
        results.append({
            'method': method_name,
            'N':      N,
            'error':  err,
            'log_N':  np.log10(N),
            'log_E':  np.log10(err) if err > 0 else -16,
        })

df = pd.DataFrame(results)
(ROOT / 'results').mkdir(exist_ok=True)
df.to_csv(ROOT / 'results' / 'convergence.csv', index=False)
print("\nSaved results/convergence.csv")

print("\nEmpirical convergence slopes:")
for method in ['FDM', 'FEM']:
    sub   = df[df['method'] == method]
    slope = np.polyfit(sub['log_N'], sub['log_E'], 1)[0]
    print(f"  {method}: slope = {slope:.3f}  "
          f"(theory: {-1 if method == 'FDM' else -2:.1f})")

# Plot 1: log-log
fig, ax = plt.subplots(figsize=(7, 5))
for method in ['FDM', 'FEM']:
    sub = df[df['method'] == method]
    ax.loglog(sub['N'], sub['error'],
              'o-', color=COLORS[method], lw=2, ms=6, label=method)
N_ref = np.array([16, 1024])
ax.loglog(N_ref, 0.8*(N_ref/16)**(-1), 'k--', lw=1, label='slope $-1$')
ax.loglog(N_ref, 0.3*(N_ref/16)**(-2), 'k:',  lw=1, label='slope $-2$')
ax.set_xlabel('N (grid points)', fontsize=12)
ax.set_ylabel('$L^2$ error', fontsize=12)
ax.set_title(f'Convergence study | nu={NU}, T={T}', fontsize=12)
ax.legend(fontsize=10)
ax.grid(True, which='both', alpha=0.3)
plt.tight_layout()
plt.savefig(ROOT / 'figures' / 'convergence_loglog.png', dpi=150)
print("Saved figures/convergence_loglog.png")
plt.close()

# Plot 2: semi-log (all three)
fig, ax = plt.subplots(figsize=(7, 5))
for method in ['FDM', 'FEM', 'Spectral']:
    sub = df[df['method'] == method]
    ax.semilogy(sub['N'], sub['error'],
                'o-', color=COLORS[method], lw=2, ms=6, label=method)
ax.set_xlabel('N (grid points)', fontsize=12)
ax.set_ylabel('$L^2$ error (log scale)', fontsize=12)
ax.set_title(f'Convergence: all methods | nu={NU}, T={T}', fontsize=12)
ax.legend(fontsize=10)
ax.grid(True, which='both', alpha=0.3)
plt.tight_layout()
plt.savefig(ROOT / 'figures' / 'convergence_semilog.png', dpi=150)
print("Saved figures/convergence_semilog.png")
plt.close()
