"""
tests/test_exact.py
===================
Verification script for solvers/exact.py
Run from anywhere: python tests/test_exact.py
"""

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import numpy as np
import matplotlib.pyplot as plt
from solvers.exact import u_exact, compute_phi

N  = 256
nu = 0.1
x  = np.linspace(0, 2*np.pi, N, endpoint=False)

# -- Check 1: recovery of initial condition
u0  = u_exact(x, 1e-10, nu)
err = np.max(np.abs(u0 - np.sin(x)))
print(f"Check 1 | Max error vs sin(x) at t~0: {err:.2e}  "
      f"{'PASS' if err < 1e-10 else 'FAIL'}")

# -- Check 2: phi stays positive
for t in [0.5, 1.0, 2.0]:
    phi_min = compute_phi(x, t, nu).min()
    print(f"Check 2 | min(phi) at t={t}: {phi_min:.4f}  "
          f"{'PASS' if phi_min > 0 else 'FAIL'}")

# -- Check 3: visual plot
fig, axes = plt.subplots(1, 4, figsize=(14, 3), sharey=True)
for ax, t in zip(axes, [0.0, 0.5, 1.0, 2.0]):
    u = np.sin(x) if t == 0 else u_exact(x, t, nu)
    ax.plot(x, u, color='steelblue', lw=2)
    ax.axhline(0, color='gray', lw=0.5, ls='--')
    ax.set_title(f't = {t}', fontsize=12)
    ax.set_xlabel('x')
axes[0].set_ylabel('u(x,t)')
plt.suptitle(f'Exact Cole-Hopf solution | nu={nu}', fontsize=12)
plt.tight_layout()
plt.savefig(ROOT / 'figures' / 'exact_verification.png', dpi=120)
print(f"Check 3 | Plot saved to {ROOT / 'figures' / 'exact_verification.png'}")
plt.show()