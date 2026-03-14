"""
tests/test_fdm.py
=================
Verification script for solvers/fdm.py
Run from project root: python tests/test_fdm.py
"""

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import numpy as np
import matplotlib.pyplot as plt
from solvers.fdm   import solve_fdm
from solvers.exact import u_exact

N   = 64
nu  = 0.1
T   = 1.0
x   = np.linspace(0, 2*np.pi, N, endpoint=False)
u0  = np.sin(x)

# -- Check 1: solution runs without errors
u_fdm, t_final = solve_fdm(u0, N, T, nu, cfl=0.5)
print(f"Check 1 | Solver ran to t={t_final:.4f}  "
      f"{'PASS' if abs(t_final - T) < 1e-10 else 'FAIL'}")

# -- Check 2: L2 error against exact solution
u_ex  = u_exact(x, T, nu)
dx    = 2*np.pi / N
err   = np.sqrt(dx * np.sum((u_fdm - u_ex)**2))
print(f"Check 2 | L2 error at T={T}: {err:.4e}  "
      f"{'PASS' if err < 0.1 else 'FAIL'}")

# -- Check 3: CFL > 1 blows up
print("Check 3 | Testing CFL=1.5 blow-up...")
try:
    u_bad, _ = solve_fdm(u0, N, T=0.5, nu=nu, cfl=1.5)
    blowup   = np.max(np.abs(u_bad)) > 1e4
    print(f"          Max|u| = {np.max(np.abs(u_bad)):.2e}  "
          f"{'PASS (blew up as expected)' if blowup else 'NOTE: did not blow up at N=64'}")
except Exception as e:
    print(f"          Exception raised: {e}  PASS")

# -- Check 4: visual comparison with exact solution
fig, axes = plt.subplots(1, 2, figsize=(12, 4))

axes[0].plot(x, u_ex,  label='Exact',   color='steelblue', lw=2)
axes[0].plot(x, u_fdm, label='FDM',     color='tomato',
             lw=1.5, ls='--')
axes[0].set_title(f'FDM vs Exact | N={N}, nu={nu}, T={T}')
axes[0].set_xlabel('x')
axes[0].set_ylabel('u(x,T)')
axes[0].legend()

axes[1].plot(x, u_ex - u_fdm, color='purple', lw=1.5)
axes[1].axhline(0, color='gray', lw=0.5, ls='--')
axes[1].set_title(f'Pointwise error | L2={err:.2e}')
axes[1].set_xlabel('x')
axes[1].set_ylabel('exact - fdm')

plt.tight_layout()
plt.savefig(ROOT / 'figures' / 'fdm_verification.png', dpi=120)
print(f"Check 4 | Plot saved to figures/fdm_verification.png")
