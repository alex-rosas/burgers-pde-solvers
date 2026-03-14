"""
tests/test_fem.py
=================
Verification script for solvers/fem.py
Run: python tests/test_fem.py
"""

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import numpy as np
import matplotlib.pyplot as plt
from solvers.fem   import solve_fem, assemble_mass
from solvers.exact import u_exact

N  = 64
nu = 0.1
T  = 1.0
x  = np.linspace(0, 2*np.pi, N, endpoint=False)
u0 = np.sin(x)

# -- Check 1: solver runs
U_fem, t_final = solve_fem(u0, N, T, nu, cfl=0.5)
print(f"Check 1 | Solver ran to t={t_final:.6f}  "
      f"{'PASS' if abs(t_final - T) < 1e-10 else 'FAIL'}")

# -- Check 2: L2 error better than FDM
u_ex = u_exact(x, T, nu)
dx   = 2*np.pi / N
err  = np.sqrt(dx * np.sum((U_fem - u_ex)**2))
print(f"Check 2 | L2 error: {err:.4e}  "
      f"{'PASS' if err < 1e-2 else 'FAIL'} (expect < 1e-2)")

# -- Check 3: mass matrix SPD
h = 2*np.pi / N
M = assemble_mass(N, h)
min_eig = np.linalg.eigvalsh(M.toarray()).min()
print(f"Check 3 | min eigenvalue of M: {min_eig:.4e}  "
      f"{'PASS' if min_eig > 0 else 'FAIL'}")

# -- Check 4: plot comparison
fig, axes = plt.subplots(1, 2, figsize=(12, 4))
axes[0].plot(x, u_ex,  label='Exact', color='steelblue', lw=2)
axes[0].plot(x, U_fem, label='FEM',   color='tomato', lw=1.5, ls='--')
axes[0].set_title(f'FEM vs Exact | N={N}, nu={nu}, T={T}')
axes[0].set_xlabel('x')
axes[0].legend()

axes[1].plot(x, u_ex - U_fem, color='purple', lw=1.5)
axes[1].axhline(0, color='gray', lw=0.5, ls='--')
axes[1].set_title(f'Pointwise error | L2={err:.2e}')
axes[1].set_xlabel('x')

plt.tight_layout()
plt.savefig(ROOT / 'figures' / 'fem_verification.png', dpi=120)
print(f"Check 4 | Plot saved to figures/fem_verification.png")
