"""
tests/test_spectral.py
======================
Verification script for solvers/spectral.py
Run: python tests/test_spectral.py
"""

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import numpy as np
import matplotlib.pyplot as plt
from solvers.spectral import solve_spectral
from solvers.exact    import u_exact

N  = 64
nu = 0.1
T  = 1.0
x  = np.linspace(0, 2*np.pi, N, endpoint=False)
u0 = np.sin(x)

# -- Check 1: solver runs
u_spec, t_final = solve_spectral(u0, N, T, nu, cfl=0.5)
print(f"Check 1 | Solver ran to t={t_final:.6f}  "
      f"{'PASS' if abs(t_final - T) < 1e-10 else 'FAIL'}")

# -- Check 2: L2 error much smaller than FEM
u_ex = u_exact(x, T, nu)
dx   = 2*np.pi / N
err  = np.sqrt(dx * np.sum((u_spec - u_ex)**2))
print(f"Check 2 | L2 error: {err:.4e}  "
      f"{'PASS' if err < 1e-4 else 'FAIL'} (expect < 1e-4)")

# -- Check 3: Gibbs phenomenon at small nu
nu_small     = 0.005
u_spec_shock, _ = solve_spectral(u0, N, T, nu_small, cfl=0.3)
gibbs        = np.max(np.abs(u_spec_shock)) > 1.2
print(f"Check 3 | Gibbs oscillations at nu={nu_small}: "
      f"max|u|={np.max(np.abs(u_spec_shock)):.3f}  "
      f"{'PASS (oscillations present)' if gibbs else 'NOTE: no oscillations yet'}")

# -- Check 4: plot
fig, axes = plt.subplots(1, 2, figsize=(12, 4))
axes[0].plot(x, u_ex,   label='Exact',    color='steelblue', lw=2)
axes[0].plot(x, u_spec, label='Spectral', color='tomato', lw=1.5, ls='--')
axes[0].set_title(f'Spectral vs Exact | N={N}, nu={nu}, T={T}')
axes[0].set_xlabel('x')
axes[0].legend()

axes[1].plot(x, u_ex - u_spec, color='purple', lw=1.5)
axes[1].axhline(0, color='gray', lw=0.5, ls='--')
axes[1].set_title(f'Pointwise error | L2={err:.2e}')
axes[1].set_xlabel('x')

plt.tight_layout()
plt.savefig(ROOT / 'figures' / 'spectral_verification.png', dpi=120)
print(f"Check 4 | Plot saved to figures/spectral_verification.png")
