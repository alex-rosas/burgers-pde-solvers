"""
tests/test_spectral.py
======================
Pytest verification for solvers/spectral.py
Run from project root: pytest tests/
"""

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import numpy as np
from solvers.spectral import solve_spectral
from solvers.exact    import u_exact

N   = 64
NU  = 0.1
T   = 1.0
CFL = 0.5


def test_solver_reaches_final_time():
    x  = np.linspace(0, 2*np.pi, N, endpoint=False)
    u0 = np.sin(x)
    _, t_final = solve_spectral(u0, N, T, NU, cfl=CFL)
    assert abs(t_final - T) < 1e-10


def test_l2_error_within_tolerance():
    x  = np.linspace(0, 2*np.pi, N, endpoint=False)
    u0 = np.sin(x)
    u_num, _ = solve_spectral(u0, N, T, NU, cfl=CFL)
    u_ex = u_exact(x, T, NU)
    dx   = 2*np.pi / N
    err  = np.sqrt(dx * np.sum((u_num - u_ex)**2))
    assert err < 1e-4


def test_solution_finite_at_small_nu():
    """Solver must not blow up or produce NaN near the shock regime."""
    x  = np.linspace(0, 2*np.pi, N, endpoint=False)
    u0 = np.sin(x)
    u_num, _ = solve_spectral(u0, N, T, nu=0.005, cfl=0.3)
    assert np.all(np.isfinite(u_num))
