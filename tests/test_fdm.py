"""
tests/test_fdm.py
=================
Pytest verification for solvers/fdm.py
Run from project root: pytest tests/
"""

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import numpy as np
import pytest
from solvers.fdm   import solve_fdm
from solvers.exact import u_exact

N   = 64
NU  = 0.1
T   = 1.0
CFL = 0.5

@pytest.fixture
def grid():
    x  = np.linspace(0, 2*np.pi, N, endpoint=False)
    u0 = np.sin(x)
    return x, u0


def test_solver_reaches_final_time(grid):
    _, u0 = grid
    _, t_final = solve_fdm(u0, N, T, NU, cfl=CFL)
    assert abs(t_final - T) < 1e-10


def test_l2_error_within_tolerance(grid):
    x, u0 = grid
    u_num, _ = solve_fdm(u0, N, T, NU, cfl=CFL)
    u_ex  = u_exact(x, T, NU)
    dx    = 2*np.pi / N
    err   = np.sqrt(dx * np.sum((u_num - u_ex)**2))
    assert err < 0.1


def test_conservative_formulation_runs(grid):
    _, u0 = grid
    u_num, t_final = solve_fdm(u0, N, T, NU, cfl=CFL, formulation='conservative')
    assert abs(t_final - T) < 1e-10
    assert np.all(np.isfinite(u_num))


def test_invalid_formulation_raises(grid):
    _, u0 = grid
    with pytest.raises(ValueError):
        solve_fdm(u0, N, T, NU, formulation='bad_value')
