"""
tests/test_exact.py
===================
Pytest verification for solvers/exact.py
Run from project root: pytest tests/
"""

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import numpy as np
from solvers.exact import u_exact, compute_phi

N  = 256
NU = 0.1
x  = np.linspace(0, 2*np.pi, N, endpoint=False)


def test_ic_recovery():
    """Exact solution at t~0 must recover sin(x)."""
    u0  = u_exact(x, 1e-10, NU)
    err = np.max(np.abs(u0 - np.sin(x)))
    assert err < 1e-10


def test_phi_stays_positive():
    """phi must remain strictly positive at all evaluated times."""
    for t in [0.5, 1.0, 2.0]:
        phi_min = compute_phi(x, t, NU).min()
        assert phi_min > 0, f"phi went non-positive at t={t}: min={phi_min}"


def test_solution_finite():
    """Exact solution must be finite for nu in the reliable range."""
    for nu in [0.1, 0.05]:
        u = u_exact(x, 1.0, nu)
        assert np.all(np.isfinite(u)), f"Non-finite values in exact solution at nu={nu}"
