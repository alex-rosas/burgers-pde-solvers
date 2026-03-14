"""
analysis/performance.py
=======================
Performance study: runtime and memory scaling vs N for all solvers.
Run from project root: python analysis/performance.py
"""

import sys, time, tracemalloc
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tqdm import tqdm

from solvers.fdm      import solve_fdm
from solvers.fem      import solve_fem
from solvers.spectral import solve_spectral

NU       = 0.05
T        = 1.0
CFL      = 0.4
N_REPS   = 3
N_VALUES = [64, 128, 256, 512, 1024, 2048]
METHODS  = {'FDM': solve_fdm, 'FEM': solve_fem, 'Spectral': solve_spectral}
COLORS   = {'FDM': 'tomato',  'FEM': 'steelblue', 'Spectral': 'seagreen'}

results = []

for method_name, solver in METHODS.items():
    print(f"\nBenchmarking {method_name}...")
    for N in tqdm(N_VALUES, desc="  N values"):
        x  = np.linspace(0, 2*np.pi, N, endpoint=False)
        u0 = np.sin(x)

        times = []
        for _ in range(N_REPS):
            t0 = time.perf_counter()
            solver(u0, N, T, NU, cfl=CFL)
            times.append(time.perf_counter() - t0)
        runtime = min(times)

        tracemalloc.start()
        solver(u0, N, T, NU, cfl=CFL)
        _, peak_bytes = tracemalloc.get_traced_memory()
        tracemalloc.stop()

        results.append({
            'method':    method_name,
            'N':         N,
            'runtime':   runtime,
            'memory_mb': peak_bytes / 1024**2,
            'log_N':     np.log10(N),
            'log_t':     np.log10(runtime),
            'log_m':     np.log10(peak_bytes / 1024**2),
        })

df = pd.DataFrame(results)
(ROOT / 'results').mkdir(exist_ok=True)
df.to_csv(ROOT / 'results' / 'performance.csv', index=False)
print("\nSaved results/performance.csv")

print("\nRuntime scaling slopes:")
for method in ['FDM', 'FEM', 'Spectral']:
    sub   = df[df['method'] == method]
    slope = np.polyfit(sub['log_N'], sub['log_t'], 1)[0]
    print(f"  {method}: {slope:.3f}")

print("\nMemory scaling slopes:")
for method in ['FDM', 'FEM', 'Spectral']:
    sub   = df[df['method'] == method]
    slope = np.polyfit(sub['log_N'], sub['log_m'], 1)[0]
    print(f"  {method}: {slope:.3f}  (expect ~1.0)")

fig, ax = plt.subplots(figsize=(7, 5))
for method in ['FDM', 'FEM', 'Spectral']:
    sub = df[df['method'] == method]
    ax.loglog(sub['N'], sub['runtime'],
              'o-', color=COLORS[method], lw=2, ms=6, label=method)
N_ref = np.array([64, 2048])
t_ref = df[df['method'] == 'FDM']['runtime'].values[0]
ax.loglog(N_ref, t_ref*(N_ref/64)**1.0, 'k--', lw=1, label='$O(N)$')
ax.loglog(N_ref, t_ref*(N_ref/64)*np.log2(N_ref/64+1), 'k:', lw=1, label='$O(N\log N)$')
ax.set_xlabel('N (grid points)', fontsize=12)
ax.set_ylabel('Runtime (seconds)', fontsize=12)
ax.set_title(f'Runtime scaling | nu={NU}, T={T}', fontsize=12)
ax.legend(fontsize=10)
ax.grid(True, which='both', alpha=0.3)
plt.tight_layout()
plt.savefig(ROOT / 'figures' / 'performance_runtime.png', dpi=150)
print("Saved figures/performance_runtime.png")
plt.close()

fig, ax = plt.subplots(figsize=(7, 5))
for method in ['FDM', 'FEM', 'Spectral']:
    sub = df[df['method'] == method]
    ax.loglog(sub['N'], sub['memory_mb'],
              'o-', color=COLORS[method], lw=2, ms=6, label=method)
ax.set_xlabel('N (grid points)', fontsize=12)
ax.set_ylabel('Peak memory (MB)', fontsize=12)
ax.set_title(f'Memory scaling | nu={NU}, T={T}', fontsize=12)
ax.legend(fontsize=10)
ax.grid(True, which='both', alpha=0.3)
plt.tight_layout()
plt.savefig(ROOT / 'figures' / 'performance_memory.png', dpi=150)
print("Saved figures/performance_memory.png")
plt.close()
