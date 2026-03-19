"""
app.py
======
Streamlit interactive app — Three Roads to Burgers.
Run: streamlit run app.py

Six tabs:
  1. Solver Explorer  -- live solver with sliders
  2. Convergence      -- plots from results/convergence.csv
  3. Performance      -- plots from results/performance.csv
  4. Shock Resolution -- panel figures + live comparison
  5. Formulation      -- conservative vs advective FDM
  6. About            -- project description and links
"""

from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import streamlit as st

ROOT = Path(__file__).resolve().parent

from solvers.exact    import u_exact
from solvers.fdm      import solve_fdm
from solvers.fem      import solve_fem
from solvers.spectral import solve_spectral

# ---- Page config --------------------------------------------------
st.set_page_config(
    page_title="Three Roads to Burgers",
    page_icon=":chart_with_upwards_trend:",
    layout="wide",
)

st.title("Three Roads to Burgers: FDM, FEM, and Spectral Methods Under Smooth and Shock Regimes")
st.caption(
    "A controlled numerical study of how discretisation choice, conservation structure, and "
    "solution regularity shape accuracy and computational cost in nonlinear PDE solvers.  "
    "[[GitHub]](https://github.com/alex-rosas/burgers-pde-solvers)"
)

SOLVERS = {"FDM": solve_fdm, "FEM": solve_fem, "Spectral": solve_spectral}
COLORS  = {"FDM": "tomato",  "FEM": "steelblue", "Spectral": "seagreen"}

tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs(
    ["Solver Explorer", "Convergence", "Performance", "Shock Resolution", "Formulation", "About"]
)


# ==================================================================
# Tab 1 -- Solver Explorer
# ==================================================================
with tab1:
    st.header("Interactive Solver Explorer")
    st.markdown(
        r"Adjust parameters and run a solver live. "
        r"Watch what happens as $\nu \to 0$ (shock forms) "
        r"or as $N$ increases (solution converges)."
    )

    col1, col2 = st.columns([1, 2])

    with col1:
        method = st.selectbox("Method", list(SOLVERS.keys()))
        N      = st.select_slider("Grid points N",
                    options=[16, 32, 64, 128, 256, 512], value=64)
        nu     = st.select_slider(r"Viscosity nu",
                    options=[0.1, 0.05, 0.02, 0.01, 0.005], value=0.1)
        T      = st.slider("Final time T", 0.1, 2.0, 1.0, step=0.1)
        cfl    = st.slider("CFL number",   0.1, 0.9, 0.4, step=0.1)
        run    = st.button("Run solver", type="primary")

    with col2:
        if run:
            x  = np.linspace(0, 2*np.pi, N, endpoint=False)
            u0 = np.sin(x)

            with st.spinner(f"Running {method}..."):
                u_num, _ = SOLVERS[method](u0, N, T, nu, cfl=cfl)
                u_ex     = u_exact(x, T, nu)
                dx       = 2*np.pi / N
                err      = np.sqrt(dx * np.sum((u_num - u_ex)**2))

            fig, axes = plt.subplots(1, 2, figsize=(10, 4))

            axes[0].plot(x, u_ex,  'k--', lw=2,  label='Exact')
            axes[0].plot(x, u_num, color=COLORS[method], lw=1.8, label=method)
            axes[0].set_title(f"{method} vs Exact  |  L2 = {err:.2e}")
            axes[0].set_xlabel("x")
            axes[0].legend()
            axes[0].grid(alpha=0.3)

            axes[1].plot(x, u_num - u_ex, color=COLORS[method], lw=1.5)
            axes[1].axhline(0, color='gray', lw=0.5, ls='--')
            axes[1].set_title("Pointwise error")
            axes[1].set_xlabel("x")
            axes[1].grid(alpha=0.3)

            plt.tight_layout()
            st.pyplot(fig)
            plt.close()
            st.metric("L2 error", f"{err:.3e}")
        else:
            st.info("Set parameters and click **Run solver**.")


# ==================================================================
# Tab 2 -- Convergence
# ==================================================================
with tab2:
    st.header("Convergence Study")
    st.markdown(
        r"$L^2$ error vs grid size $N$ for each method. "
        r"FDM converges at $O(1/N)$, FEM at $O(1/N^2)$, "
        r"Spectral **exponentially** for smooth solutions."
    )

    csv_path = ROOT / "results" / "convergence.csv"
    if csv_path.exists():
        df      = pd.read_csv(csv_path)
        methods = st.multiselect(
            "Show methods", ["FDM", "FEM", "Spectral"],
            default=["FDM", "FEM", "Spectral"]
        )

        col_a, col_b = st.columns(2)

        with col_a:
            st.subheader("Log-log plot")
            fig, ax = plt.subplots(figsize=(6, 4))
            for m in methods:
                sub = df[df['method'] == m]
                ax.loglog(sub['N'], sub['error'],
                          'o-', color=COLORS[m], lw=2, ms=6, label=m)
            N_ref = np.array([16, 1024])
            ax.loglog(N_ref, 0.8*(N_ref/16)**(-1), 'k--', lw=1, label='slope $-1$')
            ax.loglog(N_ref, 0.3*(N_ref/16)**(-2), 'k:',  lw=1, label='slope $-2$')
            ax.set_xlabel("N")
            ax.set_ylabel("$L^2$ error")
            ax.set_title("Log-log convergence")
            ax.legend()
            ax.grid(alpha=0.3)
            plt.tight_layout()
            st.pyplot(fig)
            plt.close()

        with col_b:
            st.subheader("Semi-log plot")
            fig, ax = plt.subplots(figsize=(6, 4))
            for m in methods:
                sub = df[df['method'] == m]
                ax.semilogy(sub['N'], sub['error'],
                            'o-', color=COLORS[m], lw=2, ms=6, label=m)
            ax.set_xlabel("N")
            ax.set_ylabel("$L^2$ error (log scale)")
            ax.set_title("Exponential decay of spectral method")
            ax.legend()
            ax.grid(alpha=0.3)
            plt.tight_layout()
            st.pyplot(fig)
            plt.close()

        # Slopes table
        st.subheader("Measured convergence slopes")
        rows = []
        for m in ["FDM", "FEM"]:
            sub   = df[df['method'] == m]
            slope = np.polyfit(sub['log_N'], sub['log_E'], 1)[0]
            rows.append({
                "Method":         m,
                "Measured slope": f"{slope:.2f}",
                "Theory":         "-1.0" if m == "FDM" else "-2.0",
            })
        st.table(pd.DataFrame(rows))
    else:
        st.warning("Run `python analysis/convergence.py` first to generate the CSV.")


# ==================================================================
# Tab 3 -- Performance
# ==================================================================
with tab3:
    st.header("Performance Study")
    st.markdown(
        r"Wall-clock runtime and peak memory vs $N$. "
        r"Memory is $O(N)$ for all methods. "
        r"Runtime reflects Python loop overhead in FEM."
    )

    perf_path = ROOT / "results" / "performance.csv"
    if perf_path.exists():
        df = pd.read_csv(perf_path)
        col_a, col_b = st.columns(2)

        with col_a:
            st.subheader("Runtime scaling")
            fig, ax = plt.subplots(figsize=(6, 4))
            for m in ["FDM", "FEM", "Spectral"]:
                sub = df[df['method'] == m]
                ax.loglog(sub['N'], sub['runtime'],
                          'o-', color=COLORS[m], lw=2, ms=6, label=m)
            N_ref = np.array([64, 2048])
            t_ref = df[df['method'] == 'FDM']['runtime'].values[0]
            ax.loglog(N_ref, t_ref*(N_ref/64)**1.0,
                      'k--', lw=1, label='$O(N)$')
            ax.loglog(N_ref, t_ref*(N_ref/64)*np.log2(N_ref/64+1),
                      'k:',  lw=1, label='$O(N\log N)$')
            ax.set_xlabel("N")
            ax.set_ylabel("Runtime (s)")
            ax.set_title("Runtime scaling")
            ax.legend()
            ax.grid(alpha=0.3)
            plt.tight_layout()
            st.pyplot(fig)
            plt.close()

        with col_b:
            st.subheader("Memory scaling")
            fig, ax = plt.subplots(figsize=(6, 4))
            for m in ["FDM", "FEM", "Spectral"]:
                sub = df[df['method'] == m]
                ax.loglog(sub['N'], sub['memory_mb'],
                          'o-', color=COLORS[m], lw=2, ms=6, label=m)
            ax.set_xlabel("N")
            ax.set_ylabel("Peak memory (MB)")
            ax.set_title("Memory scaling")
            ax.legend()
            ax.grid(alpha=0.3)
            plt.tight_layout()
            st.pyplot(fig)
            plt.close()
    else:
        st.warning("Run `python analysis/performance.py` first to generate the CSV.")


# ==================================================================
# Tab 4 -- Shock Resolution
# ==================================================================
with tab4:
    st.header("Shock Resolution Study")
    st.markdown(
        r"As $\nu \to 0$, a near-shock forms. Each method responds differently: "
        r"FDM **smears** it (artificial viscosity), "
        r"FEM **oscillates** (no upwinding), "
        r"Spectral **rings** (Gibbs phenomenon)."
    )

    panel_path = ROOT / "figures" / "shock_panel.png"
    zoom_path  = ROOT / "figures" / "shock_zoom.png"

    if panel_path.exists():
        st.image(str(panel_path),
                 caption="4x3 panel: rows = viscosity (decreasing), columns = method",
                 use_container_width=True)
    if zoom_path.exists():
        st.image(str(zoom_path),
                 caption=r"Shock layer zoom at nu=0.005: FDM smears, FEM and Spectral are close",
                 use_container_width=True)

    st.subheader("Live comparison")
    st.markdown(r"Pick $\nu$ and $N$ and run all three solvers simultaneously.")

    nu_shock = st.select_slider(
        r"Viscosity nu",
        options=[0.05, 0.02, 0.01, 0.005], value=0.02, key="shock_nu"
    )
    N_shock = st.select_slider(
        "Grid points N",
        options=[64, 128, 256], value=128, key="shock_N"
    )

    if st.button("Compare all methods", type="primary"):
        x  = np.linspace(0, 2*np.pi, N_shock, endpoint=False)
        u0 = np.sin(x)
        fig, axes = plt.subplots(1, 3, figsize=(13, 4))
        for ax, (method, solver) in zip(axes, SOLVERS.items()):
            with st.spinner(f"Running {method}..."):
                u_num, _ = solver(u0, N_shock, 1.0, nu_shock, cfl=0.4)
            ax.plot(x, u_num, color=COLORS[method], lw=2, label=method)
            ax.set_title(f"{method}  |  nu={nu_shock}")
            ax.set_xlabel("x")
            ax.set_ylim(-1.5, 1.5)
            ax.grid(alpha=0.3)
        plt.tight_layout()
        st.pyplot(fig)
        plt.close()


# ==================================================================
# Tab 5 -- Formulation
# ==================================================================
with tab5:
    st.header("Conservative vs Advective Formulation")
    st.markdown(
        r"""
The viscous Burgers equation admits two equivalent continuous forms:

$$\text{Advective:} \quad u_t + u\,u_x = \nu\,u_{xx}$$
$$\text{Conservative:} \quad u_t + \frac{\partial}{\partial x}\!\left(\frac{u^2}{2}\right) = \nu\,u_{xx}$$

For smooth solutions these are numerically interchangeable (both $O(\Delta x)$ first-order accurate).
Near shocks the **conservative form** inherits the Rankine-Hugoniot shock-speed condition from the
underlying flux structure; the **advective form** does not, and resolves the shock layer differently.
        """
    )

    col_img1, col_img2 = st.columns([3, 2])
    with col_img1:
        prof_path = ROOT / "figures" / "formulation_profiles.png"
        if prof_path.exists():
            st.image(str(prof_path),
                     caption="Solution overlays across viscosities — agreement for smooth, divergence near shock",
                     use_container_width=True)
    with col_img2:
        diff_path = ROOT / "figures" / "formulation_l2diff.png"
        if diff_path.exists():
            st.image(str(diff_path),
                     caption="L2 difference between formulations grows monotonically as ν → 0",
                     use_container_width=True)

    zoom_path = ROOT / "figures" / "formulation_zoom.png"
    if zoom_path.exists():
        st.image(str(zoom_path),
                 caption="Shock-layer zoom at ν=0.005 — shaded region is the pointwise difference between formulations",
                 use_container_width=True)

    st.subheader("Live comparison")
    st.markdown(r"Pick $\nu$ and see both formulations side by side in real time.")

    nu_form = st.select_slider(
        r"Viscosity ν", options=[0.05, 0.02, 0.01, 0.005], value=0.01, key="form_nu"
    )
    N_form = st.select_slider(
        "Grid points N", options=[64, 128, 256], value=256, key="form_N"
    )

    if st.button("Run both formulations", type="primary"):
        x_app  = np.linspace(0, 2*np.pi, N_form, endpoint=False)
        u0_app = np.sin(x_app)
        dx_app = 2*np.pi / N_form

        with st.spinner("Running advective..."):
            u_adv, _ = solve_fdm(u0_app, N_form, 1.0, nu_form, cfl=0.4, formulation='advective')
        with st.spinner("Running conservative..."):
            u_con, _ = solve_fdm(u0_app, N_form, 1.0, nu_form, cfl=0.4, formulation='conservative')

        l2_diff = np.sqrt(dx_app * np.sum((u_adv - u_con)**2))

        fig, axes = plt.subplots(1, 2, figsize=(11, 4))

        # Full profile overlay
        axes[0].plot(x_app, u_adv, '-',  color='tomato',    lw=2, label='Advective')
        axes[0].plot(x_app, u_con, '--', color='steelblue', lw=2, label='Conservative')
        axes[0].set_title(f'Full profile  |  ν={nu_form}, N={N_form}')
        axes[0].set_xlabel('x'); axes[0].legend(); axes[0].grid(alpha=0.3)
        axes[0].set_ylim(-1.5, 1.5)

        # Shock-layer zoom
        mask = (x_app >= 2.5) & (x_app <= 4.5)
        axes[1].plot(x_app[mask], u_adv[mask], '-',  color='tomato',    lw=2.5, label='Advective')
        axes[1].plot(x_app[mask], u_con[mask], '--', color='steelblue', lw=2.5, label='Conservative')
        axes[1].fill_between(x_app[mask], u_adv[mask], u_con[mask], alpha=0.2, color='purple')
        axes[1].set_title('Shock-layer zoom  [2.5, 4.5]')
        axes[1].set_xlabel('x'); axes[1].legend(); axes[1].grid(alpha=0.3)

        plt.tight_layout()
        st.pyplot(fig)
        plt.close()
        st.metric("L2 difference (adv − con)", f"{l2_diff:.3e}")


# ==================================================================
# Tab 6 -- About
# ==================================================================
with tab6:
    st.header("About This Project")

    st.subheader("The PDE")
    st.markdown(
        "We study the **viscous Burgers equation** on a periodic domain:"
    )
    st.latex(
        r"u_t + u\,u_x = \nu\,u_{xx}, \quad "
        r"x \in [0,\,2\pi], \quad "
        r"u(x,0) = \sin(x)"
    )
    st.markdown(
        r"The parameter $\nu > 0$ is the kinematic viscosity. "
        r"For large $\nu$ the solution stays smooth. "
        r"For small $\nu$ a steep shock layer of width $\delta \sim \nu/|u|$ forms near $x = \pi$."
    )

    st.subheader("Methods")
    st.markdown(r"""
| Method | Spatial order | Time scheme | Shock handling |
|---|---|---|---|
| **FDM** | $O(1/N)$ upwind | Crank-Nicolson | Smears (artificial viscosity) |
| **FEM** | $O(1/N^2)$ Galerkin P1 | Crank-Nicolson | Oscillates (no upwinding) |
| **Spectral** | Exponential (smooth) | Integrating factor RK4 | Gibbs ringing |
    """)

    st.subheader("Key Results")
    st.markdown(r"""
- Spectral achieves machine precision ($\sim 10^{-14}$) at $N=128$ for smooth data
- FEM is $10\times$ more accurate than FDM at the same grid size
- Near shocks ($\nu = 0.005$): all three methods fail in different ways
- Memory is $O(N)$ for all methods; runtime slopes reflect implementation overhead
    """)

    st.subheader("Links")
    st.markdown(
        "**Source code:** "
        "[github.com/alex-rosas/burgers-pde-solvers]"
        "(https://github.com/alex-rosas/burgers-pde-solvers)"
    )