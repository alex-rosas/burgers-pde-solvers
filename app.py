"""
app.py
======
Streamlit interactive app — Three Roads to Burgers.
Run: streamlit run app.py

Six tabs:
  1. Solver Explorer  -- live solver, auto-updates on slider change
  2. Convergence      -- plots from results/convergence.csv
  3. Performance      -- plots from results/performance.csv
  4. Shock Resolution -- panel figures + live comparison
  5. Formulation      -- conservative vs advective FDM
  6. About            -- project description and links
"""

from pathlib import Path
import numpy as np
import pandas as pd
import streamlit as st
import plotly.graph_objects as go
from plotly.subplots import make_subplots

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
COLORS  = {"FDM": "#FF6347",  "FEM": "#4682B4", "Spectral": "#3CB371"}

# Shared Plotly layout applied to every figure
PLOTLY_LAYOUT = dict(
    template="plotly_dark",
    paper_bgcolor="rgba(0,0,0,0)",
    plot_bgcolor="rgba(20,20,30,1)",
    font=dict(color="white"),
    margin=dict(l=50, r=20, t=50, b=50),
)

tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs(
    ["Solver Explorer", "Convergence", "Performance", "Shock Resolution", "Formulation", "About"]
)


# ---- Cached solvers -----------------------------------------------

@st.cache_data(show_spinner=False)
def run_solver(method: str, N: int, T: float, nu: float, cfl: float):
    x    = np.linspace(0, 2 * np.pi, N, endpoint=False)
    u0   = np.sin(x)
    u_num, _ = SOLVERS[method](u0, N, T, nu, cfl=cfl)
    u_ex     = u_exact(x, T, nu)
    dx        = 2 * np.pi / N
    err       = float(np.sqrt(dx * np.sum((u_num - u_ex) ** 2)))
    return x, u_num, u_ex, err


@st.cache_data(show_spinner=False)
def run_all_solvers(N: int, nu: float):
    x  = np.linspace(0, 2 * np.pi, N, endpoint=False)
    u0 = np.sin(x)
    results = {m: SOLVERS[m](u0, N, 1.0, nu, cfl=0.4)[0] for m in SOLVERS}
    return x, results


@st.cache_data(show_spinner=False)
def run_formulations(N: int, nu: float):
    x  = np.linspace(0, 2 * np.pi, N, endpoint=False)
    u0 = np.sin(x)
    u_adv, _ = solve_fdm(u0, N, 1.0, nu, cfl=0.4, formulation="advective")
    u_con, _ = solve_fdm(u0, N, 1.0, nu, cfl=0.4, formulation="conservative")
    dx       = 2 * np.pi / N
    l2_diff  = float(np.sqrt(dx * np.sum((u_adv - u_con) ** 2)))
    return x, u_adv, u_con, l2_diff


# ==================================================================
# Tab 1 -- Solver Explorer
# ==================================================================
with tab1:
    st.header("Interactive Solver Explorer")
    st.markdown(
        r"Adjust any parameter and the plot updates instantly. "
        r"Watch what happens as $\nu \to 0$ (shock forms) "
        r"or as $N$ increases (solution converges)."
    )

    col1, col2 = st.columns([1, 2])

    with col1:
        method = st.selectbox("Method", list(SOLVERS.keys()))
        N      = st.select_slider("Grid points N",
                     options=[16, 32, 64, 128, 256, 512], value=64)
        nu     = st.select_slider(r"Viscosity ν",
                     options=[0.1, 0.05, 0.02, 0.01, 0.005], value=0.1)
        T      = st.slider("Final time T", 0.1, 2.0, 1.0, step=0.1)
        cfl    = st.slider("CFL number",   0.1, 0.9, 0.4, step=0.1)

    with col2:
        with st.spinner("Solving…"):
            x, u_num, u_ex, err = run_solver(method, N, T, nu, cfl)

        fig = make_subplots(rows=1, cols=2,
                            subplot_titles=("Solution vs Exact", "Pointwise Error"))

        fig.add_trace(go.Scatter(x=x, y=u_ex, name="Exact",
                                 line=dict(color="white", dash="dash", width=2)),
                      row=1, col=1)
        fig.add_trace(go.Scatter(x=x, y=u_num, name=method,
                                 line=dict(color=COLORS[method], width=2)),
                      row=1, col=1)
        fig.add_trace(go.Scatter(x=x, y=u_num - u_ex, name="Error",
                                 line=dict(color=COLORS[method], width=1.5),
                                 showlegend=False),
                      row=1, col=2)
        fig.add_hline(y=0, line=dict(color="gray", dash="dash", width=0.5), row=1, col=2)

        fig.update_xaxes(title_text="x")
        fig.update_yaxes(title_text="u",     row=1, col=1)
        fig.update_yaxes(title_text="error", row=1, col=2)
        fig.update_layout(**PLOTLY_LAYOUT,
                          title=f"{method}  |  L² error = {err:.2e}",
                          height=420)
        st.plotly_chart(fig, use_container_width=True)
        st.metric("L² error", f"{err:.3e}")


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
            fig = go.Figure()
            for m in methods:
                sub = df[df["method"] == m]
                fig.add_trace(go.Scatter(x=sub["N"], y=sub["error"],
                                         mode="lines+markers", name=m,
                                         line=dict(color=COLORS[m], width=2),
                                         marker=dict(size=7)))
            N_ref = np.array([16, 1024])
            fig.add_trace(go.Scatter(x=N_ref, y=0.8 * (N_ref / 16) ** (-1),
                                     mode="lines", name="slope −1",
                                     line=dict(color="white", dash="dash", width=1)))
            fig.add_trace(go.Scatter(x=N_ref, y=0.3 * (N_ref / 16) ** (-2),
                                     mode="lines", name="slope −2",
                                     line=dict(color="lightgray", dash="dot", width=1)))
            fig.update_xaxes(type="log", title="N")
            fig.update_yaxes(type="log", title="L² error")
            fig.update_layout(**PLOTLY_LAYOUT, title="Log-log convergence", height=380)
            st.plotly_chart(fig, use_container_width=True)

        with col_b:
            st.subheader("Semi-log plot")
            fig = go.Figure()
            for m in methods:
                sub = df[df["method"] == m]
                fig.add_trace(go.Scatter(x=sub["N"], y=sub["error"],
                                         mode="lines+markers", name=m,
                                         line=dict(color=COLORS[m], width=2),
                                         marker=dict(size=7)))
            fig.update_xaxes(title="N")
            fig.update_yaxes(type="log", title="L² error (log scale)")
            fig.update_layout(**PLOTLY_LAYOUT,
                              title="Exponential decay of spectral method", height=380)
            st.plotly_chart(fig, use_container_width=True)

        st.subheader("Measured convergence slopes")
        rows = []
        for m in ["FDM", "FEM"]:
            sub   = df[df["method"] == m]
            slope = np.polyfit(sub["log_N"], sub["log_E"], 1)[0]
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
            fig = go.Figure()
            for m in ["FDM", "FEM", "Spectral"]:
                sub = df[df["method"] == m]
                fig.add_trace(go.Scatter(x=sub["N"], y=sub["runtime"],
                                         mode="lines+markers", name=m,
                                         line=dict(color=COLORS[m], width=2),
                                         marker=dict(size=7)))
            N_ref = np.array([64, 2048])
            t_ref = df[df["method"] == "FDM"]["runtime"].values[0]
            fig.add_trace(go.Scatter(x=N_ref, y=t_ref * (N_ref / 64) ** 1.0,
                                     mode="lines", name="O(N)",
                                     line=dict(color="white", dash="dash", width=1)))
            fig.add_trace(go.Scatter(x=N_ref,
                                     y=t_ref * (N_ref / 64) * np.log2(N_ref / 64 + 1),
                                     mode="lines", name="O(N log N)",
                                     line=dict(color="lightgray", dash="dot", width=1)))
            fig.update_xaxes(type="log", title="N")
            fig.update_yaxes(type="log", title="Runtime (s)")
            fig.update_layout(**PLOTLY_LAYOUT, title="Runtime scaling", height=380)
            st.plotly_chart(fig, use_container_width=True)

        with col_b:
            st.subheader("Memory scaling")
            fig = go.Figure()
            for m in ["FDM", "FEM", "Spectral"]:
                sub = df[df["method"] == m]
                fig.add_trace(go.Scatter(x=sub["N"], y=sub["memory_mb"],
                                         mode="lines+markers", name=m,
                                         line=dict(color=COLORS[m], width=2),
                                         marker=dict(size=7)))
            fig.update_xaxes(type="log", title="N")
            fig.update_yaxes(type="log", title="Peak memory (MB)")
            fig.update_layout(**PLOTLY_LAYOUT, title="Memory scaling", height=380)
            st.plotly_chart(fig, use_container_width=True)
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
                 caption="4×3 panel: rows = viscosity (decreasing), columns = method",
                 use_container_width=True)
    if zoom_path.exists():
        st.image(str(zoom_path),
                 caption=r"Shock layer zoom at ν=0.005: FDM smears, FEM and Spectral are close",
                 use_container_width=True)

    st.subheader("Live comparison")
    st.markdown(r"Pick $\nu$ and $N$ — all three solvers update automatically.")

    col_s1, col_s2 = st.columns(2)
    with col_s1:
        nu_shock = st.select_slider(
            r"Viscosity ν", options=[0.05, 0.02, 0.01, 0.005], value=0.02, key="shock_nu"
        )
    with col_s2:
        N_shock = st.select_slider(
            "Grid points N", options=[64, 128, 256], value=128, key="shock_N"
        )

    with st.spinner("Solving…"):
        x_s, results_s = run_all_solvers(N_shock, nu_shock)

    fig = make_subplots(rows=1, cols=3, subplot_titles=list(SOLVERS.keys()))
    for i, (m, u_num) in enumerate(results_s.items(), start=1):
        fig.add_trace(go.Scatter(x=x_s, y=u_num, name=m,
                                 line=dict(color=COLORS[m], width=2),
                                 showlegend=False),
                      row=1, col=i)
    fig.update_xaxes(title_text="x")
    fig.update_yaxes(range=[-1.5, 1.5])
    fig.update_layout(**PLOTLY_LAYOUT,
                      title=f"ν = {nu_shock},  N = {N_shock}",
                      height=380)
    st.plotly_chart(fig, use_container_width=True)


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
                     caption="L² difference grows monotonically as ν → 0",
                     use_container_width=True)

    zoom_path = ROOT / "figures" / "formulation_zoom.png"
    if zoom_path.exists():
        st.image(str(zoom_path),
                 caption="Shock-layer zoom at ν=0.005 — shaded region is the pointwise difference",
                 use_container_width=True)

    st.subheader("Live comparison")
    st.markdown(r"Pick $\nu$ and $N$ — both formulations update automatically.")

    col_f1, col_f2 = st.columns(2)
    with col_f1:
        nu_form = st.select_slider(
            r"Viscosity ν", options=[0.05, 0.02, 0.01, 0.005], value=0.01, key="form_nu"
        )
    with col_f2:
        N_form = st.select_slider(
            "Grid points N", options=[64, 128, 256], value=256, key="form_N"
        )

    with st.spinner("Solving…"):
        x_f, u_adv, u_con, l2_diff = run_formulations(N_form, nu_form)

    fig = make_subplots(rows=1, cols=2,
                        subplot_titles=("Full Profile", "Shock-layer Zoom [2.5, 4.5]"))

    fig.add_trace(go.Scatter(x=x_f, y=u_adv, name="Advective",
                             line=dict(color="#FF6347", width=2)),
                  row=1, col=1)
    fig.add_trace(go.Scatter(x=x_f, y=u_con, name="Conservative",
                             line=dict(color="#4682B4", dash="dash", width=2)),
                  row=1, col=1)

    mask = (x_f >= 2.5) & (x_f <= 4.5)
    fig.add_trace(go.Scatter(x=x_f[mask], y=u_adv[mask], name="Advective",
                             line=dict(color="#FF6347", width=2.5),
                             showlegend=False),
                  row=1, col=2)
    fig.add_trace(go.Scatter(x=x_f[mask], y=u_con[mask], name="Conservative",
                             line=dict(color="#4682B4", dash="dash", width=2.5),
                             showlegend=False),
                  row=1, col=2)
    fig.add_trace(go.Scatter(
        x=np.concatenate([x_f[mask], x_f[mask][::-1]]),
        y=np.concatenate([u_adv[mask], u_con[mask][::-1]]),
        fill="toself", fillcolor="rgba(128,0,128,0.2)",
        line=dict(color="rgba(255,255,255,0)"),
        name="Difference", showlegend=False),
        row=1, col=2)

    fig.update_xaxes(title_text="x")
    fig.update_yaxes(title_text="u", row=1, col=1)
    fig.update_layout(**PLOTLY_LAYOUT,
                      title=f"ν = {nu_form},  N = {N_form}  |  L² diff = {l2_diff:.3e}",
                      height=420)
    st.plotly_chart(fig, use_container_width=True)
    st.metric("L² difference (adv − con)", f"{l2_diff:.3e}")


# ==================================================================
# Tab 6 -- About
# ==================================================================
with tab6:
    st.header("About This Project")

    st.subheader("The PDE")
    st.markdown("We study the **viscous Burgers equation** on a periodic domain:")
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
