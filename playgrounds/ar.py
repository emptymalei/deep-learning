# /// script
# requires-python = ">=3.10"
# dependencies = [
#     "marimo",
#     "matplotlib",
#     "numpy",
# ]
# ///
"""AR(2) playground, exported to WebAssembly by scripts/build_playgrounds.sh.

Embedded in dl/time-series/timeseries-basics.ar.md. Only numpy and
matplotlib are used, so it runs in Pyodide.
"""

import marimo

__generated_with = "0.25.2"
app = marimo.App(width="medium", app_title="AR(2) playground")


@app.cell
def _():
    import marimo as mo
    import numpy as np
    from matplotlib.figure import Figure

    return Figure, mo, np


@app.cell
def _(mo):
    mo.md(
        r"""
    $$
    s(t) = \phi_0 + \phi_1 s(t-1) + \phi_2 s(t-2) + \epsilon,
    \qquad \epsilon \sim \mathcal{N}(0, \sigma^2)
    $$

    Pick a preset or move the sliders. The series starts from
    $s(-1) = s(-2) = 0$. Watch the dot in the $(\phi_1, \phi_2)$ plane:
    the process is stationary only inside the triangle.
    """
    )
    return


@app.cell
def _(mo):
    PRESETS = {
        "White noise": {"phi0": 0.0, "phi1": 0.0, "phi2": 0.0},
        "Random walk": {"phi0": 0.0, "phi1": 1.0, "phi2": 0.0},
        "Damped": {"phi0": 0.0, "phi1": 0.5, "phi2": 0.3},
        "Oscillating": {"phi0": 0.0, "phi1": 1.5, "phi2": -0.9},
        "Explosive": {"phi0": 0.0, "phi1": 0.6, "phi2": 0.45},
    }
    get_coefs, set_coefs = mo.state(PRESETS["Damped"])
    return PRESETS, get_coefs, set_coefs


@app.cell
def _(PRESETS, mo, set_coefs):
    preset_buttons = mo.ui.array(
        [
            mo.ui.button(label=name, on_click=lambda _, p=p: set_coefs(dict(p)))
            for name, p in PRESETS.items()
        ]
    )
    return (preset_buttons,)


@app.cell
def _(get_coefs, mo):
    # Re-created whenever a preset is clicked, so the sliders jump to it.
    _c = get_coefs()
    phi1 = mo.ui.slider(
        -2.5, 2.5, 0.05, value=_c["phi1"], label=r"$\phi_1$", show_value=True
    )
    phi2 = mo.ui.slider(
        -1.5, 1.5, 0.05, value=_c["phi2"], label=r"$\phi_2$", show_value=True
    )
    phi0 = mo.ui.slider(
        -2.0, 2.0, 0.1, value=_c["phi0"], label=r"$\phi_0$", show_value=True
    )
    return phi0, phi1, phi2


@app.cell
def _(mo):
    sigma = mo.ui.slider(0.0, 3.0, 0.1, value=1.0, label=r"$\sigma$", show_value=True)
    n_steps = mo.ui.slider(
        50, 1000, 10, value=300, label="steps", show_value=True, debounce=True
    )
    seed = mo.ui.slider(0, 100, 1, value=7, label="seed", show_value=True)
    return n_steps, seed, sigma


@app.cell
def _(mo, n_steps, phi0, phi1, phi2, preset_buttons, seed, sigma):
    mo.vstack(
        [
            mo.hstack(
                [mo.md("**Presets**"), *preset_buttons],
                justify="start",
                align="center",
                wrap=True,
            ),
            mo.hstack([phi1, phi2, phi0], widths="equal", gap=1.5),
            mo.hstack([sigma, n_steps, seed], widths="equal", gap=1.5),
        ],
        gap=0.75,
    )
    return


@app.cell
def _(np):
    LIMIT = 1e6  # stop simulating once |s(t)| exceeds this

    def simulate(phi0, phi1, phi2, sigma, n, seed):
        """Simulate n steps of AR(2) from s(-1) = s(-2) = 0.

        Returns the series and the step at which it blew past LIMIT (or None).
        """
        eps = np.random.default_rng(seed).normal(0.0, sigma, n)
        s = np.zeros(n)
        prev1 = prev2 = 0.0
        for t in range(n):
            x = phi0 + phi1 * prev1 + phi2 * prev2 + eps[t]
            if not abs(x) < LIMIT:
                return s[:t], t
            s[t] = x
            prev2, prev1 = prev1, x
        return s, None

    def sample_acf(s, max_lag):
        """Sample autocorrelation r_0 .. r_max_lag, or None if s is constant."""
        s = np.asarray(s, dtype=float)
        n = len(s)
        if n < 2:
            return None
        d = s - s.mean()
        denom = d @ d
        if denom <= 1e-12 * max(1.0, s @ s):
            return None
        max_lag = min(max_lag, n - 1)
        return np.array([d[: n - k] @ d[k:] for k in range(max_lag + 1)]) / denom

    def theoretical_acf(phi1, phi2, max_lag):
        """ACF of a stationary AR(2) from the Yule-Walker recursion."""
        rho = np.empty(max_lag + 1)
        rho[0] = 1.0
        if max_lag >= 1:
            rho[1] = phi1 / (1.0 - phi2)
        for k in range(2, max_lag + 1):
            rho[k] = phi1 * rho[k - 1] + phi2 * rho[k - 2]
        return rho

    return LIMIT, sample_acf, simulate, theoretical_acf


@app.cell
def _(np, phi0, phi1, phi2):
    # Round away slider float noise such as 0.15000000000000002.
    p0, p1, p2 = (round(float(w.value), 4) for w in (phi0, phi1, phi2))

    # Inverse roots: lambda^2 - phi1 lambda - phi2 = 0. The characteristic
    # roots of 1 - phi1 z - phi2 z^2 are z = 1 / lambda.
    lam = np.roots([1.0, -p1, -p2])
    lam_mod = np.sort(np.abs(lam))[::-1]
    TOL = 1e-9
    stationary = lam_mod[0] < 1.0 - TOL
    unit_root = abs(lam_mod[0] - 1.0) <= TOL
    disc = p1**2 + 4.0 * p2
    complex_roots = disc < -TOL
    return complex_roots, disc, lam, lam_mod, p0, p1, p2, stationary, unit_root


@app.cell
def _(
    complex_roots,
    disc,
    lam,
    lam_mod,
    mo,
    np,
    p0,
    p1,
    p2,
    stationary,
    unit_root,
):
    # A zero inverse root means phi(z) has lower degree: no root to report.
    _z = ", ".join(f"{1.0 / m:.3f}" for m in lam_mod if m > 1e-12)

    if stationary and not _z:
        _kind, _head = "success", "**Stationary.**"
        _roots = r"No characteristic roots ($\phi_1 = \phi_2 = 0$): white noise."
    elif stationary:
        _kind, _head = "success", "**Stationary.**"
        _roots = f"All characteristic roots lie outside the unit circle: |z| = {_z}."
    elif unit_root:
        _kind, _head = "warn", "**Not stationary: unit root.**"
        _roots = f"A characteristic root sits on the unit circle: |z| = {_z}."
    else:
        _kind, _head = "danger", "**Not stationary: explosive.**"
        _roots = f"A characteristic root lies inside the unit circle: |z| = {_z}."

    if complex_roots:
        _theta = np.arccos(np.clip(p1 / (2.0 * np.sqrt(-p2)), -1.0, 1.0))
        _osc = (
            f"Oscillatory: complex roots ($\\phi_1^2 + 4\\phi_2 = {disc:.2f} < 0$), "
            f"pseudo-period ≈ {2 * np.pi / _theta:.1f} steps."
        )
    elif not _z:
        _osc = ""
    elif np.real(lam[np.argmax(np.abs(lam))]) < -1e-12:
        _osc = (
            f"Not oscillatory in the cyclic sense (real roots, "
            f"$\\phi_1^2 + 4\\phi_2 = {disc:.2f}$), but the dominant root is "
            "negative, so the sign flips from step to step."
        )
    else:
        _osc = f"Not oscillatory: real roots ($\\phi_1^2 + 4\\phi_2 = {disc:.2f}$)."

    _mean = (
        f" Mean $\\phi_0 / (1 - \\phi_1 - \\phi_2) = {p0 / (1.0 - p1 - p2):.3f}$."
        if stationary
        else ""
    )

    mo.callout(mo.md(f"{_head} {_roots} {_osc}{_mean}"), kind=_kind)
    return


@app.cell
def _(
    Figure,
    LIMIT,
    complex_roots,
    n_steps,
    np,
    p0,
    p1,
    p2,
    sample_acf,
    seed,
    sigma,
    simulate,
    stationary,
    theoretical_acf,
    unit_root,
):
    MAX_LAG = 30
    C_DATA, C_THEORY, C_STAT, C_OSC = "#1f77b4", "#ff7f0e", "#2ca02c", "#9467bd"

    series, blew_up_at = simulate(
        p0, p1, p2, float(sigma.value), int(n_steps.value), int(seed.value)
    )

    fig = Figure(figsize=(8, 5.6), layout="constrained")
    grid = fig.add_gridspec(2, 2, height_ratios=[1, 1.15], width_ratios=[1, 1])
    ax_s = fig.add_subplot(grid[0, :])
    ax_acf = fig.add_subplot(grid[1, 0])
    ax_tri = fig.add_subplot(grid[1, 1])

    # (a) the simulated series
    ax_s.plot(np.arange(len(series)), series, color=C_DATA, lw=1)
    if stationary:
        ax_s.axhline(
            p0 / (1.0 - p1 - p2),
            color="0.4",
            ls="--",
            lw=1,
            label="stationary mean",
        )
        ax_s.legend(loc="upper right", fontsize=8, frameon=False)
    ax_s.set_xlim(0, int(n_steps.value))
    ax_s.set_xlabel("$t$")
    ax_s.set_ylabel("$s(t)$")
    if blew_up_at is not None:
        ax_s.set_title(
            f"Simulated series (stopped at t = {blew_up_at}: |s| > {LIMIT:.0e})",
            fontsize=10,
        )
    else:
        ax_s.set_title("Simulated series", fontsize=10)

    # (b) sample ACF
    acf = sample_acf(series, MAX_LAG)
    if acf is None:
        ax_acf.text(
            0.5,
            0.5,
            "constant series:\nACF undefined",
            ha="center",
            va="center",
            transform=ax_acf.transAxes,
        )
    else:
        lags = np.arange(len(acf))
        band = 1.96 / np.sqrt(len(series))
        ax_acf.axhspan(-band, band, color="0.85", lw=0)
        ax_acf.bar(lags, acf, width=0.6, color=C_DATA, label="sample")
        if stationary:
            ax_acf.plot(
                lags,
                theoretical_acf(p1, p2, len(acf) - 1),
                "o",
                ms=3,
                color=C_THEORY,
                label="theory",
            )
        ax_acf.legend(loc="upper right", fontsize=8, frameon=False)
    ax_acf.axhline(0, color="0.3", lw=0.8)
    ax_acf.set_xlim(-0.8, MAX_LAG + 0.8)
    ax_acf.set_ylim(-1.05, 1.05)
    ax_acf.set_xlabel("lag")
    ax_acf.set_title("Sample ACF (grey: ±1.96/√n)", fontsize=10)

    # (c) stationarity triangle
    f1 = np.linspace(-2.7, 2.7, 300)
    parabola = -(f1**2) / 4.0  # phi1^2 + 4 phi2 = 0
    tri = np.array([[-2.0, -1.0], [2.0, -1.0], [0.0, 1.0], [-2.0, -1.0]])
    ax_tri.fill(tri[:, 0], tri[:, 1], color=C_STAT, alpha=0.2, lw=0, label="stationary")
    ax_tri.fill_between(
        f1,
        parabola,
        -1.7,
        facecolor="none",
        edgecolor=C_OSC,
        hatch="////",
        alpha=0.45,
        lw=0,
        label=r"complex roots: $\phi_1^2 + 4\phi_2 < 0$",
    )
    ax_tri.plot(tri[:, 0], tri[:, 1], color=C_STAT, lw=1.2)
    ax_tri.plot(f1, parabola, color=C_OSC, lw=0.8, ls=":")
    if stationary:
        dot = "black"
    elif unit_root:
        dot = "#e6a700"
    else:
        dot = "#d62728"
    ax_tri.plot(
        [p1], [p2], "o", ms=9, mfc=dot, mec="white", mew=1.5, zorder=5, clip_on=False
    )
    ax_tri.axhline(0, color="0.7", lw=0.5)
    ax_tri.axvline(0, color="0.7", lw=0.5)
    ax_tri.set_xlim(-2.7, 2.7)
    ax_tri.set_ylim(-1.7, 1.7)
    ax_tri.set_xlabel(r"$\phi_1$")
    ax_tri.set_ylabel(r"$\phi_2$")
    ax_tri.set_title(
        r"$(\phi_1, \phi_2)$" + (" (oscillatory)" if complex_roots else ""),
        fontsize=10,
    )
    ax_tri.legend(loc="upper left", fontsize=7, frameon=False)

    fig
    return


if __name__ == "__main__":
    app.run()
