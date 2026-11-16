/*
 * CRPS explorer: a normal forecast N(mu, sigma) against an observed value
 * x_a = 0. The shaded area is the gap between the forecast CDF and the step
 * at x_a; CRPS integrates the square of that gap,
 *
 *     CRPS = ∫ (F(x) - H(x - x_a))² dx,
 *
 * which for a normal forecast has the closed form
 *     sigma * [ z (2Φ(z) - 1) + 2φ(z) - 1/√π ],  z = (x_a - mu) / sigma.
 *
 *     <div class="dl-widget" data-widget="crps-explorer" markdown>
 *     Fallback text for readers without JavaScript.
 *     </div>
 */
dlWidgets.register("crps-explorer", (el, api) => {
    const NS = "http://www.w3.org/2000/svg";
    const XA = 0;
    const X0 = -5;
    const X1 = 5;
    const state = { mu: 1.5, sigma: 1.0 };

    // Standard normal pdf/cdf (Abramowitz & Stegun 7.1.26 for erf).
    const phi = (z) => Math.exp(-0.5 * z * z) / Math.sqrt(2 * Math.PI);
    function erf(x) {
        const s = Math.sign(x);
        x = Math.abs(x);
        const t = 1 / (1 + 0.3275911 * x);
        const y = 1 - ((((1.061405429 * t - 1.453152027) * t + 1.421413741) * t - 0.284496736) * t + 0.254829592) * t * Math.exp(-x * x);
        return s * y;
    }
    const Phi = (z) => 0.5 * (1 + erf(z / Math.SQRT2));
    function crps(mu, sigma) {
        const z = (XA - mu) / sigma;
        return sigma * (z * (2 * Phi(z) - 1) + 2 * phi(z) - 1 / Math.sqrt(Math.PI));
    }

    // ---- DOM ---------------------------------------------------------------
    el.innerHTML = "";
    const controls = document.createElement("div");
    controls.className = "dl-widget__controls";

    function slider(label, key, min, max, step) {
        const wrap = document.createElement("label");
        const span = document.createElement("span");
        const input = document.createElement("input");
        const out = document.createElement("output");
        Object.assign(input, { type: "range", min, max, step, value: state[key] });
        span.textContent = label;
        out.textContent = state[key].toFixed(2);
        input.addEventListener("input", () => {
            state[key] = parseFloat(input.value);
            out.textContent = state[key].toFixed(2);
            render();
        });
        wrap.append(span, input, out);
        return { wrap, input, out };
    }
    const mu = slider("Mean μ", "mu", -3, 3, 0.05);
    const sigma = slider("Spread σ", "sigma", 0.05, 3, 0.05);

    const seg = document.createElement("div");
    seg.className = "dl-widget__seg";
    seg.setAttribute("role", "group");
    seg.setAttribute("aria-label", "Examples");
    const presets = [
        ["Sharp & right", 0, 0.3],
        ["Sharp & wrong", 2, 0.3],
        ["Wide & wrong", 2, 2],
        ["Point forecast", 1.5, 0.05],
    ];
    for (const [label, m, s] of presets) {
        const b = document.createElement("button");
        b.type = "button";
        b.textContent = label;
        b.addEventListener("click", () => set(m, s));
        seg.appendChild(b);
    }
    function set(m, s) {
        state.mu = m;
        state.sigma = s;
        mu.input.value = m;
        mu.out.textContent = m.toFixed(2);
        sigma.input.value = s;
        sigma.out.textContent = s.toFixed(2);
        render();
    }
    controls.append(mu.wrap, sigma.wrap, seg);

    const svg = document.createElementNS(NS, "svg");
    svg.setAttribute("role", "img");
    const caption = document.createElement("p");
    caption.className = "dl-widget__caption";
    caption.setAttribute("aria-live", "polite");
    const legend = document.createElement("div");
    legend.className = "dl-widget__legend";
    el.append(controls, svg, legend, caption);

    function node(name, attrs, parent) {
        const n = document.createElementNS(NS, name);
        for (const k in attrs) n.setAttribute(k, attrs[k]);
        if (parent) parent.appendChild(n);
        return n;
    }

    let W = 640, H = 250;
    function render() {
        const c = api.colors();
        W = Math.max(300, Math.min(720, Math.round(el.clientWidth - 2) || 640));
        H = W < 480 ? 210 : 250;
        const padL = 34, padR = 12, padT = 12, padB = 28;
        const x = (v) => padL + ((W - padL - padR) * (v - X0)) / (X1 - X0);
        const y = (v) => padT + (H - padT - padB) * (1 - v);
        svg.setAttribute("viewBox", `0 0 ${W} ${H}`);
        svg.innerHTML = "";

        const n = 240;
        const xs = Array.from({ length: n + 1 }, (_, i) => X0 + ((X1 - X0) * i) / n);
        const F = xs.map((v) => Phi((v - state.mu) / state.sigma));
        const Hs = xs.map((v) => (v >= XA ? 1 : 0));
        const peak = phi(0) / state.sigma;
        const densScale = 0.45 / Math.max(peak, 0.45); // keep the density under half height

        // Axes and grid.
        for (const v of [0, 0.5, 1]) {
            node("line", { x1: padL, x2: W - padR, y1: y(v), y2: y(v), stroke: c.fgLightest }, svg);
            node("text", { x: padL - 6, y: y(v) + 3, "text-anchor": "end", "font-size": 10, fill: c.fgLight }, svg).textContent = v;
        }
        for (let v = X0; v <= X1; v++) {
            node("text", { x: x(v), y: H - 12, "text-anchor": "middle", "font-size": 10, fill: c.fgLight }, svg).textContent = v;
        }

        // Shaded gap between the forecast CDF and the step at x_a.
        const top = xs.map((v, i) => `${x(v)},${y(Math.max(F[i], Hs[i]))}`);
        const bottom = xs.map((v, i) => `${x(v)},${y(Math.min(F[i], Hs[i]))}`).reverse();
        node("polygon", { points: top.concat(bottom).join(" "), fill: c.accent, "fill-opacity": 0.25 }, svg);

        // Forecast density (faint) for intuition.
        node("polyline", {
            points: xs.map((v) => `${x(v)},${y((phi((v - state.mu) / state.sigma) / state.sigma) * densScale)}`).join(" "),
            fill: "none", stroke: c.fgLighter, "stroke-width": 1.2, "stroke-dasharray": "4 3",
        }, svg);

        // Step at x_a and the forecast CDF.
        node("polyline", { points: `${x(X0)},${y(0)} ${x(XA)},${y(0)} ${x(XA)},${y(1)} ${x(X1)},${y(1)}`, fill: "none", stroke: c.fg, "stroke-width": 1.5 }, svg);
        node("polyline", { points: xs.map((v, i) => `${x(v)},${y(F[i])}`).join(" "), fill: "none", stroke: c.train, "stroke-width": 2.5 }, svg);
        node("text", { x: x(XA) + 4, y: y(0) - 4, "font-size": 10, fill: c.fg }, svg).textContent = "observed xₐ";

        // Legend, below the chart so it never covers the curves.
        legend.innerHTML = "";
        for (const [label, col] of [
            ["forecast CDF F(x)", c.train],
            ["step H(x − xₐ)", c.fg],
            ["gap that CRPS squares and integrates", c.accent],
            ["forecast density (scaled, dashed)", c.fgLighter],
        ]) {
            const span = document.createElement("span");
            span.style.setProperty("--swatch", col);
            span.textContent = label;
            legend.appendChild(span);
        }

        const score = crps(state.mu, state.sigma);
        const ae = Math.abs(state.mu - XA);
        svg.setAttribute("aria-label", `Forecast N(${state.mu.toFixed(2)}, ${state.sigma.toFixed(2)}) against observation 0; CRPS ${score.toFixed(3)}.`);
        caption.textContent =
            `CRPS = ${score.toFixed(3)} (lower is better). The absolute error of the mean alone is ${ae.toFixed(2)}: ` +
            (state.sigma < 0.1
                ? "as the spread shrinks to a point, CRPS approaches that absolute error."
                : ae > 0.8326 * state.sigma
                ? "the mean misses by more than about 0.83σ, so widening σ would lower the score."
                : "the truth sits inside the spread, so a sharper (smaller σ) forecast would score better.");
    }

    render();
    const ro = "ResizeObserver" in window
        ? new ResizeObserver(() => {
              const w = Math.round(el.clientWidth - 2);
              if (w && Math.abs(w - W) > 8) render();
          })
        : null;
    if (ro) ro.observe(el);
    const offTheme = api.onThemeChange(render);
    return () => {
        if (ro) ro.disconnect();
        offTheme();
    };
});
