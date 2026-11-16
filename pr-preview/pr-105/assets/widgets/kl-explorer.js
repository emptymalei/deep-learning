/*
 * KL divergence explorer: a Gaussian p against a two-mode Gaussian mixture q,
 * as in the figure on the KL divergence page. The reader moves p (mean and
 * spread) and the distance between the modes of q, and sees
 *
 *   - both densities,
 *   - where each direction of the divergence comes from: the integrands
 *     p log(p/q) and q log(q/p), point by point,
 *   - D_KL(p‖q) and D_KL(q‖p) in nats, each with its curve over the mean of p.
 *
 * The two numbers differ: the KL divergence is not symmetric.
 *
 *     <div class="dl-widget" data-widget="kl-explorer" markdown>
 *     Fallback text for readers without JavaScript.
 *     </div>
 *
 * Integrals are trapezoid sums on [-12, 12], computed with log densities so
 * that far tails do not underflow.
 */
dlWidgets.register("kl-explorer", (el, api) => {
    const NS = "http://www.w3.org/2000/svg";
    const X0 = -12, X1 = 12, N = 1200; // integration grid
    const V0 = -7, V1 = 7; // what the chart shows
    const Q_SIGMA = 0.7;
    const MU = { min: -5, max: 5, step: 0.1 };
    const state = { mu: 0, sigma: 1, d: 2 };
    const PRESETS = [
        ["On one mode", { mu: 2, sigma: 0.7, d: 2 }],
        ["Between the modes", { mu: 0, sigma: 0.7, d: 2 }],
        ["Covering both", { mu: 0, sigma: 2.2, d: 2 }],
    ];

    const xs = Array.from({ length: N + 1 }, (_, i) => X0 + ((X1 - X0) * i) / N);
    const dx = (X1 - X0) / N;
    const LOG_SQRT_2PI = 0.5 * Math.log(2 * Math.PI);
    const logNormal = (x, m, s) => -0.5 * ((x - m) / s) ** 2 - Math.log(s) - LOG_SQRT_2PI;
    function logMix(x, d) {
        const a = logNormal(x, -d, Q_SIGMA), b = logNormal(x, d, Q_SIGMA);
        const hi = Math.max(a, b);
        return hi + Math.log(0.5 * Math.exp(a - hi) + 0.5 * Math.exp(b - hi));
    }

    // Densities and the two integrands on the grid.
    function compute(mu, sigma, d) {
        const p = new Float64Array(N + 1), q = new Float64Array(N + 1);
        const f = new Float64Array(N + 1), r = new Float64Array(N + 1);
        let forward = 0, reverse = 0;
        for (let i = 0; i <= N; i++) {
            const lp = logNormal(xs[i], mu, sigma), lq = logMix(xs[i], d);
            p[i] = Math.exp(lp);
            q[i] = Math.exp(lq);
            f[i] = p[i] > 0 ? p[i] * (lp - lq) : 0; // p log(p/q)
            r[i] = q[i] > 0 ? q[i] * (lq - lp) : 0; // q log(q/p)
            const w = i === 0 || i === N ? 0.5 : 1;
            forward += w * f[i] * dx;
            reverse += w * r[i] * dx;
        }
        return { p, q, f, r, forward, reverse };
    }

    // ---- DOM ---------------------------------------------------------------
    el.innerHTML = "";
    const style = document.createElement("style");
    style.textContent = `
.dl-kl { display: grid; grid-template-columns: repeat(2, minmax(0, 1fr)); gap: .4rem; margin: .5rem 0 .2rem; }
.dl-kl > div { min-width: 0; padding: .3rem .45rem; border-radius: .3rem; background: var(--md-default-fg-color--lightest); line-height: 1.25; }
.dl-kl b { display: block; font-size: .6rem; font-weight: 600; color: var(--md-default-fg-color--light); }
.dl-kl span { font-variant-numeric: tabular-nums; font-size: .78rem; }
.dl-kl small { display: block; font-size: .58rem; color: var(--md-default-fg-color--light); }
.dl-kl svg { margin: .15rem 0; overflow: hidden; }
.dl-kl__sliders output { min-width: 2.6em; font-variant-numeric: tabular-nums; }`;

    const controls = document.createElement("div");
    controls.className = "dl-widget__controls dl-kl__sliders";
    const inputs = {};
    function slider(label, key, min, max, step) {
        const wrap = document.createElement("label");
        const span = document.createElement("span");
        span.textContent = label;
        const input = document.createElement("input");
        Object.assign(input, { type: "range", min, max, step, value: state[key] });
        const out = document.createElement("output");
        input.addEventListener("input", () => {
            state[key] = parseFloat(input.value);
            render();
        });
        wrap.append(span, input, out);
        inputs[key] = [input, out];
        return wrap;
    }
    controls.append(
        slider("Mean of p", "mu", MU.min, MU.max, MU.step),
        slider("Spread of p", "sigma", 0.3, 3, 0.1),
        slider("Modes of q at ±", "d", 0, 3, 0.1)
    );
    const seg = document.createElement("div");
    seg.className = "dl-widget__seg";
    seg.setAttribute("role", "group");
    seg.setAttribute("aria-label", "Examples");
    for (const [label, values] of PRESETS) {
        const b = document.createElement("button");
        b.type = "button";
        b.textContent = label;
        b.addEventListener("click", () => {
            Object.assign(state, values);
            render();
        });
        seg.appendChild(b);
    }
    const presets = document.createElement("div");
    presets.className = "dl-widget__controls";
    presets.appendChild(seg);

    const svg = document.createElementNS(NS, "svg");
    svg.setAttribute("role", "img");
    const legend = document.createElement("div");
    legend.className = "dl-widget__legend";
    const tiles = document.createElement("div");
    tiles.className = "dl-kl";
    const caption = document.createElement("p");
    caption.className = "dl-widget__caption";
    caption.setAttribute("aria-live", "polite");
    el.append(style, presets, controls, svg, legend, tiles, caption);

    function node(tag, attrs, parent) {
        const n = document.createElementNS(NS, tag);
        for (const k in attrs) n.setAttribute(k, attrs[k]);
        if (parent) parent.appendChild(n);
        return n;
    }
    const fmt = (v) => (v >= 100 ? v.toFixed(0) : v >= 10 ? v.toFixed(1) : v.toFixed(2));

    // ---- Chart: densities on top, the two integrands below -------------------
    let W = 640;
    function drawChart(c, res) {
        W = Math.max(280, Math.min(720, Math.round(el.clientWidth - 2) || 640));
        const padL = 8, padR = 8, topH = W < 480 ? 110 : 130, gap = 26, botH = W < 480 ? 90 : 105, axis = 16;
        const H = topH + gap + botH + axis;
        const x = (v) => padL + ((W - padL - padR) * (v - V0)) / (V1 - V0);
        svg.setAttribute("viewBox", `0 0 ${W} ${H}`);
        svg.setAttribute("aria-label", `A Gaussian p with mean ${state.mu.toFixed(1)} and spread ${state.sigma.toFixed(1)}, and a mixture q with modes at plus and minus ${state.d.toFixed(1)}. Below, the pointwise contributions to the two KL divergences.`);
        svg.innerHTML = "";
        const visible = [];
        for (let i = 0; i <= N; i++) if (xs[i] >= V0 && xs[i] <= V1) visible.push(i);

        // Top: densities.
        let peak = 0;
        for (const i of visible) peak = Math.max(peak, res.p[i], res.q[i]);
        const yT = (v) => topH - (topH - 14) * (v / peak);
        node("text", { x: padL, y: 10, "font-size": 10, fill: c.fgLight }, svg).textContent = "densities";
        node("line", { x1: padL, x2: W - padR, y1: topH, y2: topH, stroke: c.fgLighter }, svg);
        const line = (arr, y) => visible.map((i) => `${x(xs[i]).toFixed(1)},${y(arr[i]).toFixed(1)}`).join(" ");
        node("polygon", { points: `${x(V0)},${topH} ${line(res.q, yT)} ${x(V1)},${topH}`, fill: c.fg, "fill-opacity": 0.07 }, svg);
        node("polyline", { points: line(res.q, yT), fill: "none", stroke: c.fg, "stroke-width": 1.6 }, svg);
        node("polyline", { points: line(res.p, yT), fill: "none", stroke: c.train, "stroke-width": 2.2 }, svg);

        // Bottom: p log(p/q) and q log(q/p), around a zero line.
        const top2 = topH + gap, mid = top2 + botH * 0.62;
        let hi = 1e-9, lo = 1e-9;
        for (const i of visible) {
            hi = Math.max(hi, res.f[i], res.r[i]);
            lo = Math.max(lo, -res.f[i], -res.r[i]);
        }
        const unit = Math.min((mid - top2 - 2) / hi, (top2 + botH - mid) / lo);
        const yB = (v) => mid - v * unit;
        node("text", { x: padL, y: top2 - 8, "font-size": 10, fill: c.fgLight }, svg).textContent = "contribution to each divergence at every point";
        for (const [arr, col] of [[res.f, c.train], [res.r, c.accent]]) {
            node("polygon", { points: `${x(V0)},${mid} ${line(arr, yB)} ${x(V1)},${mid}`, fill: col, "fill-opacity": 0.22 }, svg);
            node("polyline", { points: line(arr, yB), fill: "none", stroke: col, "stroke-width": 1.6 }, svg);
        }
        node("line", { x1: padL, x2: W - padR, y1: mid, y2: mid, stroke: c.fgLighter }, svg);
        for (let v = -6; v <= 6; v += 2) {
            node("text", { x: x(v), y: H - 3, "text-anchor": "middle", "font-size": 10, fill: c.fgLight }, svg).textContent = v < 0 ? "−" + -v : v;
        }

        legend.innerHTML = "";
        for (const [label, col] of [
            ["p and p log(p/q)", c.train],
            ["q", c.fg],
            ["q log(q/p)", c.accent],
        ]) {
            const s = document.createElement("span");
            s.style.setProperty("--swatch", col);
            s.textContent = label;
            legend.appendChild(s);
        }
    }

    // ---- Tiles: each divergence and its curve over the mean of p -------------
    function drawTiles(c, res) {
        const grid = [];
        for (let m = MU.min; m <= MU.max + 1e-9; m += 0.25) {
            const g = compute(m, state.sigma, state.d);
            grid.push([m, g.forward, g.reverse]);
        }
        tiles.innerHTML = "";
        const sparks = [];
        for (const [title, value, col, k] of [
            ["D(p ‖ q)", res.forward, c.train, 1],
            ["D(q ‖ p)", res.reverse, c.accent, 2],
        ]) {
            const cell = document.createElement("div");
            const b = document.createElement("b");
            b.textContent = title;
            const span = document.createElement("span");
            span.textContent = `${fmt(value)} nats`;
            const spark = document.createElementNS(NS, "svg");
            spark.setAttribute("aria-hidden", "true");
            const small = document.createElement("small");
            small.textContent = "against the mean of p";
            cell.append(b, span, spark, small);
            tiles.appendChild(cell);
            sparks.push([spark, value, col, k]);
        }
        for (const [spark, value, col, k] of sparks) {
            const w = Math.max(60, Math.round(spark.getBoundingClientRect().width) || 140);
            const h = 50, top = 4, base = h - 12;
            const cap = Math.max(...grid.map((g) => g[k]), value, 1e-9) * 1.08;
            const xS = (m) => 5 + ((w - 10) * (m - MU.min)) / (MU.max - MU.min);
            const yS = (v) => base - (base - top) * (v / cap);
            spark.setAttribute("viewBox", `0 0 ${w} ${h}`);
            const pts = grid.map((g) => `${xS(g[0]).toFixed(1)},${yS(g[k]).toFixed(1)}`);
            node("path", { d: `M${pts.join("L")}L${xS(MU.max)},${base}L${xS(MU.min)},${base}Z`, fill: col, "fill-opacity": 0.14 }, spark);
            node("path", { d: `M${pts.join("L")}`, fill: "none", stroke: col, "stroke-width": 1.8 }, spark);
            node("line", { x1: xS(MU.min), x2: xS(MU.max), y1: base, y2: base, stroke: c.fgLighter }, spark);
            for (const [m, anchor] of [[MU.min, "start"], [0, "middle"], [MU.max, "end"]]) {
                node("text", { x: xS(m), y: h - 1, "text-anchor": anchor, "font-size": 8, fill: c.fgLight }, spark).textContent = m < 0 ? "−" + -m : m;
            }
            node("line", { x1: xS(state.mu), x2: xS(state.mu), y1: top - 2, y2: base, stroke: c.fg, "stroke-opacity": 0.45 }, spark);
            node("circle", { cx: xS(state.mu), cy: yS(value), r: 3.2, fill: col, stroke: c.bg, "stroke-width": 1 }, spark);
        }
    }

    function explain(res) {
        const a = res.forward, b = res.reverse;
        const head = `D(p ‖ q) = ${fmt(a)} and D(q ‖ p) = ${fmt(b)} nats`;
        if (Math.abs(a - b) < 0.005 * Math.max(a, b, 1e-9) || Math.max(a, b) < 0.005) {
            return `${head}: here the two directions happen to agree.`;
        }
        const why = a > b
            ? "D(p ‖ q) is larger: p puts mass where q has little, and that is what this direction counts."
            : "D(q ‖ p) is larger: q puts mass where p has little, and that is what this direction counts.";
        return `${head}. Swapping the two distributions changes the value, so the KL divergence is not symmetric. ${why}`;
    }

    function render() {
        const c = api.colors();
        const res = compute(state.mu, state.sigma, state.d);
        for (const key in inputs) {
            inputs[key][0].value = state[key];
            inputs[key][1].textContent = key === "mu" && state.mu < 0 ? "−" + (-state.mu).toFixed(1) : state[key].toFixed(1);
        }
        for (const [i, b] of [...seg.children].entries()) {
            const v = PRESETS[i][1];
            b.setAttribute("aria-pressed", String(v.mu === state.mu && v.sigma === state.sigma && v.d === state.d));
        }
        drawChart(c, res);
        drawTiles(c, res);
        caption.textContent = explain(res);
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
