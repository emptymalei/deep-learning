/*
 * DTW explorer: two short series with the same shape, the second one shifted
 * and stretched in time by two sliders. The widget shows
 *
 *   - both series, with a line between every pair of points that dynamic
 *     time warping aligns,
 *   - the matrix of accumulated cost D(i, j) from the recursive definition
 *     on the page, with the cheapest path through it,
 *   - the DTW distance next to the plain point-by-point distance.
 *
 * The local distance is d(x, y) = |x − y|.
 *
 *     <div class="dl-widget" data-widget="dtw-explorer" markdown>
 *     Fallback text for readers without JavaScript.
 *     </div>
 */
dlWidgets.register("dtw-explorer", (el, api) => {
    const NS = "http://www.w3.org/2000/svg";
    const N = 18;
    const state = { shift: 2, stretch: 1.3 };
    const PRESETS = [
        ["Identical", { shift: 0, stretch: 1 }],
        ["Shifted", { shift: 3, stretch: 1 }],
        ["Stretched", { shift: 0, stretch: 1.6 }],
        ["Both", { shift: 2, stretch: 1.3 }],
    ];
    // One bump followed by a smaller dip.
    const shape = (t) => Math.exp(-((t - 5) ** 2) / 5) - 0.6 * Math.exp(-((t - 11) ** 2) / 6);
    const A = Array.from({ length: N }, (_, i) => shape(i));
    const seriesB = () => Array.from({ length: N }, (_, j) => shape((j - state.shift) / state.stretch));

    function dtw(a, b) {
        const n = a.length, m = b.length;
        const D = Array.from({ length: n }, () => new Float64Array(m));
        for (let i = 0; i < n; i++) {
            for (let j = 0; j < m; j++) {
                const d = Math.abs(a[i] - b[j]);
                if (i === 0 && j === 0) D[i][j] = d;
                else if (i === 0) D[i][j] = d + D[i][j - 1];
                else if (j === 0) D[i][j] = d + D[i - 1][j];
                else D[i][j] = d + Math.min(D[i - 1][j - 1], D[i][j - 1], D[i - 1][j]);
            }
        }
        // Walk back from the end along the cheapest predecessors.
        const path = [[n - 1, m - 1]];
        let i = n - 1, j = m - 1;
        while (i > 0 || j > 0) {
            if (i === 0) j--;
            else if (j === 0) i--;
            else {
                const best = Math.min(D[i - 1][j - 1], D[i][j - 1], D[i - 1][j]);
                if (best === D[i - 1][j - 1]) { i--; j--; }
                else if (best === D[i - 1][j]) i--;
                else j--;
            }
            path.push([i, j]);
        }
        return { D, path: path.reverse(), distance: D[n - 1][m - 1] };
    }

    el.innerHTML = "";
    const presets = document.createElement("div");
    presets.className = "dl-widget__controls";
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
    presets.appendChild(seg);
    const controls = document.createElement("div");
    controls.className = "dl-widget__controls";
    const inputs = {};
    function slider(label, key, min, max, step) {
        const wrap = document.createElement("label");
        const span = document.createElement("span");
        span.textContent = label;
        const input = document.createElement("input");
        Object.assign(input, { type: "range", min, max, step, value: state[key] });
        const out = document.createElement("output");
        out.style.minWidth = "2.4em";
        input.addEventListener("input", () => {
            state[key] = parseFloat(input.value);
            render();
        });
        wrap.append(span, input, out);
        inputs[key] = [input, out];
        return wrap;
    }
    controls.append(slider("Shift in time", "shift", -3, 5, 0.5), slider("Stretch in time", "stretch", 0.6, 1.8, 0.1));
    const svg = document.createElementNS(NS, "svg");
    svg.setAttribute("role", "img");
    const legend = document.createElement("div");
    legend.className = "dl-widget__legend";
    const caption = document.createElement("p");
    caption.className = "dl-widget__caption";
    caption.setAttribute("aria-live", "polite");
    el.append(presets, controls, svg, legend, caption);

    function node(tag, attrs, parent) {
        const n = document.createElementNS(NS, tag);
        for (const k in attrs) n.setAttribute(k, attrs[k]);
        if (parent) parent.appendChild(n);
        return n;
    }

    let W = 640;
    function render() {
        const c = api.colors();
        const B = seriesB();
        const res = dtw(A, B);
        const lock = A.reduce((s, v, i) => s + Math.abs(v - B[i]), 0);
        inputs.shift[0].value = state.shift;
        inputs.shift[1].textContent = (state.shift > 0 ? "+" : state.shift < 0 ? "−" : "") + Math.abs(state.shift).toFixed(1);
        inputs.stretch[0].value = state.stretch;
        inputs.stretch[1].textContent = `${state.stretch.toFixed(1)}×`;
        for (const [i, b] of [...seg.children].entries()) {
            const v = PRESETS[i][1];
            b.setAttribute("aria-pressed", String(v.shift === state.shift && v.stretch === state.stretch));
        }

        W = Math.max(280, Math.min(720, Math.round(el.clientWidth - 2) || 640));
        const wide = W >= 600;
        // Wide: series on the left, matrix on the right. Narrow: stacked.
        const sw = wide ? W * 0.56 : W, sh = wide ? 230 : 190;
        const cell = Math.floor(Math.min(wide ? (W - sw - 34) / N : (W - 40) / N, 15));
        const mw = cell * N, mx = wide ? sw + 26 : (W - mw) / 2 + 8, my = wide ? 16 : sh + 22;
        const H = wide ? Math.max(sh, my + mw + 18) : my + mw + 18;
        svg.setAttribute("viewBox", `0 0 ${W} ${H}`);
        svg.setAttribute("aria-label", `Two series of ${N} points. DTW distance ${res.distance.toFixed(2)}, point-by-point distance ${lock.toFixed(2)}.`);
        svg.innerHTML = "";

        // Series: A above, B below, joined where DTW aligns them.
        const x = (i) => 14 + ((sw - 28) * i) / (N - 1);
        const band = (sh - 20) / 2;
        const yA = (v) => 10 + band * 0.75 - v * band * 0.6;
        const yB = (v) => 10 + band + band * 0.75 - v * band * 0.6;
        for (const [i, j] of res.path) {
            node("line", { x1: x(i), y1: yA(A[i]), x2: x(j), y2: yB(B[j]), stroke: c.accent, "stroke-opacity": 0.45, "stroke-width": 1 }, svg);
        }
        node("polyline", { points: A.map((v, i) => `${x(i)},${yA(v)}`).join(" "), fill: "none", stroke: c.train, "stroke-width": 1.8 }, svg);
        node("polyline", { points: B.map((v, j) => `${x(j)},${yB(v)}`).join(" "), fill: "none", stroke: c.fg, "stroke-width": 1.8 }, svg);
        A.forEach((v, i) => node("circle", { cx: x(i), cy: yA(v), r: 2.6, fill: c.train }, svg));
        B.forEach((v, j) => node("circle", { cx: x(j), cy: yB(v), r: 2.6, fill: c.fg }, svg));
        node("text", { x: 14, y: 10, "font-size": 10, fill: c.fgLight }, svg).textContent = "series 1";
        node("text", { x: 14, y: 10 + band + 4, "font-size": 10, fill: c.fgLight }, svg).textContent = "series 2";

        // Accumulated cost matrix, with series 1 down the side and series 2 along the top.
        let top = 0;
        for (const row of res.D) for (const v of row) top = Math.max(top, v);
        for (let i = 0; i < N; i++) {
            for (let j = 0; j < N; j++) {
                node("rect", { x: mx + j * cell, y: my + i * cell, width: cell - 1, height: cell - 1, fill: c.fg, "fill-opacity": (0.04 + 0.5 * (res.D[i][j] / top)).toFixed(3) }, svg);
            }
        }
        for (const [i, j] of res.path) {
            node("rect", { x: mx + j * cell, y: my + i * cell, width: cell - 1, height: cell - 1, fill: c.accent }, svg);
        }
        node("text", { x: mx, y: my - 5, "font-size": 10, fill: c.fgLight }, svg).textContent = "accumulated cost: series 2 →";
        node("text", { x: mx - 6, y: my + mw, "font-size": 10, fill: c.fgLight, transform: `rotate(-90 ${mx - 6} ${my + mw})` }, svg).textContent = "← series 1";

        legend.innerHTML = "";
        for (const [t, col] of [["series 1", c.train], ["series 2", c.fg], ["pairs aligned by DTW, and the path", c.accent]]) {
            const s = document.createElement("span");
            s.style.setProperty("--swatch", col);
            s.textContent = t;
            legend.appendChild(s);
        }
        const head = `DTW distance ${res.distance.toFixed(2)}; comparing point by point gives ${lock.toFixed(2)}.`;
        caption.textContent =
            lock < 1e-9 ? `${head} The two series are identical, so the path runs straight down the diagonal.`
            : `${head} The path leaves the diagonal to match each point with the one that looks like it, even when it comes earlier or later, so the same shape at a different pace costs much less.`;
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
