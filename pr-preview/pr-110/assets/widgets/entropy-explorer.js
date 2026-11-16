/*
 * Entropy explorer: two distributions p and q over six classes. Each has a
 * peak and a sharpness. The widget shows both as bars, and the identity
 *
 *     H(p, q) = H(p) + D_KL(p ‖ q)
 *
 * as one bar split into its two parts, in nats.
 *
 *     <div class="dl-widget" data-widget="entropy-explorer" markdown>
 *     Fallback text for readers without JavaScript.
 *     </div>
 */
dlWidgets.register("entropy-explorer", (el, api) => {
    const NS = "http://www.w3.org/2000/svg";
    const K = 6;
    const state = { pm: 2, ps: 1, qm: 3.5, qs: 0.6 };
    const PRESETS = [
        ["q equals p", { pm: 2, ps: 1, qm: 2, qs: 1 }],
        ["p is certain", { pm: 2, ps: 6, qm: 3.5, qs: 0.6 }],
        ["p is uniform", { pm: 2, ps: 0, qm: 3.5, qs: 0.6 }],
        ["q misses p", { pm: 1, ps: 2, qm: 4, qs: 2 }],
    ];

    function dist(m, s) {
        const w = Array.from({ length: K }, (_, i) => Math.exp(-s * (i - m) * (i - m)));
        const z = w.reduce((a, b) => a + b, 0);
        return w.map((v) => v / z);
    }
    function measures(p, q) {
        let h = 0, hpq = 0;
        for (let i = 0; i < K; i++) {
            if (p[i] > 0) {
                h -= p[i] * Math.log(p[i]);
                hpq -= p[i] * Math.log(Math.max(q[i], 1e-300));
            }
        }
        return { h, hpq, kl: Math.max(0, hpq - h) };
    }

    el.innerHTML = "";
    const presets = document.createElement("div");
    presets.className = "dl-widget__controls";
    const seg = document.createElement("div");
    seg.className = "dl-widget__seg";
    seg.style.flexWrap = "wrap";
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
        out.style.minWidth = "2.2em";
        input.addEventListener("input", () => {
            state[key] = parseFloat(input.value);
            render();
        });
        wrap.append(span, input, out);
        inputs[key] = [input, out];
        return wrap;
    }
    controls.append(
        slider("Peak of p", "pm", 0, K - 1, 0.1),
        slider("Sharpness of p", "ps", 0, 6, 0.1),
        slider("Peak of q", "qm", 0, K - 1, 0.1),
        slider("Sharpness of q", "qs", 0, 6, 0.1)
    );
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
        const p = dist(state.pm, state.ps), q = dist(state.qm, state.qs);
        const m = measures(p, q);
        for (const key in inputs) {
            inputs[key][0].value = state[key];
            inputs[key][1].textContent = state[key].toFixed(1);
        }
        for (const [i, b] of [...seg.children].entries()) {
            const v = PRESETS[i][1];
            b.setAttribute("aria-pressed", String(Object.keys(v).every((k) => v[k] === state[k])));
        }

        W = Math.max(280, Math.min(720, Math.round(el.clientWidth - 2) || 640));
        const padL = 8, padR = 8, barsH = W < 480 ? 120 : 140, gap = 34, rowH = 22, H = barsH + 16 + gap + rowH * 2 + 18;
        svg.setAttribute("viewBox", `0 0 ${W} ${H}`);
        svg.setAttribute("aria-label", `Two distributions over six classes. Entropy of p ${m.h.toFixed(2)}, cross entropy ${m.hpq.toFixed(2)}, KL divergence ${m.kl.toFixed(2)} nats.`);
        svg.innerHTML = "";

        // Bars: p and q side by side for each class.
        const slot = (W - padL - padR) / K, bw = Math.min(26, slot * 0.32);
        const top = Math.max(...p, ...q, 0.3);
        const y = (v) => barsH - (barsH - 12) * (v / top);
        node("line", { x1: padL, x2: W - padR, y1: barsH, y2: barsH, stroke: c.fgLighter }, svg);
        for (let i = 0; i < K; i++) {
            const cx = padL + slot * (i + 0.5);
            node("rect", { x: cx - bw - 1, y: y(p[i]), width: bw, height: barsH - y(p[i]), fill: c.train, rx: 1.5 }, svg);
            node("rect", { x: cx + 1, y: y(q[i]), width: bw, height: barsH - y(q[i]), fill: c.fgLight, rx: 1.5 }, svg);
            node("text", { x: cx, y: barsH + 12, "text-anchor": "middle", "font-size": 10, fill: c.fgLight }, svg).textContent = `class ${i + 1}`;
        }

        // The decomposition: cross entropy = entropy + KL, on one scale.
        const y0 = barsH + 16 + gap, labelW = W < 480 ? 62 : 110;
        const span = W - padL - padR - labelW - 44;
        const full = Math.max(m.hpq, Math.log(K), 1e-9) * 1.02;
        const xw = (v) => (span * Math.min(v, full)) / full;
        const x0 = padL + labelW;
        const label = (t, yy) => (node("text", { x: padL, y: yy, "font-size": 10, fill: c.fgLight }, svg).textContent = t);
        label(W < 480 ? "H(p)" : "entropy H(p)", y0 + 14);
        node("rect", { x: x0, y: y0, width: Math.max(xw(m.h), 0.5), height: rowH - 4, fill: c.train, rx: 2 }, svg);
        node("text", { x: x0 + xw(m.h) + 5, y: y0 + 14, "font-size": 10, fill: c.fg }, svg).textContent = m.h.toFixed(2);
        label(W < 480 ? "H(p, q)" : "cross entropy H(p, q)", y0 + rowH + 14);
        node("rect", { x: x0, y: y0 + rowH, width: Math.max(xw(m.h), 0.5), height: rowH - 4, fill: c.train, rx: 2 }, svg);
        node("rect", { x: x0 + xw(m.h), y: y0 + rowH, width: Math.max(xw(m.hpq) - xw(m.h), 0), height: rowH - 4, fill: c.accent, rx: 2 }, svg);
        node("text", { x: x0 + xw(m.hpq) + 5, y: y0 + rowH + 14, "font-size": 10, fill: c.fg }, svg).textContent = m.hpq > full ? `${m.hpq.toFixed(1)} →` : m.hpq.toFixed(2);
        // The largest entropy six classes can have.
        const xm = x0 + xw(Math.log(K));
        node("line", { x1: xm, x2: xm, y1: y0 - 6, y2: y0 + rowH * 2, stroke: c.fgLighter, "stroke-dasharray": "3 3" }, svg);
        node("text", { x: xm, y: y0 - 9, "text-anchor": "middle", "font-size": 9, fill: c.fgLight }, svg).textContent = `ln 6 = ${Math.log(K).toFixed(2)}`;

        legend.innerHTML = "";
        for (const [t, col] of [["p, and its entropy", c.train], ["q", c.fgLight], ["KL divergence from p to q", c.accent]]) {
            const s = document.createElement("span");
            s.style.setProperty("--swatch", col);
            s.textContent = t;
            legend.appendChild(s);
        }

        const sum = `H(p, q) = H(p) + D(p ‖ q): ${m.hpq.toFixed(2)} = ${m.h.toFixed(2)} + ${m.kl.toFixed(2)} nats.`;
        let why;
        if (m.kl < 0.005) why = "q matches p, so the KL part vanishes and the cross entropy is as small as it can be for this p.";
        else if (m.h < 0.05) why = "p is almost certain of one class, so its entropy is close to 0 and nearly all of the cross entropy is the KL part: what classification losses measure against a one-hot label.";
        else if (Math.log(K) - m.h < 0.005) why = "p is uniform, the most uncertain it can be, so its entropy reaches ln 6.";
        else why = "The blue part depends on p alone; the orange part is the price of describing p with q.";
        caption.textContent = `${sum} ${why}`;
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
