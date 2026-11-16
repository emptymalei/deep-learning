/*
 * Two small explainers for tree splits.
 *
 * gini-explorer: a two-class dataset whose share of class 1 is set by a
 * slider, with its Gini impurity G = Σ p(i)(1 − p(i)) and the curve of G
 * over all shares.
 *
 * split-gain: a node of 40 records is split into a left and a right group.
 * Sliders set how many records are class 1, how many go left, and how many
 * of those are class 1. The widget shows the three groups, their entropies
 * (in nats) and the information gain: the entropy of the node minus the
 * size-weighted entropy of the two groups.
 *
 *     <div class="dl-widget" data-widget="gini-explorer" markdown>…</div>
 *     <div class="dl-widget" data-widget="split-gain" markdown>…</div>
 */
(function () {
    const NS = "http://www.w3.org/2000/svg";
    function node(tag, attrs, parent) {
        const n = document.createElementNS(NS, tag);
        for (const k in attrs) n.setAttribute(k, attrs[k]);
        if (parent) parent.appendChild(n);
        return n;
    }
    function slider(label, min, max, step, value, onInput) {
        const wrap = document.createElement("label");
        const span = document.createElement("span");
        span.textContent = label;
        const input = document.createElement("input");
        Object.assign(input, { type: "range", min, max, step, value });
        const out = document.createElement("output");
        out.style.minWidth = "2.4em";
        input.addEventListener("input", () => onInput(parseFloat(input.value)));
        wrap.append(span, input, out);
        return { wrap, input, out };
    }
    function watch(el, api, render, width) {
        const ro = "ResizeObserver" in window
            ? new ResizeObserver(() => {
                  const w = Math.round(el.clientWidth - 2);
                  if (w && Math.abs(w - width()) > 8) render();
              })
            : null;
        if (ro) ro.observe(el);
        const offTheme = api.onThemeChange(render);
        return () => {
            if (ro) ro.disconnect();
            offTheme();
        };
    }
    // A block of records as dots: the first `ones` are class 1.
    function dots(svg, x, y, cols, total, ones, size, c) {
        for (let i = 0; i < total; i++) {
            node("rect", {
                x: x + (i % cols) * size, y: y + Math.floor(i / cols) * size, width: size - 2, height: size - 2, rx: 2,
                fill: i < ones ? c.accent : c.train,
            }, svg);
        }
    }

    dlWidgets.register("gini-explorer", (el, api) => {
        const N = 20;
        let ones = 10, W = 640;
        const gini = (p) => 2 * p * (1 - p);
        el.innerHTML = "";
        const controls = document.createElement("div");
        controls.className = "dl-widget__controls";
        const s = slider("Records of class 1", 0, N, 1, ones, (v) => {
            ones = v;
            render();
        });
        controls.appendChild(s.wrap);
        const svg = document.createElementNS(NS, "svg");
        svg.setAttribute("role", "img");
        const legend = document.createElement("div");
        legend.className = "dl-widget__legend";
        const caption = document.createElement("p");
        caption.className = "dl-widget__caption";
        caption.setAttribute("aria-live", "polite");
        el.append(controls, svg, legend, caption);

        function render() {
            const c = api.colors();
            const p = ones / N, g = gini(p);
            s.input.value = ones;
            s.out.textContent = `${ones} of ${N}`;
            W = Math.max(280, Math.min(720, Math.round(el.clientWidth - 2) || 640));
            const size = Math.min(22, (W - 16) / N), dotsH = size + 14, plotH = 130, padL = 34, padR = 12, H = dotsH + plotH + 22;
            svg.setAttribute("viewBox", `0 0 ${W} ${H}`);
            svg.setAttribute("aria-label", `${ones} of ${N} records are class 1. Gini impurity ${g.toFixed(3)}.`);
            svg.innerHTML = "";
            dots(svg, (W - size * N) / 2, 2, N, N, ones, size, c);
            const x = (v) => padL + (W - padL - padR) * v;
            const y = (v) => dotsH + plotH - (plotH - 10) * (v / 0.5);
            for (const v of [0, 0.25, 0.5]) {
                node("line", { x1: padL, x2: W - padR, y1: y(v), y2: y(v), stroke: c.fgLightest }, svg);
                node("text", { x: padL - 5, y: y(v) + 3, "text-anchor": "end", "font-size": 10, fill: c.fgLight }, svg).textContent = v;
            }
            const pts = Array.from({ length: 101 }, (_, i) => `${x(i / 100).toFixed(1)},${y(gini(i / 100)).toFixed(1)}`);
            node("polyline", { points: pts.join(" "), fill: "none", stroke: c.fg, "stroke-width": 1.8 }, svg);
            node("line", { x1: x(p), x2: x(p), y1: y(0), y2: y(g), stroke: c.fgLighter, "stroke-dasharray": "3 3" }, svg);
            node("circle", { cx: x(p), cy: y(g), r: 5, fill: c.accent, stroke: c.bg, "stroke-width": 1.5 }, svg);
            for (const v of [0, 0.5, 1]) {
                node("text", { x: x(v), y: H - 4, "text-anchor": v === 0 ? "start" : v === 1 ? "end" : "middle", "font-size": 10, fill: c.fgLight }, svg).textContent = `${v * 100}% class 1`;
            }
            legend.innerHTML = "";
            for (const [t, col] of [["class 0", c.train], ["class 1", c.accent], ["Gini impurity over all mixes", c.fg]]) {
                const sp = document.createElement("span");
                sp.style.setProperty("--swatch", col);
                sp.textContent = t;
                legend.appendChild(sp);
            }
            const q = 1 - p;
            const calc = `G = ${q.toFixed(2)} × ${p.toFixed(2)} + ${p.toFixed(2)} × ${q.toFixed(2)} = ${g.toFixed(3)}`;
            caption.textContent =
                g === 0 ? `${calc}. Every record has the same class: the dataset is pure and the impurity is 0.`
                : ones === N / 2 ? `${calc}. An even mix is the most impure a two-class dataset can be.`
                : `${calc}. The closer the mix is to even, the higher the impurity.`;
        }
        render();
        return watch(el, api, render, () => W);
    });

    dlWidgets.register("split-gain", (el, api) => {
        const N = 40;
        const state = { ones: 20, left: 20, leftOnes: 16 };
        let W = 640;
        const entropy = (k, n) => {
            if (!n || k === 0 || k === n) return 0;
            const p = k / n;
            return -p * Math.log(p) - (1 - p) * Math.log(1 - p);
        };
        const PRESETS = [
            ["Perfect split", { ones: 20, left: 20, leftOnes: 20 }],
            ["Good split", { ones: 20, left: 20, leftOnes: 16 }],
            ["Useless split", { ones: 20, left: 20, leftOnes: 10 }],
        ];
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
        const sOnes = slider("Class 1 in the node", 0, N, 1, state.ones, (v) => { state.ones = v; render(); });
        const sLeft = slider("Records going left", 1, N - 1, 1, state.left, (v) => { state.left = v; render(); });
        const sLeftOnes = slider("Class 1 going left", 0, N, 1, state.leftOnes, (v) => { state.leftOnes = v; render(); });
        controls.append(sOnes.wrap, sLeft.wrap, sLeftOnes.wrap);
        const svg = document.createElementNS(NS, "svg");
        svg.setAttribute("role", "img");
        const legend = document.createElement("div");
        legend.className = "dl-widget__legend";
        const caption = document.createElement("p");
        caption.className = "dl-widget__caption";
        caption.setAttribute("aria-live", "polite");
        el.append(presets, controls, svg, legend, caption);

        function render() {
            const c = api.colors();
            // Keep the left group possible: it cannot hold more of a class than the node has.
            const lo = Math.max(0, state.left - (N - state.ones)), hi = Math.min(state.ones, state.left);
            state.leftOnes = Math.max(lo, Math.min(hi, state.leftOnes));
            const right = N - state.left, rightOnes = state.ones - state.leftOnes;
            const sM = entropy(state.ones, N), sL = entropy(state.leftOnes, state.left), sR = entropy(rightOnes, right);
            const after = (state.left / N) * sL + (right / N) * sR;
            const gain = Math.max(0, sM - after);
            sOnes.input.value = state.ones; sOnes.out.textContent = state.ones;
            sLeft.input.value = state.left; sLeft.out.textContent = state.left;
            Object.assign(sLeftOnes.input, { min: lo, max: hi });
            sLeftOnes.input.value = state.leftOnes; sLeftOnes.out.textContent = state.leftOnes;
            sLeftOnes.input.disabled = lo === hi;
            for (const [i, b] of [...seg.children].entries()) {
                const v = PRESETS[i][1];
                b.setAttribute("aria-pressed", String(Object.keys(v).every((k) => v[k] === state[k])));
            }

            W = Math.max(280, Math.min(720, Math.round(el.clientWidth - 2) || 640));
            const size = Math.min(16, (W - 40) / 22), cols = 10;
            const blockW = cols * size, rows = (n) => Math.ceil(n / cols);
            const topH = 14 + rows(N) * size, midGap = 30;
            const childH = 14 + Math.max(rows(state.left), rows(right)) * size;
            const barsY = topH + midGap + childH + 26, rowH = 20, H = barsY + rowH * 2 + 6;
            svg.setAttribute("viewBox", `0 0 ${W} ${H}`);
            svg.setAttribute("aria-label", `A node of ${N} records with ${state.ones} of class 1 is split into ${state.left} and ${right}. Information gain ${gain.toFixed(3)} nats.`);
            svg.innerHTML = "";
            const cx = W / 2, lx = W / 2 - blockW - 14, rx = W / 2 + 14;
            const text = (t, x, y, anchor = "start") => (node("text", { x, y, "text-anchor": anchor, "font-size": 10, fill: c.fgLight }, svg).textContent = t);
            text(`node: ${N} records, entropy ${sM.toFixed(3)}`, cx, 10, "middle");
            dots(svg, cx - blockW / 2, 14, cols, N, state.ones, size, c);
            const y1 = topH + 4, y2 = topH + midGap - 14;
            node("line", { x1: cx, y1, x2: lx + blockW / 2, y2, stroke: c.fgLighter }, svg);
            node("line", { x1: cx, y1, x2: rx + blockW / 2, y2, stroke: c.fgLighter }, svg);
            const cy = topH + midGap;
            text(`left: ${state.left}, entropy ${sL.toFixed(3)}`, lx + blockW / 2, cy - 4, "middle");
            text(`right: ${right}, entropy ${sR.toFixed(3)}`, rx + blockW / 2, cy - 4, "middle");
            dots(svg, lx, cy, cols, state.left, state.leftOnes, size, c);
            dots(svg, rx, cy, cols, right, rightOnes, size, c);

            // Entropy before and after, on one scale (ln 2 is the most a node can have).
            const labelW = W < 480 ? 46 : 150, x0 = 8 + labelW, span = W - x0 - 52, full = Math.log(2);
            const xw = (v) => (span * v) / full;
            text(W < 480 ? "before" : "entropy before the split", 8, barsY + 12);
            node("rect", { x: x0, y: barsY, width: Math.max(xw(sM), 0.5), height: rowH - 5, rx: 2, fill: c.fgLight }, svg);
            node("text", { x: x0 + xw(sM) + 5, y: barsY + 12, "font-size": 10, fill: c.fg }, svg).textContent = sM.toFixed(3);
            text(W < 480 ? "after" : "weighted entropy after", 8, barsY + rowH + 12);
            node("rect", { x: x0, y: barsY + rowH, width: Math.max(xw(after), 0.5), height: rowH - 5, rx: 2, fill: c.fgLight }, svg);
            node("rect", { x: x0 + xw(after), y: barsY + rowH, width: xw(gain), height: rowH - 5, rx: 2, fill: c.accent, "fill-opacity": 0.55 }, svg);
            node("text", { x: x0 + xw(sM) + 5, y: barsY + rowH + 12, "font-size": 10, fill: c.fg }, svg).textContent = after.toFixed(3);

            legend.innerHTML = "";
            for (const [t, col] of [["class 0", c.train], ["class 1", c.accent], ["entropy", c.fgLight], ["information gain", c.accent]]) {
                const sp = document.createElement("span");
                sp.style.setProperty("--swatch", col);
                sp.textContent = t;
                legend.appendChild(sp);
            }
            const calc = `Gain = ${sM.toFixed(3)} − (${state.left}/${N} × ${sL.toFixed(3)} + ${right}/${N} × ${sR.toFixed(3)}) = ${gain.toFixed(3)} nats.`;
            caption.textContent =
                sM === 0 ? `${calc} The node is already pure, so there is nothing to gain.`
                : after < 1e-9 ? `${calc} Both groups are pure: the split removes all of the entropy.`
                : gain < 0.002 ? `${calc} Both groups have about the same mix as the node, so the split tells us almost nothing.`
                : `${calc} The purer the two groups, the more entropy the split removes.`;
        }
        render();
        return watch(el, api, render, () => W);
    });
})();
