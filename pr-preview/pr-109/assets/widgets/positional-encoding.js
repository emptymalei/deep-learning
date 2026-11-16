/*
 * Positional encoding explorer: the sinusoidal encoding of the transformer,
 *
 *     PE(pos, 2i)   = sin(pos / 10000^(2i/d))
 *     PE(pos, 2i+1) = cos(pos / 10000^(2i/d)),
 *
 * as a heatmap (one column per position, one row per dimension) and, below
 * it, how similar every position's encoding is to the selected position.
 *
 *     <div class="dl-widget" data-widget="positional-encoding" markdown>
 *     Fallback text for readers without JavaScript.
 *     </div>
 *
 * Optional data attributes: data-positions (40), data-dims (32).
 */
dlWidgets.register("positional-encoding", (el, api) => {
    const NS = "http://www.w3.org/2000/svg";
    const P = parseInt(el.dataset.positions || "40", 10);
    const state = { pos: 10, d: parseInt(el.dataset.dims || "32", 10) };

    const pe = (pos, j, d) => {
        const angle = pos / Math.pow(10000, (2 * Math.floor(j / 2)) / d);
        return j % 2 === 0 ? Math.sin(angle) : Math.cos(angle);
    };
    // Cosine similarity of two encodings; every encoding has squared norm d/2.
    function similarity(a, b, d) {
        let s = 0;
        for (let j = 0; j < d; j++) s += pe(a, j, d) * pe(b, j, d);
        return s / (d / 2);
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
        out.textContent = state[key];
        input.addEventListener("input", () => {
            state[key] = parseInt(input.value, 10);
            out.textContent = input.value;
            render();
        });
        wrap.append(span, input, out);
        return { wrap, input, out };
    }
    const posSlider = slider("Position", "pos", 0, P - 1, 1);
    controls.append(posSlider.wrap, slider("Dimensions d", "d", 8, 64, 8).wrap);

    const svg = document.createElementNS(NS, "svg");
    svg.setAttribute("role", "img");
    const legend = document.createElement("div");
    legend.className = "dl-widget__legend";
    const caption = document.createElement("p");
    caption.className = "dl-widget__caption";
    caption.setAttribute("aria-live", "polite");
    el.append(controls, svg, legend, caption);

    function node(name, attrs, parent) {
        const n = document.createElementNS(NS, name);
        for (const k in attrs) n.setAttribute(k, attrs[k]);
        if (parent) parent.appendChild(n);
        return n;
    }

    let W = 640;
    function render() {
        const c = api.colors();
        const d = state.d;
        W = Math.max(300, Math.min(720, Math.round(el.clientWidth - 2) || 640));
        const padL = W < 480 ? 30 : 38, padR = 6;
        const cw = (W - padL - padR) / P;
        const heatH = W < 480 ? 150 : 190;
        const ch = heatH / d;
        const simTop = heatH + 34, simH = 70;
        const H = simTop + simH + 22;
        svg.setAttribute("viewBox", `0 0 ${W} ${H}`);
        svg.innerHTML = "";

        // Heatmap: positive values in the accent colour, negative in blue.
        const heat = node("g", { "shape-rendering": "crispEdges" }, svg);
        for (let pos = 0; pos < P; pos++) {
            for (let j = 0; j < d; j++) {
                const v = pe(pos, j, d);
                node("rect", {
                    x: padL + pos * cw, y: j * ch, width: cw + 0.3, height: ch + 0.3,
                    fill: v >= 0 ? c.accent : c.train, "fill-opacity": Math.abs(v).toFixed(3),
                }, heat);
            }
        }
        // Selected position and click targets.
        node("rect", { x: padL + state.pos * cw, y: 0, width: cw, height: heatH, fill: "none", stroke: c.fg, "stroke-width": 1.5 }, svg);
        const hit = node("rect", { x: padL, y: 0, width: W - padL - padR, height: heatH, fill: "transparent", style: "cursor: pointer" }, svg);
        hit.addEventListener("click", (ev) => {
            const r = svg.getBoundingClientRect();
            const px = ((ev.clientX - r.left) * W) / r.width;
            state.pos = Math.max(0, Math.min(P - 1, Math.floor((px - padL) / cw)));
            posSlider.input.value = state.pos;
            posSlider.out.textContent = state.pos;
            render();
        });
        const ylab = (text, y) => (node("text", { x: padL - 5, y, "text-anchor": "end", "font-size": 9, fill: c.fgLight }, svg).textContent = text);
        ylab("dim 0", 8);
        ylab(`dim ${d - 1}`, heatH - 1);
        for (const pos of [0, 10, 20, 30, P - 1]) {
            node("text", { x: padL + (pos + 0.5) * cw, y: heatH + 11, "text-anchor": "middle", "font-size": 9, fill: c.fgLight }, svg).textContent = pos;
        }
        node("text", { x: padL, y: simTop - 7, "font-size": 10, fill: c.fgLight }, svg).textContent = `similarity to position ${state.pos}`;

        // Similarity of every position to the selected one.
        const sims = Array.from({ length: P }, (_, q) => similarity(state.pos, q, d));
        const y = (v) => simTop + simH * (1 - (v + 0.25) / 1.25);
        node("line", { x1: padL, x2: W - padR, y1: y(0), y2: y(0), stroke: c.fgLighter, "stroke-dasharray": "3 3" }, svg);
        ylab("1", y(1) + 3);
        ylab("0", y(0) + 3);
        sims.forEach((s, q) => {
            node("rect", {
                x: padL + q * cw + 0.5, width: Math.max(1, cw - 1),
                y: Math.min(y(s), y(0)), height: Math.abs(y(s) - y(0)),
                fill: q === state.pos ? c.fg : c.accent, "fill-opacity": q === state.pos ? 1 : 0.7,
            }, svg);
        });
        node("text", { x: padL + (W - padL - padR) / 2, y: H - 4, "text-anchor": "middle", "font-size": 9, fill: c.fgLight }, svg).textContent = "position";

        legend.innerHTML = "";
        for (const [label, col] of [["positive values", c.accent], ["negative values", c.train]]) {
            const span = document.createElement("span");
            span.style.setProperty("--swatch", col);
            span.textContent = label;
            legend.appendChild(span);
        }
        const near = state.pos + 1 < P ? state.pos + 1 : state.pos - 1;
        const far = state.pos + 10 < P ? state.pos + 10 : state.pos - 10;
        svg.setAttribute("aria-label", `Sinusoidal positional encodings for ${P} positions and ${d} dimensions; position ${state.pos} selected.`);
        caption.textContent =
            `Each column is the ${d}-number encoding of one position. The top rows flip quickly from position to position and the bottom rows change slowly, ` +
            `so together they give every position its own pattern. Similarity to position ${state.pos}: ` +
            `${similarity(state.pos, near, d).toFixed(2)} for its neighbour, ${similarity(state.pos, far, d).toFixed(2)} ten steps away.`;
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
