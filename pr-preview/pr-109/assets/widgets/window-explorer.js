/*
 * Window explorer: how time delay embedding (rolling) turns one series into
 * rows of fixed-length inputs and targets. Choose the history length p and
 * the horizon H, then step or play through the rows.
 *
 *     <div class="dl-widget" data-widget="window-explorer" markdown>
 *     Fallback text for readers without JavaScript.
 *     </div>
 *
 * Optional data attributes: data-steps (20), data-history (4), data-horizon (1),
 * data-max-history (8), data-max-horizon (4). Setting data-gap adds a gap
 * slider between input and target (data-max-gap, default 4); the three maxima
 * must add up to at most data-steps. data-labels="dataset" speaks the language
 * of the `DataFrameDataset` class: the series is y = index, rows count from 0
 * and the sliders are called history_length, horizon and gap.
 */
dlWidgets.register("window-explorer", (el, api) => {
    const NS = "http://www.w3.org/2000/svg";
    const int = (v, d) => (v === undefined ? d : parseInt(v, 10));
    const N = int(el.dataset.steps, 20);
    const hasGap = el.dataset.gap !== undefined;
    const asDataset = el.dataset.labels === "dataset";
    const max = {
        p: int(el.dataset.maxHistory, 8),
        h: int(el.dataset.maxHorizon, 4),
        g: hasGap ? int(el.dataset.maxGap, 4) : 0,
    };
    const state = {
        p: int(el.dataset.history, 4),
        h: int(el.dataset.horizon, 1),
        g: int(el.dataset.gap, 0),
        row: 0,
        playing: false,
    };

    // A deterministic toy series: a slow wave with a little jitter. In dataset
    // mode the series is its own index, like the example table on the page.
    const series = Array.from({ length: N }, (_, t) => {
        if (asDataset) return t;
        const r = Math.sin(t * 78.233) * 43758.5453;
        return 5 + 2.2 * Math.sin((2 * Math.PI * t) / 9) + 0.08 * t + 0.5 * (r - Math.floor(r) - 0.5);
    });
    const rows = () => N - state.p - state.h - state.g + 1;

    // ---- DOM ---------------------------------------------------------------
    el.innerHTML = "";
    const controls = document.createElement("div");
    controls.className = "dl-widget__controls";

    function slider(label, key, min, max) {
        const wrap = document.createElement("label");
        const span = document.createElement("span");
        const input = document.createElement("input");
        const out = document.createElement("output");
        Object.assign(input, { type: "range", min, max, value: state[key] });
        span.textContent = label;
        out.textContent = state[key];
        input.addEventListener("input", () => {
            state[key] = parseInt(input.value, 10);
            out.textContent = input.value;
            state.row = Math.min(state.row, rows() - 1);
            render();
        });
        wrap.append(span, input, out);
        return wrap;
    }

    const playBtn = document.createElement("button");
    playBtn.type = "button";
    playBtn.className = "dl-widget__btn";
    playBtn.addEventListener("click", () => setPlaying(!state.playing));
    const stepBtn = document.createElement("button");
    stepBtn.type = "button";
    stepBtn.className = "dl-widget__btn";
    stepBtn.textContent = asDataset ? "Next data point" : "Next row";
    stepBtn.addEventListener("click", () => {
        setPlaying(false);
        advance();
    });
    controls.append(
        slider(asDataset ? "history_length" : "History p", "p", 1, max.p),
        slider(asDataset ? "horizon" : "Horizon H", "h", 1, max.h),
    );
    if (hasGap) controls.append(slider(asDataset ? "gap" : "Gap", "g", 0, max.g));
    controls.append(playBtn, stepBtn);

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
        W = Math.max(300, Math.min(720, Math.round(el.clientWidth - 2) || 640));
        const padL = W < 480 ? 26 : 34, padR = 8, chartH = W < 480 ? 110 : 130, gapY = 18;
        const n = rows();
        const cell = (W - padL - padR) / N;
        const rowH = Math.max(5, Math.min(12, 150 / n));
        const H = chartH + gapY + n * rowH + 8;
        svg.setAttribute("viewBox", `0 0 ${W} ${H}`);
        svg.innerHTML = "";

        const lo = Math.min(...series), hi = Math.max(...series);
        const x = (t) => padL + cell * (t + 0.5);
        const axisH = asDataset ? 14 : 0; // room for the row numbers
        const y = (v) => 10 + (chartH - 20 - axisH) * (1 - (v - lo) / (hi - lo));
        const inStart = state.row, inEnd = state.row + state.p - 1;
        const outStart = inEnd + 1 + state.g, outEnd = outStart + state.h - 1;

        // Highlighted windows on the series.
        node("rect", { x: padL + cell * inStart, y: 2, width: cell * state.p, height: chartH - 4, rx: 3, fill: c.train, "fill-opacity": 0.16 }, svg);
        if (state.g) node("rect", { x: padL + cell * (inEnd + 1), y: 2, width: cell * state.g, height: chartH - 4, rx: 3, fill: c.fgLighter, "fill-opacity": 0.14 }, svg);
        node("rect", { x: padL + cell * outStart, y: 2, width: cell * state.h, height: chartH - 4, rx: 3, fill: c.accent, "fill-opacity": 0.2 }, svg);
        node("polyline", { points: series.map((v, t) => `${x(t)},${y(v)}`).join(" "), fill: "none", stroke: c.fgLighter, "stroke-width": 1.2 }, svg);
        series.forEach((v, t) => {
            const fill = t >= inStart && t <= inEnd ? c.train : t >= outStart && t <= outEnd ? c.accent : c.fgLighter;
            node("circle", { cx: x(t), cy: y(v), r: W < 480 ? 3 : 3.6, fill }, svg);
            if (asDataset) node("text", { x: x(t), y: chartH - 7, "text-anchor": "middle", "font-size": W < 480 ? 8 : 9, fill: c.fgLight }, svg).textContent = t;
        });

        // One strip per row of the embedding table.
        const top = chartH + gapY;
        node("text", { x: padL, y: top - 5, "font-size": 10, fill: c.fgLight }, svg).textContent = asDataset ? "data points of the dataset" : "rows of the table";
        for (let i = 0; i < n; i++) {
            const yy = top + i * rowH;
            const g = node("g", { style: "cursor: pointer" }, svg);
            node("rect", { x: 0, y: yy, width: W, height: rowH, fill: i === state.row ? c.fgLightest : "transparent" }, g);
            node("rect", { x: padL + cell * i + 0.5, y: yy + 1, width: cell * state.p - 1, height: rowH - 2, rx: 1.5, fill: c.train, "fill-opacity": i === state.row ? 1 : 0.4 }, g);
            if (state.g) node("rect", { x: padL + cell * (i + state.p) + 0.5, y: yy + rowH / 2 - 0.5, width: cell * state.g - 1, height: 1, fill: c.fgLighter }, g);
            node("rect", { x: padL + cell * (i + state.p + state.g) + 0.5, y: yy + 1, width: cell * state.h - 1, height: rowH - 2, rx: 1.5, fill: c.accent, "fill-opacity": i === state.row ? 1 : 0.4 }, g);
            if (i === state.row && rowH >= 9) {
                node("text", { x: padL - 5, y: yy + rowH - 2, "text-anchor": "end", "font-size": 9, fill: c.fgLight }, g).textContent = asDataset ? i : i + 1;
            }
            g.addEventListener("click", () => {
                setPlaying(false);
                state.row = i;
                render();
            });
        }

        legend.innerHTML = "";
        const swatches = asDataset
            ? [["x: history_length rows", c.train], ["y: horizon rows", c.accent]]
            : [["input: p past values", c.train], ["target: next H values", c.accent]];
        if (hasGap) swatches.push([asDataset ? "gap: rows skipped in between" : "gap: values skipped in between", c.fgLighter]);
        for (const [label, col] of swatches) {
            const span = document.createElement("span");
            span.style.setProperty("--swatch", col);
            span.textContent = label;
            legend.appendChild(span);
        }
        const fmt = (a, b) => (a === b ? `y${sub(a + 1)}` : `y${sub(a + 1)}…y${sub(b + 1)}`);
        const gapTerm = hasGap ? ` − ${state.g}` : "";
        if (asDataset) {
            // The series is its own index, so these are the values as well.
            const vals = (a, b) => (b - a < 5 ? `[${series.slice(a, b + 1).join(", ")}]` : `[${a}, ${a + 1}, …, ${b}]`);
            const skipped = state.g ? `, with ${state.g} row${state.g > 1 ? "s" : ""} skipped in between` : "";
            svg.setAttribute("aria-label", `Data point ${state.row} of ${n}: x is rows ${inStart} to ${inEnd}, y is rows ${outStart} to ${outEnd}.`);
            caption.textContent =
                `ds[${state.row}]: x = ${vals(inStart, inEnd)} and y = ${vals(outStart, outEnd)}${skipped}. ` +
                `len(ds) = ${N} − ${state.p} − ${state.h}${gapTerm} + 1 = ${n} data point${n > 1 ? "s" : ""}.`;
        } else {
            svg.setAttribute("aria-label", `Row ${state.row + 1} of ${n}: inputs ${fmt(inStart, inEnd)}, targets ${fmt(outStart, outEnd)}.`);
            caption.textContent =
                `Row ${state.row + 1} of ${n}: the model reads ${fmt(inStart, inEnd)} and predicts ${fmt(outStart, outEnd)}. ` +
                `A series of ${N} points gives ${N} − ${state.p} − ${state.h}${gapTerm} + 1 = ${n} input–target rows, each with ${state.p} input columns.`;
        }
        playBtn.textContent = state.playing ? "Pause" : "Play";
    }

    function sub(k) {
        return String(k).replace(/\d/g, (d) => "₀₁₂₃₄₅₆₇₈₉"[d]);
    }

    let timer = null;
    function advance() {
        state.row = (state.row + 1) % rows();
        render();
    }
    function setPlaying(on) {
        state.playing = on;
        clearInterval(timer);
        if (on) timer = setInterval(advance, 700);
        render();
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
        clearInterval(timer);
        if (ro) ro.disconnect();
        offTheme();
    };
});
