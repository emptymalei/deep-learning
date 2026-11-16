/*
 * Metrics under a level shift: add the same constant c to every true value
 * and every forecast. The errors y - ŷ do not change, so metrics built only
 * from the errors (MAE, MSE, RMSE) are translation invariant. Metrics that
 * divide by the values (MAPE, sMAPE) or take their logarithm (RMSLE) are
 * not: they shrink as the level grows and blow up around zero.
 *
 *     <div class="dl-widget" data-widget="metrics-shift" markdown>
 *     Fallback text for readers without JavaScript.
 *     </div>
 *
 * sMAPE uses |y| + |ŷ| in the denominator, as the metrics playground does;
 * RMSLE is the RMSE of ln(1 + value) and needs every value above -1.
 */
dlWidgets.register("metrics-shift", (el, api) => {
    const NS = "http://www.w3.org/2000/svg";
    const TRUTH = [3.1, 4.6, 3.4, 5.7, 5.2, 6.4, 6.1, 7.6, 7.3, 8.2];
    const ERR = [2, -1.5, 2.5, -2, 1.5, -2.5, 2, -1.5, 2.5, -2]; // forecast − truth
    const FORECAST = TRUTH.map((y, i) => y + ERR[i]);
    const C_MIN = -12, C_MAX = 20, STEP = 0.5;
    const Y_MIN = -10, Y_MAX = 30; // fixed axis, so the series visibly moves
    const INVARIANT = ["MAE", "MSE", "RMSE"];
    const VARIANT = ["MAPE", "sMAPE", "RMSLE"];
    const PRESETS = [["c = 0", 0], ["Around zero", -5], ["High level", 20]];
    const state = { c: 0 };

    function metricsAt(c) {
        const n = TRUTH.length;
        let ae = 0, se = 0, ape = 0, sape = 0, sle = 0, logOk = true;
        for (let i = 0; i < n; i++) {
            const y = TRUTH[i] + c;
            const f = FORECAST[i] + c;
            const e = y - f;
            ae += Math.abs(e);
            se += e * e;
            ape += y === 0 ? Infinity : Math.abs(e / y);
            const denom = Math.abs(y) + Math.abs(f);
            sape += denom === 0 ? 0 : (2 * Math.abs(e)) / denom;
            if (y > -1 && f > -1) sle += (Math.log1p(y) - Math.log1p(f)) ** 2;
            else logOk = false;
        }
        return {
            MAE: ae / n,
            MSE: se / n,
            RMSE: Math.sqrt(se / n),
            MAPE: (100 * ape) / n,
            sMAPE: (100 * sape) / n,
            RMSLE: logOk ? Math.sqrt(sle / n) : NaN,
        };
    }

    // Every metric as a function of c, on a fine grid, for the small curves.
    const GRID = [];
    for (let k = 0; k <= (C_MAX - C_MIN) * 10; k++) {
        const c = Math.round((C_MIN + k / 10) * 10) / 10;
        GRID.push([c, metricsAt(c)]);
    }
    const AT_ZERO = metricsAt(0);
    // Top of each small curve: flat ones sit mid-height, sMAPE shows its bound,
    // and the two unbounded ones are cut off at four times their value at c = 0.
    const CAP = {
        MAE: 2 * AT_ZERO.MAE,
        MSE: 2 * AT_ZERO.MSE,
        RMSE: 2 * AT_ZERO.RMSE,
        MAPE: 4 * AT_ZERO.MAPE,
        sMAPE: 200,
        RMSLE: 4 * AT_ZERO.RMSLE,
    };

    // ---- DOM ---------------------------------------------------------------
    el.innerHTML = "";
    const style = document.createElement("style");
    style.textContent = `
.dl-shift { display: grid; grid-template-columns: minmax(0, 1fr); gap: .5rem .9rem; align-items: end; margin: .5rem 0 .2rem; }
.dl-shift--wide { grid-template-columns: repeat(2, minmax(0, 1fr)); }
.dl-shift__row { display: grid; grid-template-columns: repeat(3, minmax(0, 1fr)); gap: .4rem; }
.md-typeset .dl-shift__group { margin: 0 0 .25rem; font-size: .62rem; font-weight: 600; line-height: 1.3; color: var(--md-default-fg-color--light); }
.dl-shift__tile { min-width: 0; padding: .3rem .45rem; border-radius: .3rem; background: var(--md-default-fg-color--lightest); line-height: 1.25; }
.dl-shift__tile b { display: block; font-size: .6rem; font-weight: 600; color: var(--md-default-fg-color--light); }
.dl-shift__tile span { font-variant-numeric: tabular-nums; font-size: .78rem; }
.dl-shift__tile small { display: block; font-size: .58rem; color: var(--md-default-fg-color--light); }
.dl-shift__tile svg { margin: .15rem 0; overflow: hidden; }
.dl-widget__controls .dl-shift__slider input[type="range"] { width: 11rem; max-width: 45vw; }
.dl-shift__slider output { min-width: 3.2em; font-variant-numeric: tabular-nums; }`;

    const controls = document.createElement("div");
    controls.className = "dl-widget__controls";
    const wrap = document.createElement("label");
    wrap.className = "dl-shift__slider";
    const span = document.createElement("span");
    span.textContent = "Shift c";
    const input = document.createElement("input");
    Object.assign(input, { type: "range", min: C_MIN, max: C_MAX, step: STEP, value: state.c });
    const out = document.createElement("output");
    input.addEventListener("input", () => set(parseFloat(input.value)));
    wrap.append(span, input, out);

    const seg = document.createElement("div");
    seg.className = "dl-widget__seg";
    seg.setAttribute("role", "group");
    seg.setAttribute("aria-label", "Examples");
    const buttons = PRESETS.map(([label, c]) => {
        const b = document.createElement("button");
        b.type = "button";
        b.textContent = label;
        b.addEventListener("click", () => set(c));
        seg.appendChild(b);
        return [c, b];
    });
    controls.append(wrap, seg);

    const svg = document.createElementNS(NS, "svg");
    svg.setAttribute("role", "img");
    const legend = document.createElement("div");
    legend.className = "dl-widget__legend";
    const tiles = document.createElement("div");
    tiles.className = "dl-shift";
    const caption = document.createElement("p");
    caption.className = "dl-widget__caption";
    caption.setAttribute("aria-live", "polite");
    el.append(style, controls, svg, legend, tiles, caption);

    function set(c) {
        state.c = c;
        input.value = c;
        render();
    }

    function node(name, attrs, parent) {
        const n = document.createElementNS(NS, name);
        for (const k in attrs) n.setAttribute(k, attrs[k]);
        if (parent) parent.appendChild(n);
        return n;
    }

    const signed = (c) => (c > 0 ? "+" : c < 0 ? "−" : "") + Math.abs(c).toFixed(1);
    function fmt(key, v) {
        if (Number.isNaN(v)) return "undefined";
        if (!isFinite(v)) return "∞";
        if (key === "MAPE" || key === "sMAPE") {
            if (v >= 10000) return "> 9999%";
            return (v >= 100 ? v.toFixed(0) : v.toFixed(1)) + "%";
        }
        return v >= 10 ? v.toFixed(1) : v.toFixed(key === "RMSLE" ? 3 : 2);
    }

    // ---- Main chart: the two series at their shifted level ------------------
    let W = 640, H = 260;
    function drawChart(c) {
        const cs = api.colors();
        W = Math.max(280, Math.min(720, Math.round(el.clientWidth - 2) || 640));
        H = W < 480 ? 200 : 230;
        const padL = W < 480 ? 30 : 36, padR = 12, padT = 10, padB = 10;
        const xOf = (i) => padL + ((W - padL - padR) * (i + 0.5)) / TRUTH.length;
        const yOf = (v) => padT + ((H - padT - padB) * (Y_MAX - v)) / (Y_MAX - Y_MIN);
        svg.setAttribute("viewBox", `0 0 ${W} ${H}`);
        svg.setAttribute(
            "aria-label",
            `Truth and forecast for 10 time steps, both shifted by ${signed(c)}. The gaps between them are the same at every shift.`
        );
        svg.innerHTML = "";

        for (let v = Y_MIN; v <= Y_MAX; v += 10) {
            const zero = v === 0;
            node("line", { x1: padL, x2: W - padR, y1: yOf(v), y2: yOf(v), stroke: zero ? cs.fgLighter : cs.fgLightest, "stroke-width": zero ? 1.4 : 1 }, svg);
            node("text", { x: padL - 6, y: yOf(v) + 3, "text-anchor": "end", "font-size": 10, "font-weight": zero ? 700 : 400, fill: zero ? cs.fg : cs.fgLight }, svg).textContent = v;
        }

        // Where the truth was before the shift.
        if (c !== 0) {
            node("polyline", { points: TRUTH.map((y, i) => `${xOf(i)},${yOf(y)}`).join(" "), fill: "none", stroke: cs.fgLighter, "stroke-width": 1.2, "stroke-dasharray": "4 3" }, svg);
        }

        // Errors: the same length at every shift.
        TRUTH.forEach((y, i) => {
            node("line", { x1: xOf(i), x2: xOf(i), y1: yOf(y + c), y2: yOf(FORECAST[i] + c), stroke: cs.accent, "stroke-opacity": 0.5, "stroke-width": 6, "stroke-linecap": "round" }, svg);
        });
        node("polyline", { points: TRUTH.map((y, i) => `${xOf(i)},${yOf(y + c)}`).join(" "), fill: "none", stroke: cs.fg, "stroke-width": 1.5 }, svg);
        TRUTH.forEach((y, i) => node("circle", { cx: xOf(i), cy: yOf(y + c), r: 3.5, fill: cs.fg }, svg));
        FORECAST.forEach((f, i) => node("circle", { cx: xOf(i), cy: yOf(f + c), r: 4.5, fill: cs.bg, stroke: cs.accent, "stroke-width": 2.2 }, svg));

        legend.innerHTML = "";
        for (const [label, col] of [
            ["truth", cs.fg],
            ["forecast", cs.accent],
            ["truth at c = 0", cs.fgLighter],
        ]) {
            const s = document.createElement("span");
            s.style.setProperty("--swatch", col);
            s.textContent = label;
            legend.appendChild(s);
        }
    }

    // ---- Tiles: each metric's value, and its curve over the whole slider -----
    function drawTiles(c, m) {
        const cs = api.colors();
        tiles.innerHTML = "";
        tiles.classList.toggle("dl-shift--wide", W >= 600);
        const groups = [
            ["Errors only: the same at every level", INVARIANT, cs.train],
            ["Divide by the values or take logs: change with the level", VARIANT, cs.accent],
        ];
        for (const [title, keys, color] of groups) {
            const group = document.createElement("div");
            const head = document.createElement("p");
            head.className = "dl-shift__group";
            head.textContent = title;
            const row = document.createElement("div");
            row.className = "dl-shift__row";
            group.append(head, row);
            tiles.appendChild(group);
            for (const key of keys) {
                const tile = document.createElement("div");
                tile.className = "dl-shift__tile";
                const name = document.createElement("b");
                name.textContent = key;
                const value = document.createElement("span");
                value.textContent = fmt(key, m[key]);
                const spark = document.createElementNS(NS, "svg");
                spark.setAttribute("aria-hidden", "true");
                const note = document.createElement("small");
                if (INVARIANT.includes(key)) note.textContent = "unchanged";
                else if (Number.isNaN(m[key])) note.textContent = "values ≤ −1";
                else note.textContent = c === 0 ? "starting value" : `${fmt(key, AT_ZERO[key])} at c = 0`;
                tile.append(name, value, spark, note);
                row.appendChild(tile);
                drawSpark(spark, key, c, m[key], color, cs);
            }
        }
    }

    function drawSpark(spark, key, c, v, color, cs) {
        const w = Math.max(60, Math.round(spark.getBoundingClientRect().width) || 120);
        const h = 46, top = 4, base = h - 12;
        const xS = (cc) => 5 + ((w - 10) * (cc - C_MIN)) / (C_MAX - C_MIN);
        const yS = (vv) => base - (base - top) * Math.min(vv / CAP[key], 1.5);
        spark.setAttribute("viewBox", `0 0 ${w} ${h}`);

        // The stretch of c where the metric is undefined (RMSLE below −1).
        const firstOk = GRID.findIndex(([, mm]) => !Number.isNaN(mm[key]));
        if (firstOk > 0) {
            node("rect", { x: xS(C_MIN), y: top, width: xS(GRID[firstOk][0]) - xS(C_MIN), height: base - top, fill: cs.fgLightest }, spark);
        }
        node("line", { x1: xS(C_MIN), x2: xS(C_MAX), y1: base, y2: base, stroke: cs.fgLighter }, spark);
        const runs = []; // stretches of c where the metric has a value
        let run = null;
        for (const [cc, mm] of GRID) {
            const vv = mm[key];
            if (Number.isNaN(vv)) { run = null; continue; }
            if (!run) runs.push((run = []));
            run.push(`${xS(cc).toFixed(1)},${yS(vv).toFixed(1)}`);
        }
        const clip = "dl-shift-clip-" + key;
        const cp = node("clipPath", { id: clip }, spark);
        node("rect", { x: 0, y: top - 2, width: w, height: base - top + 2 }, cp);
        for (const pts of runs) {
            const x0 = pts[0].split(",")[0], x1 = pts[pts.length - 1].split(",")[0];
            if (!INVARIANT.includes(key)) {
                node("path", { d: `M${pts.join("L")}L${x1},${base}L${x0},${base}Z`, fill: color, "fill-opacity": 0.16, "clip-path": `url(#${clip})` }, spark);
            }
            node("path", { d: `M${pts.join("L")}`, fill: "none", stroke: color, "stroke-width": 1.8, "clip-path": `url(#${clip})` }, spark);
        }

        // Axis: both ends of the slider and c = 0.
        for (const [cc, anchor] of [[C_MIN, "start"], [0, "middle"], [C_MAX, "end"]]) {
            node("line", { x1: xS(cc), x2: xS(cc), y1: base, y2: base + 3, stroke: cs.fgLighter }, spark);
            node("text", { x: xS(cc), y: h - 1, "text-anchor": anchor, "font-size": 8, fill: cs.fgLight }, spark).textContent = cc < 0 ? "−" + -cc : cc;
        }
        // Where the slider is now.
        node("line", { x1: xS(c), x2: xS(c), y1: top - 2, y2: base, stroke: cs.fg, "stroke-opacity": 0.45 }, spark);
        if (!Number.isNaN(v)) {
            const over = !isFinite(v) || v > CAP[key];
            node("circle", { cx: xS(c), cy: over ? top : yS(v), r: 3.2, fill: over ? cs.bg : color, stroke: color, "stroke-width": 1.5 }, spark);
        }
    }

    function explain(c, m) {
        const p = (key) => fmt(key, m[key]);
        const p0 = (key) => fmt(key, AT_ZERO[key]);
        const lo = Math.min(...TRUTH, ...FORECAST) + c;
        const hi = Math.max(...TRUTH, ...FORECAST) + c;
        const nearZero = TRUTH.some((y) => Math.abs(y + c) < 1);
        const same = "Every error is as long as before, so MAE, MSE and RMSE have not moved.";
        if (c === 0) {
            return "This is the data as given. Move the slider: the errors keep their length, so MAE, MSE and RMSE stay where they are, while MAPE, sMAPE and RMSLE change.";
        }
        if (hi <= 0) {
            return `Shifted by ${signed(c)}: all values are now negative. ${same} MAPE divides by the size of the truth, so it is moderate again (${p("MAPE")}); RMSLE takes ln(1 + value) and has no value below −1.`;
        }
        if (nearZero || lo < 0) {
            const rmsle = Number.isNaN(m.RMSLE) ? " RMSLE takes ln(1 + value) and has no value once a value drops to −1 or below." : "";
            return `Shifted by ${signed(c)}: some values are now at or near zero. ${same} MAPE divides each error by a true value close to 0 and explodes (${p("MAPE")}, from ${p0("MAPE")}); sMAPE is pushed towards its bound of 200% (${p("sMAPE")}).${rmsle}`;
        }
        const dir = c > 0 ? ["fell", "larger"] : ["rose", "smaller"];
        return `Shifted by ${signed(c)}. ${same} MAPE ${dir[0]} from ${p0("MAPE")} to ${p("MAPE")}, only because the same errors are now divided by ${dir[1]} true values; sMAPE and RMSLE follow for the same reason.`;
    }

    function render() {
        const c = state.c;
        const m = metricsAt(c);
        out.textContent = signed(c);
        for (const [pc, b] of buttons) b.setAttribute("aria-pressed", String(pc === c));
        drawChart(c);
        drawTiles(c, m);
        caption.textContent = explain(c, m);
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
