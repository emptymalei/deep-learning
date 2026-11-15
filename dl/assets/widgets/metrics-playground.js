/*
 * Metrics playground: pick a scenario, move its slider, and watch MAE, MSE,
 * RMSE, MAPE and sMAPE react. Each scenario turns one knob:
 *
 *   Error size    every error is scaled by a factor
 *   One big miss  one forecast misses by a chosen amount
 *   Bias          every forecast is shifted by the same amount
 *   Near zero     the smallest true value moves towards zero, errors stay
 *                 the same
 *
 * Under each metric's value, a small chart shows that metric over the whole
 * range of the slider, with a marker at the current setting. The forecast
 * points can also be dragged (or moved with the arrow keys); the curves then
 * include those changes.
 *
 *     <div class="dl-widget" data-widget="metrics-playground" markdown>
 *     Fallback text for readers without JavaScript.
 *     </div>
 */
dlWidgets.register("metrics-playground", (el, api) => {
    const NS = "http://www.w3.org/2000/svg";
    const KEYS = ["MAE", "MSE", "RMSE", "MAPE", "sMAPE"];
    const BASE_TRUTH = [12, 14, 13, 17, 16, 19, 18, 21, 20, 23];
    // The value at ZERO_AT is set by the slider (0.02 to 2). Every other value
    // is above 2, so the slider's value is the smallest one at every setting.
    const NEAR_ZERO = [2.4, 3.2, 0.06, 4.0, 2.6, 3.8, 2.2, 4.4, 3.0, 4.6];
    const WIGGLE = [0.8, -0.6, 1.0, -0.9, 0.5, -0.7, 0.9, -0.5, 0.7, -0.8];
    const MISS_AT = 6; // which step carries the big miss
    const ZERO_AT = 2; // which true value the Near zero slider moves
    const GRID = 80; // points per small curve

    const signed = (v, d = 1) => (v > 0 ? "+" : v < 0 ? "−" : "") + Math.abs(v).toFixed(d);
    const PRESETS = {
        size: {
            label: "Error size",
            slider: "Error size",
            min: 0, max: 4, step: 0.1, start: 1,
            show: (p) => `${p.toFixed(1)}×`,
            truth: () => BASE_TRUTH,
            forecast: (p) => BASE_TRUTH.map((y, i) => y + p * WIGGLE[i]),
        },
        miss: {
            label: "One big miss",
            slider: "Size of the miss",
            min: 0, max: 15, step: 0.5, start: 9,
            show: (p) => p.toFixed(1),
            truth: () => BASE_TRUTH,
            forecast: (p) => BASE_TRUTH.map((y, i) => y + (i === MISS_AT ? p : WIGGLE[i])),
        },
        bias: {
            label: "Bias",
            slider: "Bias",
            min: -8, max: 8, step: 0.5, start: 3,
            show: (p) => signed(p),
            truth: () => BASE_TRUTH,
            forecast: (p) => BASE_TRUTH.map((y) => y + p),
        },
        zero: {
            label: "Near zero",
            slider: "Smallest true value",
            min: 0.02, max: 2, step: 0.02, start: 0.06,
            show: (p) => p.toFixed(2),
            truth: (p) => NEAR_ZERO.map((y, i) => (i === ZERO_AT ? p : y)),
            forecast: (p) => NEAR_ZERO.map((y, i) => (i === ZERO_AT ? p : y) + 0.5),
        },
    };

    let name = "size";
    let preset = PRESETS[name];
    let param = preset.start;
    let deltas = BASE_TRUTH.map(() => 0); // what the reader dragged, per point
    let start = null; // metrics of the scenario as loaded
    let yMin = 0;
    let yMax = 1;

    function metrics(t, f) {
        const n = t.length;
        let ae = 0, se = 0, ape = 0, sape = 0;
        for (let i = 0; i < n; i++) {
            const e = t[i] - f[i];
            ae += Math.abs(e);
            se += e * e;
            ape += t[i] === 0 ? Infinity : Math.abs(e / t[i]);
            const denom = Math.abs(t[i]) + Math.abs(f[i]);
            sape += denom === 0 ? 0 : (2 * Math.abs(e)) / denom;
        }
        return {
            MAE: ae / n,
            MSE: se / n,
            RMSE: Math.sqrt(se / n),
            MAPE: (100 * ape) / n,
            sMAPE: (100 * sape) / n,
        };
    }
    const truthAt = (p) => preset.truth(p);
    const forecastAt = (p) => preset.forecast(p).map((v, i) => v + deltas[i]);
    const metricsAt = (p) => metrics(truthAt(p), forecastAt(p));
    const moved = () => deltas.some((d) => Math.abs(d) > 1e-9);

    function load(key) {
        name = key;
        preset = PRESETS[key];
        param = preset.start;
        deltas = deltas.map(() => 0);
        start = metricsAt(param);
        // A fixed axis that fits the whole range of the slider.
        let lo = Infinity, hi = -Infinity;
        for (const p of [preset.min, preset.max]) {
            for (const v of truthAt(p).concat(preset.forecast(p))) {
                lo = Math.min(lo, v);
                hi = Math.max(hi, v);
            }
        }
        const pad = Math.max(0.6, (hi - lo) * 0.12);
        yMin = lo - pad;
        yMax = hi + pad;
        Object.assign(input, { min: preset.min, max: preset.max, step: preset.step });
        sliderLabel.textContent = preset.slider;
        render();
    }

    // ---- DOM ---------------------------------------------------------------
    el.innerHTML = "";
    const style = document.createElement("style");
    style.textContent = `
.dl-metrics { display: grid; grid-template-columns: repeat(auto-fit, minmax(5.8rem, 1fr)); gap: .4rem; margin: .4rem 0 .2rem; }
.dl-metrics div { min-width: 0; padding: .3rem .45rem; border-radius: .3rem; background: var(--md-default-fg-color--lightest); line-height: 1.25; }
.dl-metrics b { display: block; font-size: .6rem; font-weight: 600; color: var(--md-default-fg-color--light); }
.dl-metrics span { font-variant-numeric: tabular-nums; font-size: .78rem; }
.dl-metrics small { display: block; font-size: .58rem; color: var(--md-default-fg-color--light); }
.dl-metrics small.up { color: var(--md-accent-fg-color); }
.dl-metrics svg { margin: .15rem 0; overflow: hidden; }
.dl-widget__controls .dl-metrics__slider input[type="range"] { width: 11rem; max-width: 42vw; }
.dl-metrics__slider output { min-width: 3em; font-variant-numeric: tabular-nums; }`;

    const tabs = document.createElement("div");
    tabs.className = "dl-widget__controls";
    const seg = document.createElement("div");
    seg.className = "dl-widget__seg";
    seg.setAttribute("role", "group");
    seg.setAttribute("aria-label", "Scenario");
    const buttons = Object.entries(PRESETS).map(([key, p]) => {
        const b = document.createElement("button");
        b.type = "button";
        b.textContent = p.label;
        b.addEventListener("click", () => load(key));
        seg.appendChild(b);
        return [key, b];
    });
    tabs.appendChild(seg);

    const controls = document.createElement("div");
    controls.className = "dl-widget__controls";
    const wrap = document.createElement("label");
    wrap.className = "dl-metrics__slider";
    const sliderLabel = document.createElement("span");
    const input = document.createElement("input");
    input.type = "range";
    const out = document.createElement("output");
    input.addEventListener("input", () => {
        param = parseFloat(input.value);
        render();
    });
    wrap.append(sliderLabel, input, out);
    const reset = document.createElement("button");
    reset.type = "button";
    reset.className = "dl-widget__btn";
    reset.textContent = "Reset";
    reset.addEventListener("click", () => load(name));
    controls.append(wrap, reset);

    const svg = document.createElementNS(NS, "svg");
    svg.setAttribute("role", "img");
    svg.style.touchAction = "none";
    const table = document.createElement("div");
    table.className = "dl-metrics";
    const caption = document.createElement("p");
    caption.className = "dl-widget__caption";
    caption.setAttribute("aria-live", "polite");
    el.append(style, tabs, controls, svg, table, caption);

    function node(tag, attrs, parent) {
        const n = document.createElementNS(NS, tag);
        for (const k in attrs) n.setAttribute(k, attrs[k]);
        if (parent) parent.appendChild(n);
        return n;
    }

    function fmt(v) {
        if (!isFinite(v)) return "∞";
        if (v >= 1000) return v.toExponential(1);
        if (v >= 100) return v.toFixed(0);
        if (v >= 10) return v.toFixed(1);
        return v.toFixed(2);
    }
    const unit = (key) => (key.includes("MAPE") ? "%" : "");
    const val = (key, m) => fmt(m[key]) + unit(key);

    // ---- Chart: truth and forecast, forecast points draggable ---------------
    let W = 640, H = 240, padL = 40;
    const padR = 14, padT = 12, padB = 24;
    let truth = [];
    let forecast = [];
    const xOf = (i) => padL + ((W - padL - padR) * (i + 0.5)) / truth.length;
    const yOf = (v) => padT + ((H - padT - padB) * (yMax - v)) / (yMax - yMin);
    const vOf = (y) => yMax - ((y - padT) * (yMax - yMin)) / (H - padT - padB);
    const clamp = (v) => Math.max(yMin, Math.min(yMax, v));

    function drawChart(c) {
        W = Math.max(280, Math.min(720, Math.round(el.clientWidth - 2) || 640));
        H = W < 480 ? 200 : 240;
        padL = W < 480 ? 32 : 40;
        svg.setAttribute("viewBox", `0 0 ${W} ${H}`);
        svg.setAttribute("aria-label", `Truth and forecast for 10 time steps in the scenario ${preset.label}; the forecast points can be dragged.`);
        svg.innerHTML = "";

        for (let k = 0; k <= 4; k++) {
            const v = yMin + ((yMax - yMin) * k) / 4;
            node("line", { x1: padL, x2: W - padR, y1: yOf(v), y2: yOf(v), stroke: c.fgLightest }, svg);
            node("text", { x: padL - 6, y: yOf(v) + 3, "text-anchor": "end", "font-size": 10, fill: c.fgLight }, svg).textContent =
                Math.abs(v) >= 10 ? v.toFixed(0) : v.toFixed(1);
        }
        if (yMin < 0 && yMax > 0) {
            node("line", { x1: padL, x2: W - padR, y1: yOf(0), y2: yOf(0), stroke: c.fgLighter, "stroke-dasharray": "3 3" }, svg);
        }
        truth.forEach((y, i) => {
            node("line", { x1: xOf(i), x2: xOf(i), y1: yOf(y), y2: yOf(clamp(forecast[i])), stroke: c.accent, "stroke-opacity": 0.4, "stroke-width": 6, "stroke-linecap": "round" }, svg);
        });
        node("polyline", { points: truth.map((y, i) => `${xOf(i)},${yOf(y)}`).join(" "), fill: "none", stroke: c.fg, "stroke-width": 1.5 }, svg);
        truth.forEach((y, i) => node("circle", { cx: xOf(i), cy: yOf(y), r: 3.5, fill: c.fg }, svg));
        if (name === "zero") {
            // The true value the slider moves.
            node("circle", { cx: xOf(ZERO_AT), cy: yOf(truth[ZERO_AT]), r: 8, fill: "none", stroke: c.fg, "stroke-width": 1.2, "stroke-dasharray": "2 2" }, svg);
            node("text", { x: xOf(ZERO_AT) + 12, y: yOf(truth[ZERO_AT]) + 13, "font-size": 10, fill: c.fgLight }, svg).textContent = "smallest";
        }

        forecast.forEach((v, i) => {
            const g = node("g", { tabindex: 0, role: "slider", "aria-label": `Forecast at step ${i + 1}`, "aria-valuenow": v.toFixed(2), style: "cursor: ns-resize; outline: none" }, svg);
            node("circle", { cx: xOf(i), cy: yOf(clamp(v)), r: 14, fill: "transparent" }, g);
            node("circle", { cx: xOf(i), cy: yOf(clamp(v)), r: 6, fill: c.bg, stroke: c.accent, "stroke-width": 2.5 }, g);
            g.addEventListener("pointerdown", (ev) => drag(ev, i));
            g.addEventListener("keydown", (ev) => {
                if (ev.key !== "ArrowUp" && ev.key !== "ArrowDown") return;
                ev.preventDefault();
                const step = (yMax - yMin) / 40;
                moveTo(i, forecast[i] + (ev.key === "ArrowUp" ? step : -step));
                svg.querySelectorAll("g[tabindex]")[i].focus();
            });
        });

        const lx = padL + 4;
        node("circle", { cx: lx + 4, cy: H - 8, r: 3.5, fill: c.fg }, svg);
        node("text", { x: lx + 12, y: H - 4.5, "font-size": 10, fill: c.fgLight }, svg).textContent = "truth";
        node("circle", { cx: lx + 54, cy: H - 8, r: 4.5, fill: c.bg, stroke: c.accent, "stroke-width": 2 }, svg);
        node("text", { x: lx + 63, y: H - 4.5, "font-size": 10, fill: c.fgLight }, svg).textContent = "forecast (drag me)";
    }

    function moveTo(i, value) {
        deltas[i] = clamp(value) - preset.forecast(param)[i];
        render();
    }

    function drag(ev, i) {
        ev.preventDefault();
        const rect = svg.getBoundingClientRect();
        const scale = H / rect.height;
        const move = (e) => moveTo(i, vOf((e.clientY - rect.top) * scale));
        const up = () => {
            window.removeEventListener("pointermove", move);
            window.removeEventListener("pointerup", up);
        };
        window.addEventListener("pointermove", move);
        window.addEventListener("pointerup", up);
    }

    // ---- Tiles: value, change against the start, and the curve over the slider
    function drawTiles(c, m) {
        const grid = [];
        for (let k = 0; k <= GRID; k++) {
            const p = preset.min + ((preset.max - preset.min) * k) / GRID;
            grid.push([p, metricsAt(p)]);
        }
        table.innerHTML = "";
        const sparks = [];
        for (const key of KEYS) {
            const cell = document.createElement("div");
            const ratio = start[key] > 0 ? m[key] / start[key] : m[key] > 0 ? Infinity : 1;
            const changed = Math.abs(ratio - 1) > 0.005;
            const label = document.createElement("b");
            label.textContent = key;
            const value = document.createElement("span");
            value.textContent = val(key, m);
            const spark = document.createElementNS(NS, "svg");
            spark.setAttribute("aria-hidden", "true");
            const note = document.createElement("small");
            if (ratio > 1.005) note.className = "up";
            note.textContent = changed ? (isFinite(ratio) ? `${ratio.toFixed(ratio < 10 ? 2 : 1)}× start` : "from 0") : "start";
            cell.append(label, value, spark, note);
            table.appendChild(cell);
            sparks.push([spark, key]);
        }
        // Drawn once all tiles are in place, so each chart gets its final width.
        for (const [spark, key] of sparks) drawSpark(spark, key, grid, m[key], c);
    }

    function drawSpark(spark, key, grid, current, c) {
        const w = Math.max(60, Math.round(spark.getBoundingClientRect().width) || 120);
        const h = 46, top = 4, base = h - 12;
        const top0 = Math.max(...grid.map(([, mm]) => (isFinite(mm[key]) ? mm[key] : 0)), current, 1e-9) * 1.08;
        const xS = (p) => 5 + ((w - 10) * (p - preset.min)) / (preset.max - preset.min);
        const yS = (v) => base - (base - top) * Math.min(v / top0, 1);
        spark.setAttribute("viewBox", `0 0 ${w} ${h}`);
        const pts = grid.map(([p, mm]) => `${xS(p).toFixed(1)},${yS(mm[key]).toFixed(1)}`);
        node("path", { d: `M${pts.join("L")}L${xS(preset.max)},${base}L${xS(preset.min)},${base}Z`, fill: c.accent, "fill-opacity": 0.14 }, spark);
        node("path", { d: `M${pts.join("L")}`, fill: "none", stroke: c.accent, "stroke-width": 1.8 }, spark);
        node("line", { x1: xS(preset.min), x2: xS(preset.max), y1: base, y2: base, stroke: c.fgLighter }, spark);
        // Axis: both ends of the slider, and a tick where the scenario starts.
        for (const [p, anchor] of [[preset.min, "start"], [preset.max, "end"]]) {
            node("text", { x: xS(p), y: h - 1, "text-anchor": anchor, "font-size": 8, fill: c.fgLight }, spark).textContent = preset.show(p);
        }
        node("line", { x1: xS(preset.start), x2: xS(preset.start), y1: base, y2: base + 4, stroke: c.fgLight }, spark);
        // Where the slider is now.
        node("line", { x1: xS(param), x2: xS(param), y1: top - 2, y2: base, stroke: c.fg, "stroke-opacity": 0.45 }, spark);
        node("circle", { cx: xS(param), cy: yS(current), r: 3.2, fill: c.accent, stroke: c.bg, "stroke-width": 1 }, spark);
    }

    // ---- Caption: what the current setting shows ----------------------------
    function explain(m) {
        const extra = moved() ? " You moved some points; the curves include those changes." : "";
        const e = truth.map((y, i) => y - forecast[i]);
        if (name === "size") {
            if (param === 0) return "With no error at all, every metric is 0." + extra;
            if (param === preset.start) {
                return "Every error is one unit or less, so all metrics are small. Move the slider to scale all errors at once: MAE, RMSE, MAPE and sMAPE follow straight lines, while MSE bends upwards because it squares the errors." + extra;
            }
            return `Every error is ${param.toFixed(1)} times its starting size. MAE, RMSE, MAPE and sMAPE grow in step with it, so their curves are straight lines; MSE grows with its square (${(param * param).toFixed(2)}×), so its curve bends upwards.` + extra;
        }
        if (name === "miss") {
            const sumAbs = e.reduce((s, v) => s + Math.abs(v), 0);
            const sumSq = e.reduce((s, v) => s + v * v, 0);
            const a = sumAbs ? (100 * Math.abs(e[MISS_AT])) / sumAbs : 0;
            const b = sumSq ? (100 * e[MISS_AT] * e[MISS_AT]) / sumSq : 0;
            if (Math.abs(e[MISS_AT]) <= 1) {
                return "Without the big miss every error is 1 or less, and no single point stands out. Move the slider up to see how differently MAE and MSE react to one large miss." + extra;
            }
            return `One forecast misses by ${Math.abs(e[MISS_AT]).toFixed(1)}; the others by 1 or less. That single point makes up ${a.toFixed(0)}% of MAE but ${b.toFixed(0)}% of MSE: squaring lets one large miss dominate, and RMSE inherits it.` + extra;
        }
        if (name === "bias") {
            if (param === 0) return "No bias: every forecast is exactly right and every metric is 0." + extra;
            const mirror = metrics(truth, preset.forecast(-param).map((v, i) => v + deltas[i]));
            const dir = param > 0 ? ["high", "low"] : ["low", "high"];
            return `Every forecast is ${Math.abs(param).toFixed(1)} too ${dir[0]}. MAE and RMSE both equal the size of the bias, and MAPE is the same in either direction. sMAPE is not: ${val("sMAPE", m)} here, ${val("sMAPE", mirror)} if the forecasts were ${Math.abs(param).toFixed(1)} too ${dir[1]}. Its curve is lopsided around 0.` + extra;
        }
        const share = (100 * Math.abs(e[ZERO_AT] / truth[ZERO_AT])) / truth.length;
        return `Every forecast is off by 0.5, so MAE, MSE and RMSE do not change. The smallest true value (circled) is ${truth[ZERO_AT].toFixed(2)}, where that error is ${fmt(Math.abs((100 * e[ZERO_AT]) / truth[ZERO_AT]))}% of the truth: this one point contributes ${fmt(share)} percentage points of MAPE's ${val("MAPE", m)}. sMAPE cannot exceed 200%.` + extra;
    }

    function render() {
        const c = api.colors();
        truth = truthAt(param);
        forecast = forecastAt(param);
        const m = metrics(truth, forecast);
        input.value = param;
        out.textContent = preset.show(param);
        reset.hidden = !moved() && param === preset.start;
        for (const [key, b] of buttons) b.setAttribute("aria-pressed", String(key === name));
        drawChart(c);
        drawTiles(c, m);
        caption.textContent = explain(m);
    }

    load("size");

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
