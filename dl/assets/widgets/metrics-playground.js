/*
 * Metrics playground: drag the forecast points and watch MAE, MSE, RMSE,
 * MAPE and sMAPE react. Presets show the classic surprises: one big miss
 * dominates RMSE, a constant bias, and truths near zero blowing up MAPE.
 *
 *     <div class="dl-widget" data-widget="metrics-playground" markdown>
 *     Fallback text for readers without JavaScript.
 *     </div>
 *
 * Forecast points are focusable: arrow keys move them up and down.
 */
dlWidgets.register("metrics-playground", (el, api) => {
    const NS = "http://www.w3.org/2000/svg";
    const BASE_TRUTH = [12, 14, 13, 17, 16, 19, 18, 21, 20, 23];
    const NEAR_ZERO = [0.4, 1.2, 0.05, 2.0, 0.6, 1.8, 0.3, 2.4, 1.0, 2.6];
    const WIGGLE = [0.8, -0.6, 1.0, -0.9, 0.5, -0.7, 0.9, -0.5, 0.7, -0.8];

    const PRESETS = {
        small: {
            label: "Small errors",
            truth: BASE_TRUTH,
            forecast: BASE_TRUTH.map((y, i) => y + WIGGLE[i]),
            note: "Every error is under one unit, so all metrics are small. Drag a point to see which metric moves most.",
        },
        miss: {
            label: "One big miss",
            truth: BASE_TRUTH,
            forecast: BASE_TRUTH.map((y, i) => y + (i === 6 ? 9 : WIGGLE[i])),
            note: "Same as Small errors except one 9-unit miss. Compared with Small errors, MAE only doubles (0.74 → 1.55), but MSE grows about 15× and RMSE about 4×: squaring lets a single large miss dominate.",
        },
        bias: {
            label: "Bias",
            truth: BASE_TRUTH,
            forecast: BASE_TRUTH.map((y) => y + 3),
            note: "Every forecast is 3 units too high. MAE and RMSE both equal 3: when all errors are the same size, the norms agree.",
        },
        zero: {
            label: "Near zero",
            truth: NEAR_ZERO,
            forecast: NEAR_ZERO.map((y) => y + 0.5),
            note: "Every forecast is off by just 0.5, yet MAPE is about 160%: it divides by the truth, and the third truth is 0.05. Drag that point to see it dominate. sMAPE stays bounded (at most 200%).",
        },
    };

    let preset = "small";
    let truth = [];
    let forecast = [];
    let start = null; // metrics of the preset as loaded, for the "× start" ratios
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

    function load(name) {
        preset = name;
        truth = PRESETS[name].truth.slice();
        forecast = PRESETS[name].forecast.slice();
        start = metrics(truth, forecast);
        const all = truth.concat(forecast);
        const lo = Math.min(...all);
        const hi = Math.max(...all);
        const pad = Math.max(2, (hi - lo) * 0.35);
        yMin = name === "zero" ? Math.min(-1, lo - pad) : lo - pad;
        yMax = hi + pad;
        render();
    }

    // ---- DOM ---------------------------------------------------------------
    el.innerHTML = "";
    const controls = document.createElement("div");
    controls.className = "dl-widget__controls";
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
    controls.appendChild(seg);

    const svg = document.createElementNS(NS, "svg");
    svg.setAttribute("role", "img");
    svg.style.touchAction = "none";

    const table = document.createElement("div");
    table.className = "dl-metrics";

    const caption = document.createElement("p");
    caption.className = "dl-widget__caption";
    caption.setAttribute("aria-live", "polite");

    const style = document.createElement("style");
    style.textContent = `
.dl-metrics { display: grid; grid-template-columns: repeat(auto-fit, minmax(5.2rem, 1fr)); gap: .4rem; margin: .4rem 0 .2rem; }
.dl-metrics div { padding: .3rem .45rem; border-radius: .3rem; background: var(--md-default-fg-color--lightest); line-height: 1.25; }
.dl-metrics b { display: block; font-size: .6rem; font-weight: 600; color: var(--md-default-fg-color--light); }
.dl-metrics span { font-variant-numeric: tabular-nums; font-size: .78rem; }
.dl-metrics small { display: block; font-size: .58rem; color: var(--md-default-fg-color--light); }
.dl-metrics small.up { color: var(--md-accent-fg-color); }`;
    el.append(style, controls, svg, table, caption);

    function node(name, attrs, parent) {
        const n = document.createElementNS(NS, name);
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

    // Geometry, recomputed on render so it follows the column width.
    let W = 640, H = 260, padL = 40, padR = 14, padT = 14, padB = 26;
    const xOf = (i) => padL + ((W - padL - padR) * (i + 0.5)) / truth.length;
    const yOf = (v) => padT + ((H - padT - padB) * (yMax - v)) / (yMax - yMin);
    const vOf = (y) => yMax - ((y - padT) * (yMax - yMin)) / (H - padT - padB);

    function render() {
        const c = api.colors();
        W = Math.max(300, Math.min(720, Math.round(el.clientWidth - 2) || 640));
        H = W < 480 ? 220 : 260;
        padL = W < 480 ? 32 : 40;
        svg.setAttribute("viewBox", `0 0 ${W} ${H}`);
        svg.setAttribute(
            "aria-label",
            "Truth and forecast for 10 time steps; drag the forecast points to change the errors."
        );
        svg.innerHTML = "";

        // Grid and y labels.
        const ticks = 4;
        for (let k = 0; k <= ticks; k++) {
            const v = yMin + ((yMax - yMin) * k) / ticks;
            const y = yOf(v);
            node("line", { x1: padL, x2: W - padR, y1: y, y2: y, stroke: c.fgLightest }, svg);
            const t = node("text", { x: padL - 6, y: y + 3, "text-anchor": "end", "font-size": 10, fill: c.fgLight }, svg);
            t.textContent = Math.abs(v) >= 10 ? v.toFixed(0) : v.toFixed(1);
        }
        if (yMin < 0 && yMax > 0) {
            node("line", { x1: padL, x2: W - padR, y1: yOf(0), y2: yOf(0), stroke: c.fgLighter, "stroke-dasharray": "3 3" }, svg);
        }

        // Errors as vertical bars.
        truth.forEach((y, i) => {
            node("line", { x1: xOf(i), x2: xOf(i), y1: yOf(y), y2: yOf(forecast[i]), stroke: c.accent, "stroke-opacity": 0.35, "stroke-width": 6, "stroke-linecap": "round" }, svg);
        });

        // Truth line and points.
        node("polyline", { points: truth.map((y, i) => `${xOf(i)},${yOf(y)}`).join(" "), fill: "none", stroke: c.fg, "stroke-width": 1.5 }, svg);
        truth.forEach((y, i) => node("circle", { cx: xOf(i), cy: yOf(y), r: 3.5, fill: c.fg }, svg));

        // Forecast points: draggable and keyboard-adjustable.
        forecast.forEach((v, i) => {
            const g = node("g", { tabindex: 0, role: "slider", "aria-label": `Forecast at step ${i + 1}`, "aria-valuenow": v.toFixed(2), style: "cursor: ns-resize; outline: none" }, svg);
            node("circle", { cx: xOf(i), cy: yOf(v), r: 14, fill: "transparent" }, g);
            node("circle", { cx: xOf(i), cy: yOf(v), r: 6, fill: c.bg, stroke: c.accent, "stroke-width": 2.5 }, g);
            g.addEventListener("pointerdown", (ev) => drag(ev, i));
            g.addEventListener("keydown", (ev) => {
                const step = (yMax - yMin) / 40;
                if (ev.key === "ArrowUp" || ev.key === "ArrowDown") {
                    ev.preventDefault();
                    forecast[i] = clamp(forecast[i] + (ev.key === "ArrowUp" ? step : -step));
                    render();
                    svg.querySelectorAll("g[tabindex]")[i].focus();
                }
            });
        });

        // Legend.
        const lx = padL + 4;
        node("circle", { cx: lx + 4, cy: H - 8, r: 3.5, fill: c.fg }, svg);
        node("text", { x: lx + 12, y: H - 4.5, "font-size": 10, fill: c.fgLight }, svg).textContent = "truth";
        node("circle", { cx: lx + 54, cy: H - 8, r: 4.5, fill: c.bg, stroke: c.accent, "stroke-width": 2 }, svg);
        node("text", { x: lx + 63, y: H - 4.5, "font-size": 10, fill: c.fgLight }, svg).textContent = "forecast (drag me)";

        // Metrics with the change against the preset as loaded.
        const m = metrics(truth, forecast);
        table.innerHTML = "";
        for (const key of ["MAE", "MSE", "RMSE", "MAPE", "sMAPE"]) {
            const cell = document.createElement("div");
            const unit = key.includes("MAPE") ? "%" : "";
            const ratio = start[key] > 0 ? m[key] / start[key] : m[key] > 0 ? Infinity : 1;
            const changed = Math.abs(ratio - 1) > 0.005;
            cell.innerHTML = `<b>${key}</b><span>${fmt(m[key])}${unit}</span>` +
                `<small class="${ratio > 1.005 ? "up" : ""}">${changed ? (isFinite(ratio) ? `${ratio.toFixed(1)}× start` : "from 0") : "start"}</small>`;
            table.appendChild(cell);
        }
        for (const [key, b] of buttons) b.setAttribute("aria-pressed", String(key === preset));
        const moved = forecast.some((v, i) => Math.abs(v - PRESETS[preset].forecast[i]) > 1e-9);
        caption.textContent = moved ? insight(m) : PRESETS[preset].note;
    }

    // What the reader's drag just showed: the metric that changed most.
    function insight(m) {
        let best = null, ratio = 1;
        for (const key of ["MAE", "MSE", "RMSE", "MAPE", "sMAPE"]) {
            const r = start[key] > 0 ? m[key] / start[key] : 1;
            if (Math.abs(Math.log(r)) > Math.abs(Math.log(ratio))) [best, ratio] = [key, r];
        }
        if (!best || Math.abs(ratio - 1) < 0.02) return "Barely any change: try dragging a point further.";
        const how = ratio > 1 ? `grew most (${ratio.toFixed(1)}×)` : `shrank most (to ${ratio.toFixed(2)}×)`;
        const why = {
            MAE: "every unit of error counts the same, wherever it is.",
            MSE: "squaring makes the biggest errors dominate.",
            RMSE: "it squares errors before averaging, so big misses weigh more than in MAE.",
            MAPE: "it divides each error by the truth, so errors where the truth is small count for much more.",
            sMAPE: "it divides by the average of truth and forecast, which softens MAPE's blow-ups near zero.",
        }[best];
        return `${best} ${how}: ${why}`;
    }

    function clamp(v) {
        return Math.max(yMin, Math.min(yMax, v));
    }

    function drag(ev, i) {
        ev.preventDefault();
        const rect = svg.getBoundingClientRect();
        const scale = H / rect.height;
        const move = (e) => {
            forecast[i] = clamp(vOf((e.clientY - rect.top) * scale));
            render();
        };
        const up = () => {
            window.removeEventListener("pointermove", move);
            window.removeEventListener("pointerup", up);
        };
        window.addEventListener("pointermove", move);
        window.addEventListener("pointerup", up);
    }

    load("small");

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
