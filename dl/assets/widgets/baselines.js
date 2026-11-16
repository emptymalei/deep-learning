/*
 * Two explainers for the page "Baselines and Skill Scores" (the third one,
 * baseline-skill, has its own file).
 *
 *   baseline-forecasts  a series with trend and season, and the mean, naive,
 *                       seasonal naive and drift forecasts from a movable origin
 *   mase-scale          the same series and forecast in another unit: the MAE
 *                       changes with the unit, MASE does not
 */
(function () {
    const { rng, node, html, slider, seg, button, legend, widthOf, lifecycle, polyline } = window.dlWidgetKit;
    const mean = (a) => a.reduce((s, v) => s + v, 0) / (a.length || 1);

    dlWidgets.register("baseline-forecasts", (el, api) => {
        const N = 84, M = 7, H = 14;
        const PATTERN = [0, 0.3, 0.5, 0.4, 0.9, 1.6, 1.2].map((v) => v - 0.7);
        const state = { trend: 0.04, season: 1.5, origin: 56, playing: false };
        const g = rng(8);
        const NOISE = Array.from({ length: N }, () => 0.3 * g.normal());
        const series = () => Array.from({ length: N }, (_, t) => 10 + state.trend * t + state.season * PATTERN[t % M] + NOISE[t]);
        el.innerHTML = "";
        const controls = html("div", "dl-widget__controls", el);
        slider(controls, "Trend", -0.08, 0.12, 0.01, state.trend, 2, (v) => { state.trend = v; render(); });
        slider(controls, "Weekly pattern", 0, 3, 0.1, state.season, 1, (v) => { state.season = v; render(); });
        const oS = slider(controls, "Forecast origin", 21, N - H, 1, state.origin, 0, (v) => { state.origin = v; stop(); render(); });
        const play = button(controls, "Play", () => (state.playing ? stop() : start()));
        const svg = node("svg", { role: "img" }, el);
        const legendBox = html("div", "dl-widget__legend", el);
        const caption = html("p", "dl-widget__caption", el);
        caption.setAttribute("aria-live", "polite");
        let timer = 0;
        function start() {
            state.playing = true;
            play.textContent = "Pause";
            state.origin = 20;
            const tick = () => {
                state.origin += 1;
                oS.set(state.origin);
                render();
                if (state.origin >= N - H) return stop();
                timer = setTimeout(tick, 220);
            };
            tick();
        }
        function stop() {
            clearTimeout(timer);
            state.playing = false;
            play.textContent = "Play";
        }
        let W = 640;
        function render() {
            const c = api.colors();
            W = widthOf(el);
            const y = series();
            const T = state.origin;
            const hist = y.slice(0, T);
            const fut = y.slice(T, T + H);
            const methods = [
                ["Mean", "#8e6bbf", fut.map(() => mean(hist))],
                ["Naive", c.accent, fut.map(() => hist[T - 1])],
                ["Seasonal naive", c.train, fut.map((_, j) => hist[T + j - M * (Math.floor(j / M) + 1)])],
                ["Drift", "#3a9d6e", fut.map((_, j) => hist[T - 1] + ((j + 1) * (hist[T - 1] - hist[0])) / (T - 1))],
            ];
            const padL = 28, padR = 8, Hh = W < 480 ? 210 : 240, padB = 20;
            svg.setAttribute("viewBox", `0 0 ${W} ${Hh}`);
            svg.innerHTML = "";
            const all = y.concat(...methods.map((m) => m[2]));
            const lo = Math.min(...all) - 0.3, hi = Math.max(...all) + 0.3;
            const x = (t) => padL + ((W - padL - padR) * t) / (N - 1);
            const yy = (v) => 6 + (Hh - padB - 6) * (1 - (v - lo) / (hi - lo));
            node("rect", { x: x(T - 0.5), y: 4, width: x(T + H - 0.5) - x(T - 0.5), height: Hh - padB - 4, fill: c.fgLightest }, svg);
            for (const v of [Math.ceil(lo), Math.round((lo + hi) / 2), Math.floor(hi)]) node("text", { x: padL - 5, y: yy(v) + 3, "text-anchor": "end", "font-size": 10, fill: c.fgLight }, svg, v);
            node("polyline", { points: polyline(y.map((v, t) => [x(t), yy(v)])), fill: "none", stroke: c.fgLighter, "stroke-width": 1.2 }, svg);
            node("polyline", { points: polyline(hist.map((v, t) => [x(t), yy(v)])), fill: "none", stroke: c.fg, "stroke-width": 1.6 }, svg);
            for (const [, col, f] of methods) node("polyline", { points: polyline(f.map((v, j) => [x(T + j), yy(v)])), fill: "none", stroke: col, "stroke-width": 2 }, svg);
            node("text", { x: x(T), y: Hh - 5, "text-anchor": "middle", "font-size": 10, fill: c.fg }, svg, "origin");
            const errs = methods.map(([name, col, f]) => [name, col, mean(f.map((v, j) => Math.abs(v - fut[j])))]);
            legend(legendBox, [["observed (known part in dark)", c.fg]].concat(errs.map(([name, col, e]) => [`${name}: MAE ${e.toFixed(2)}`, col])));
            const best = errs.slice().sort((a, b) => a[2] - b[2])[0];
            svg.setAttribute("aria-label", "A daily series with a trend and a weekly pattern, and four simple forecasts for the next 14 days.");
            caption.textContent =
                `Forecasts for days ${T + 1} to ${T + H}, made after day ${T}. The best of the four here is ${best[0].toLowerCase()} with an MAE of ${best[2].toFixed(2)}: ` +
                (best[0] === "Seasonal naive" ? "the weekly pattern dominates, and only this method repeats it."
                    : best[0] === "Drift" ? "the trend dominates, and only this method continues it."
                    : best[0] === "Naive" ? "with little trend and pattern, the last value is hard to beat."
                    : "with no trend, the long-run level is the steadiest guess.") +
                " A model has to beat this one, not the weakest of the four.";
        }
        render();
        return lifecycle(el, api, render, stop);
    });

    dlWidgets.register("mase-scale", (el, api) => {
        const TRAIN = 40, TEST = 12;
        const state = { unit: 1, error: 0.6, playing: false };
        const g = rng(31);
        const BASE = [];
        let v = 10;
        for (let t = 0; t < TRAIN + TEST; t++) {
            v += 0.5 * g.normal();
            BASE.push(v);
        }
        const EPS = Array.from({ length: TEST }, () => g.normal());
        el.innerHTML = "";
        const controls = html("div", "dl-widget__controls", el);
        const uS = slider(controls, "Unit factor", 1, 100, 1, state.unit, 0, (x) => { state.unit = x; stop(); render(); });
        slider(controls, "Model error", 0, 2, 0.1, state.error, 1, (x) => { state.error = x; render(); });
        const play = button(controls, "Play", () => (state.playing ? stop() : start()));
        const svg = node("svg", { role: "img" }, el);
        const legendBox = html("div", "dl-widget__legend", el);
        const caption = html("p", "dl-widget__caption", el);
        caption.setAttribute("aria-live", "polite");
        let timer = 0;
        function start() {
            state.playing = true;
            play.textContent = "Pause";
            state.unit = 0;
            const tick = () => {
                state.unit = Math.min(100, state.unit + (state.unit < 10 ? 1 : 5));
                uS.set(state.unit);
                render();
                if (state.unit >= 100) return stop();
                timer = setTimeout(tick, 200);
            };
            tick();
        }
        function stop() {
            clearTimeout(timer);
            state.playing = false;
            play.textContent = "Play";
        }
        let W = 640;
        function render() {
            const c = api.colors();
            W = widthOf(el);
            const y = BASE.map((b) => b * state.unit);
            const f = y.slice(TRAIN).map((b, j) => b + state.error * 0.5 * EPS[j] * state.unit);
            const padL = 44, padR = 8, topH = 150, gap = 26, botH = 80, padB = 18;
            const Hh = topH + gap + botH + padB;
            svg.setAttribute("viewBox", `0 0 ${W} ${Hh}`);
            svg.innerHTML = "";
            const N = TRAIN + TEST;
            const lo = Math.min(...y, ...f), hi = Math.max(...y, ...f);
            const x = (t) => padL + ((W - padL - padR) * t) / (N - 1);
            const yy = (val) => 8 + (topH - 14) * (1 - (val - lo) / (hi - lo));
            node("rect", { x: x(TRAIN - 0.5), y: 4, width: x(N - 1) - x(TRAIN - 0.5), height: topH - 4, fill: c.fgLightest }, svg);
            for (const val of [lo, hi]) node("text", { x: padL - 5, y: yy(val) + 3, "text-anchor": "end", "font-size": 10, fill: c.fgLight }, svg, val.toFixed(0));
            node("polyline", { points: polyline(y.map((val, t) => [x(t), yy(val)])), fill: "none", stroke: c.fg, "stroke-width": 1.5 }, svg);
            node("polyline", { points: polyline(f.map((val, j) => [x(TRAIN + j), yy(val)])), fill: "none", stroke: c.accent, "stroke-width": 2 }, svg);
            // Bottom: one-step changes in the training part, and forecast errors in the test part.
            const steps = y.slice(1, TRAIN).map((val, i) => Math.abs(val - y[i]));
            const errs = f.map((val, j) => Math.abs(val - y[TRAIN + j]));
            const scale = mean(steps), mae = mean(errs);
            const y0 = topH + gap;
            const bmax = Math.max(...steps, ...errs) * 1.05;
            const yb = (val) => y0 + botH * (1 - val / bmax);
            const bw = Math.max(2, (W - padL - padR) / N - 2);
            steps.forEach((val, i) => node("rect", { x: x(i + 1) - bw / 2, width: bw, y: yb(val), height: yb(0) - yb(val), fill: c.train, "fill-opacity": 0.65 }, svg));
            errs.forEach((val, j) => node("rect", { x: x(TRAIN + j) - bw / 2, width: bw, y: yb(val), height: yb(0) - yb(val), fill: c.accent, "fill-opacity": 0.75 }, svg));
            node("line", { x1: x(1), x2: x(TRAIN - 1), y1: yb(scale), y2: yb(scale), stroke: c.train, "stroke-width": 2 }, svg);
            node("line", { x1: x(TRAIN), x2: x(N - 1), y1: yb(mae), y2: yb(mae), stroke: c.accent, "stroke-width": 2 }, svg);
            node("text", { x: x(1), y: y0 - 5, "font-size": 10, fill: c.train }, svg, `training: one-step changes, mean ${scale.toFixed(2)}`);
            node("text", { x: x(N - 1), y: y0 - 5, "text-anchor": "end", "font-size": 10, fill: c.accent }, svg, `test: errors, mean ${mae.toFixed(2)}`);
            legend(legendBox, [
                ["series", c.fg],
                ["model forecast, and its absolute errors", c.accent],
                ["absolute one-step changes in the training data", c.train],
            ]);
            svg.setAttribute("aria-label", "A series with a forecast, the absolute one-step changes of the training data and the absolute forecast errors.");
            caption.textContent =
                `MAE = ${mae.toFixed(2)}, in the unit of the data. Scale = ${scale.toFixed(2)}, the MAE the one-step naive forecast had on the training data. ` +
                `MASE = ${mae.toFixed(2)} / ${scale.toFixed(2)} = ${(mae / scale).toFixed(2)}. ` +
                (state.unit > 1 ? `Measuring the same series in a unit ${state.unit} times smaller multiplies the MAE by ${state.unit} and leaves MASE where it was.` : "Change the unit factor: both rows of bars grow together, and their ratio stays the same.");
        }
        render();
        return lifecycle(el, api, render, stop);
    });

    // ==========================================================================
    // metric-target: which point forecast each error measure rewards
    // ==========================================================================
    dlWidgets.register("metric-target", (el, api) => {
        const state = { x: 2, mode: 0, playing: false };
        // Outcomes on a grid: Gamma(2, 1) has density y exp(-y), mean 2, median 1.678.
        const DY = 0.01;
        const YS = Array.from({ length: 2500 }, (_, i) => (i + 0.5) * DY);
        const GAMMA = YS.map((y) => y * Math.exp(-y) * DY);
        const P0 = 0.6; // share of zeros in the intermittent case
        const expect = (f) => {
            let s = 0;
            for (let i = 0; i < YS.length; i++) s += GAMMA[i] * f(YS[i]);
            return s;
        };
        const losses = (mode) => {
            const w = mode === 0 ? 1 : 1 - P0;
            const z = mode === 0 ? 0 : P0;
            return [
                ["Squared error", "rewards the mean", (x) => z * x * x + w * expect((y) => (x - y) * (x - y))],
                ["Absolute error", "rewards the median", (x) => z * Math.abs(x) + w * expect((y) => Math.abs(x - y))],
                mode === 0
                    ? ["Absolute percentage error", "rewards something lower", (x) => expect((y) => Math.abs(x - y) / y)]
                    : ["Absolute percentage error", "undefined: division by zero", null],
            ];
        };
        el.innerHTML = "";
        const controls = html("div", "dl-widget__controls", el);
        const mode = seg(controls, "Demand", ["Skewed demand", "Intermittent demand"], (i) => { state.mode = i; stop(); render(); });
        const xS = slider(controls, "Point forecast", 0, 5, 0.01, state.x, 2, (v) => { state.x = v; stop(); render(); });
        const play = button(controls, "Play", () => (state.playing ? stop() : start()));
        const svg = node("svg", { role: "img" }, el);
        const legendBox = html("div", "dl-widget__legend", el);
        const caption = html("p", "dl-widget__caption", el);
        caption.setAttribute("aria-live", "polite");
        let raf = 0;
        function start() {
            state.playing = true;
            play.textContent = "Pause";
            const t0 = performance.now();
            const frame = (now) => {
                const u = Math.min(1, (now - t0) / 8000);
                state.x = 5 * (1 - u);
                xS.set(state.x);
                render();
                if (u < 1 && state.playing) raf = requestAnimationFrame(frame);
                else stop();
            };
            raf = requestAnimationFrame(frame);
        }
        function stop() {
            cancelAnimationFrame(raf);
            state.playing = false;
            play.textContent = "Play";
        }
        let W = 640;
        function render() {
            const c = api.colors();
            W = widthOf(el);
            mode.select(state.mode);
            const wide = W >= 560;
            const padL = 34, padR = 10, topH = 110, cols = wide ? 3 : 1;
            const pw = (W - padL - padR - (cols - 1) * 40) / cols, ph = wide ? 140 : 110, rowH = ph + 44;
            const H = topH + 30 + (wide ? rowH : 3 * rowH);
            svg.setAttribute("viewBox", `0 0 ${W} ${H}`);
            svg.innerHTML = "";
            const x = (v) => padL + ((W - padL - padR) * v) / 6;
            const dens = (y) => (state.mode === 0 ? 1 : 1 - P0) * y * Math.exp(-y);
            const yT = (d) => 8 + (topH - 8) * (1 - d / 0.4);
            const pts = [];
            for (let v = 0; v <= 6; v += 0.05) pts.push([x(v), yT(dens(v))]);
            node("polygon", { points: polyline(pts.concat([[x(6), topH], [x(0), topH]])), fill: c.fgLighter, "fill-opacity": 0.35 }, svg);
            if (state.mode === 1) {
                node("line", { x1: x(0), x2: x(0), y1: yT(0.38), y2: topH, stroke: c.fg, "stroke-width": 5 }, svg);
                node("text", { x: x(0) + 6, y: yT(0.36), "font-size": 10, fill: c.fg }, svg, "60% of periods: no demand");
            }
            node("line", { x1: padL, x2: W - padR, y1: topH, y2: topH, stroke: c.fgLighter }, svg);
            node("line", { x1: x(state.x), x2: x(state.x), y1: 6, y2: topH + 4, stroke: c.accent, "stroke-width": 2.2 }, svg);
            node("text", { x: x(state.x) + 5, y: 16, "font-size": 10, fill: c.accent }, svg, "forecast");
            for (let v = 0; v <= 6; v++) node("text", { x: x(v), y: topH + 14, "text-anchor": "middle", "font-size": 10, fill: c.fgLight }, svg, v);
            const xs = [];
            for (let i = 0; i <= 500; i++) xs.push(i / 100);
            const found = [];
            losses(state.mode).forEach(([name, note, f], k) => {
                const px = padL + (wide ? k * (pw + 40) : 0);
                const py = topH + 30 + (wide ? 0 : k * rowH) + 16;
                node("rect", { x: px, y: py, width: pw, height: ph, fill: "none", stroke: c.fgLightest }, svg);
                node("text", { x: px, y: py - 5, "font-size": 11, "font-weight": 700, fill: c.fg }, svg, name);
                if (!f) {
                    node("text", { x: px + pw / 2, y: py + ph / 2, "text-anchor": "middle", "font-size": 11, fill: c.accent }, svg, note);
                    found.push(null);
                    return;
                }
                const vals = xs.map(f);
                let iMin = 0;
                vals.forEach((v, i) => { if (v < vals[iMin]) iMin = i; });
                const lo = vals[iMin], hi = Math.min(Math.max(...vals), lo * 4 + 0.5);
                const sx = (v) => px + (pw * v) / 5;
                const sy = (v) => py + ph * (1 - (Math.min(v, hi) - lo * 0.9) / (hi - lo * 0.9));
                node("polyline", { points: polyline(xs.map((v, i) => [sx(v), sy(vals[i])])), fill: "none", stroke: c.train, "stroke-width": 2.2 }, svg);
                node("line", { x1: sx(xs[iMin]), x2: sx(xs[iMin]), y1: py, y2: py + ph, stroke: c.fgLighter, "stroke-dasharray": "3 3" }, svg);
                node("circle", { cx: sx(xs[iMin]), cy: sy(lo), r: 5, fill: "none", stroke: c.fg, "stroke-width": 1.5 }, svg);
                node("circle", { cx: sx(state.x), cy: sy(f(state.x)), r: 4, fill: c.accent }, svg);
                node("text", { x: sx(xs[iMin]), y: py + ph + 13, "text-anchor": "middle", "font-size": 10, fill: c.fg }, svg, `best: ${xs[iMin].toFixed(2)}`);
                node("text", { x: px + pw, y: py - 5, "text-anchor": "end", "font-size": 10, fill: c.fgLight }, svg, note);
                found.push(xs[iMin]);
            });
            legend(legendBox, [
                ["distribution of what will happen", c.fgLighter],
                ["the point forecast, and its expected error", c.accent],
                ["expected error over all possible forecasts; open circle = lowest", c.train],
            ]);
            svg.setAttribute("aria-label", "A distribution of future demand and the expected squared, absolute and percentage error of every possible point forecast.");
            caption.textContent = state.mode === 0
                ? `One distribution, three best forecasts: ${found[0].toFixed(2)} under squared error (the mean), ${found[1].toFixed(2)} under absolute error (the median), ${found[2].toFixed(2)} under absolute percentage error. A model trained and judged with different measures is aimed at one number and scored against another.`
                : `With demand in only 40% of periods, the mean is ${found[0].toFixed(2)}, but the forecast with the lowest absolute error is ${found[1].toFixed(2)}: always predicting no demand. Any measure built on absolute errors, MASE included, ranks that forecast first.`;
        }
        render();
        return lifecycle(el, api, render, stop);
    });

    // ==========================================================================
    // aggregate-ratios: arithmetic mean, geometric mean and win share of relative errors
    // ==========================================================================
    dlWidgets.register("aggregate-ratios", (el, api) => {
        const Z = [-1.4, -0.9, -0.5, -0.15, 0.15, 0.5, 0.9, 1.4];
        const state = { typical: 1, spread: 0.5, swapped: false, playing: false };
        el.innerHTML = "";
        const controls = html("div", "dl-widget__controls", el);
        slider(controls, "Typical ratio", 0.6, 1.4, 0.02, state.typical, 2, (v) => { state.typical = v; stop(); render(); });
        const sS = slider(controls, "Spread across series", 0, 1.2, 0.02, state.spread, 2, (v) => { state.spread = v; stop(); render(); });
        const swap = button(controls, "Swap model and benchmark", () => { state.swapped = !state.swapped; render(); });
        const play = button(controls, "Play", () => (state.playing ? stop() : start()));
        const svg = node("svg", { role: "img" }, el);
        const legendBox = html("div", "dl-widget__legend", el);
        const caption = html("p", "dl-widget__caption", el);
        caption.setAttribute("aria-live", "polite");
        let timer = 0;
        function start() {
            state.playing = true;
            play.textContent = "Pause";
            state.spread = -0.04;
            const tick = () => {
                state.spread = Math.min(1.2, Math.round((state.spread + 0.04) * 100) / 100);
                sS.set(state.spread);
                render();
                if (state.spread >= 1.2) return stop();
                timer = setTimeout(tick, 160);
            };
            tick();
        }
        function stop() {
            clearTimeout(timer);
            state.playing = false;
            play.textContent = "Play";
        }
        const stats = (r) => ({
            am: mean(r),
            gm: Math.exp(mean(r.map(Math.log))),
            win: r.filter((v) => v < 1).length / r.length,
        });
        let W = 640;
        function render() {
            const c = api.colors();
            W = widthOf(el);
            swap.textContent = state.swapped ? "Swap back" : "Swap model and benchmark";
            let r = Z.map((z) => state.typical * Math.exp(state.spread * z));
            const other = stats(r.map((v) => 1 / v));
            if (state.swapped) r = r.map((v) => 1 / v);
            const s = stats(r);
            const padL = 40, padR = 12, H = 230, padB = 26;
            svg.setAttribute("viewBox", `0 0 ${W} ${H}`);
            svg.innerHTML = "";
            // Ratios on a log axis: one half of the distance up means twice the error, one down means half.
            const y = (v) => 10 + (H - padB - 10) * (1 - (Math.log(Math.max(0.15, Math.min(6.5, v))) - Math.log(0.15)) / (Math.log(6.5) - Math.log(0.15)));
            for (const v of [0.25, 0.5, 1, 2, 4]) {
                node("line", { x1: padL, x2: W - padR, y1: y(v), y2: y(v), stroke: v === 1 ? c.fg : c.fgLightest }, svg);
                node("text", { x: padL - 5, y: y(v) + 3, "text-anchor": "end", "font-size": 10, fill: c.fgLight }, svg, v);
            }
            const bw = (W - padL - padR - 150) / r.length;
            r.forEach((v, i) => {
                node("rect", { x: padL + 8 + i * bw, width: bw - 8, y: Math.min(y(v), y(1)), height: Math.abs(y(v) - y(1)), fill: v < 1 ? c.train : c.accent, "fill-opacity": 0.7 }, svg);
                node("text", { x: padL + 4 + i * bw + bw / 2, y: H - 8, "text-anchor": "middle", "font-size": 9.5, fill: c.fgLight }, svg, `series ${i + 1}`);
            });
            const x0 = W - padR - 130;
            for (const [name, v, dash] of [["arithmetic mean", s.am, ""], ["geometric mean", s.gm, "5 3"]]) {
                node("line", { x1: padL, x2: x0 + 6, y1: y(v), y2: y(v), stroke: c.fg, "stroke-width": 2, "stroke-dasharray": dash }, svg);
            }
            const close = Math.abs(y(s.am) - y(s.gm)) < 12;
            node("text", { x: x0 + 10, y: y(s.am) + (close ? -4 : 3), "font-size": 10, fill: c.fg }, svg, `arithmetic mean ${s.am.toFixed(2)}`);
            node("text", { x: x0 + 10, y: y(s.gm) + (close ? 11 : 3), "font-size": 10, fill: c.fg }, svg, `geometric mean ${s.gm.toFixed(2)}`);
            legend(legendBox, [
                ["model better than the benchmark on this series (ratio below 1)", c.train],
                ["model worse", c.accent],
            ]);
            svg.setAttribute("aria-label", "Relative errors of a model against a benchmark on eight series, with their arithmetic and geometric means.");
            caption.textContent =
                `Error of the ${state.swapped ? "benchmark over the model" : "model over the benchmark"} on eight series. Arithmetic mean ${s.am.toFixed(2)}, geometric mean ${s.gm.toFixed(2)}, better on ${Math.round(s.win * 8)} of 8. ` +
                `With the roles swapped: arithmetic mean ${other.am.toFixed(2)}, geometric mean ${other.gm.toFixed(2)}. ` +
                (s.am > 1.02 && other.am > 1.02
                    ? "The arithmetic mean says each method is worse than the other. The two geometric means are reciprocals, so they always agree on who is ahead."
                    : "The two geometric means are reciprocals of each other; the two arithmetic means are not.");
        }
        render();
        return lifecycle(el, api, render, stop);
    });

    // ==========================================================================
    // skill-bias: the skill score of two equally good forecasters (after Wheatcroft 2019)
    // ==========================================================================
    dlWidgets.register("skill-bias", (el, api) => {
        const REPS = 3000;
        const state = { n: 10, playing: false };
        let vals = [];
        function simulate() {
            const g = rng(1000 + state.n);
            vals = [];
            for (let r = 0; r < REPS; r++) {
                let a = 0, b = 0;
                for (let i = 0; i < state.n; i++) {
                    const u = g.normal(), v = g.normal();
                    a += u * u;
                    b += v * v;
                }
                vals.push(1 - a / b);
            }
        }
        el.innerHTML = "";
        const controls = html("div", "dl-widget__controls", el);
        const nS = slider(controls, "Forecasts in the test set N", 5, 200, 1, state.n, 0, (v) => { state.n = v; stop(); simulate(); render(); });
        const play = button(controls, "Play", () => (state.playing ? stop() : start()));
        const svg = node("svg", { role: "img" }, el);
        const legendBox = html("div", "dl-widget__legend", el);
        const caption = html("p", "dl-widget__caption", el);
        caption.setAttribute("aria-live", "polite");
        let timer = 0;
        function start() {
            state.playing = true;
            play.textContent = "Pause";
            state.n = 4;
            const tick = () => {
                state.n = Math.min(200, state.n + (state.n < 30 ? 1 : 6));
                nS.set(state.n);
                simulate();
                render();
                if (state.n >= 200) return stop();
                timer = setTimeout(tick, 140);
            };
            tick();
        }
        function stop() {
            clearTimeout(timer);
            state.playing = false;
            play.textContent = "Play";
        }
        let W = 640;
        function render() {
            const c = api.colors();
            W = widthOf(el);
            const padL = 12, padR = 12, H = 200, padB = 30;
            svg.setAttribute("viewBox", `0 0 ${W} ${H}`);
            svg.innerHTML = "";
            const LO = -3, HI = 1, BINS = 40;
            const counts = new Array(BINS).fill(0);
            let below = 0;
            for (const v of vals) {
                if (v < LO) below++;
                else counts[Math.min(BINS - 1, Math.floor(((v - LO) / (HI - LO)) * BINS))]++;
            }
            const cmax = Math.max(...counts);
            const x = (v) => padL + ((W - padL - padR) * (v - LO)) / (HI - LO);
            const y = (k) => 10 + (H - padB - 10) * (1 - k / cmax);
            counts.forEach((k, i) => {
                const v0 = LO + ((HI - LO) * i) / BINS;
                node("rect", { x: x(v0) + 0.5, width: x(v0 + (HI - LO) / BINS) - x(v0) - 1, y: y(k), height: y(0) - y(k), fill: v0 + 0.05 < 0 ? c.accent : c.train, "fill-opacity": 0.65 }, svg);
            });
            const m = mean(vals), theory = -2 / (state.n - 2);
            node("line", { x1: x(0), x2: x(0), y1: 6, y2: H - padB + 4, stroke: c.fg, "stroke-width": 1.5 }, svg);
            node("line", { x1: x(Math.max(LO, m)), x2: x(Math.max(LO, m)), y1: 6, y2: H - padB + 4, stroke: c.fg, "stroke-width": 2, "stroke-dasharray": "5 3" }, svg);
            node("text", { x: x(Math.max(LO, m)) - 5, y: 16, "text-anchor": "end", "font-size": 10, fill: c.fg }, svg, `mean ${m.toFixed(2)}`);
            for (let v = LO; v <= HI; v++) node("text", { x: x(v), y: H - padB + 15, "text-anchor": "middle", "font-size": 10, fill: c.fgLight }, svg, v);
            node("text", { x: (padL + W - padR) / 2, y: H - 3, "text-anchor": "middle", "font-size": 10, fill: c.fgLight }, svg, "MSE skill score of one forecaster against an equally good one");
            legend(legendBox, [
                ["experiments in which the skill score is negative", c.accent],
                ["positive", c.train],
                ["zero (solid) and the mean over experiments (dashed)", c.fg],
            ]);
            svg.setAttribute("aria-label", `Histogram of the skill score of two equally good forecasters over ${REPS} experiments with ${state.n} forecasts each.`);
            caption.textContent =
                `Two forecasters of exactly the same quality, compared on ${state.n} forecasts, ${REPS} times. The skill score should be 0. Its mean is ${m.toFixed(2)}; the formula −2/(N−2) gives ${theory.toFixed(2)}. ` +
                (below ? `${below} experiments fall below −3 and are not drawn. ` : "") +
                (state.n < 30 ? "With a short test set the skill score is biased against the forecaster being evaluated, and very noisy." : "With a long test set the bias is small.");
        }
        simulate();
        render();
        return lifecycle(el, api, render, stop);
    });
})();
