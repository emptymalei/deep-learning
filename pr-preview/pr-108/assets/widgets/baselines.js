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
    const { rng, node, html, slider, button, legend, widthOf, lifecycle, polyline } = window.dlWidgetKit;
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
})();
