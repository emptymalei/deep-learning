/*
 * Four explainers for the page "Leakage and Contamination".
 *
 *   feature-timeline  which features are known at the forecast origin
 *   split-leak        the same local model scored with and without access
 *                     to the observation after each test point
 *   selection-leak    picking the best of many noise features on all the
 *                     data, against picking it on the training part only
 *   pretrain-overlap  a model that memorizes its pretraining corpus, scored
 *                     on a benchmark that partly overlaps with the corpus
 *
 * All data is simulated in the browser from seeded random numbers. The
 * feature list in feature-timeline is an illustration, not from a dataset.
 */
(function () {
    const { rng, node, html, slider, button, legend, widthOf, lifecycle, polyline } = window.dlWidgetKit;
    const mean = (a) => a.reduce((s, v) => s + v, 0) / (a.length || 1);

    // ==========================================================================
    // feature-timeline
    // ==========================================================================
    dlWidgets.register("feature-timeline", (el, api) => {
        const state = { h: 3, delay: 5, playing: false };
        // [name, day the value refers to, day it becomes known]; the target day is 0.
        const features = () => [
            ["Promotion plan for the target day", 0, -14],
            ["Weather forecast for the target day", 0, -state.h],
            ["Sales seven days before the target", -7, -7],
            ["Price index, period ending on day −10", -10, -10 + state.delay],
            ["Sales on the day before the target", -1, -1],
            ["Observed weather on the target day", 0, 0],
            ["Returns of the target day's sales", 0, 7],
        ];
        el.innerHTML = "";
        const controls = html("div", "dl-widget__controls", el);
        const hS = slider(controls, "Horizon h (days)", 1, 10, 1, state.h, 0, (v) => { state.h = v; stop(); render(); });
        slider(controls, "Publication delay (days)", 0, 12, 1, state.delay, 0, (v) => { state.delay = v; stop(); render(); });
        const play = button(controls, "Play", () => (state.playing ? stop() : start()));
        const svg = node("svg", { role: "img" }, el);
        const legendBox = html("div", "dl-widget__legend", el);
        const caption = html("p", "dl-widget__caption", el);
        caption.setAttribute("aria-live", "polite");
        let timer = 0;
        function start() {
            state.playing = true;
            play.textContent = "Pause";
            state.h = 0;
            const tick = () => {
                state.h += 1;
                hS.set(state.h);
                render();
                if (state.h >= 10) return stop();
                timer = setTimeout(tick, 900);
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
            const narrow = W < 520;
            const list = features();
            const rowH = narrow ? 34 : 24;
            const nameW = narrow ? 0 : 236;
            const padR = 12, top = 30;
            const H = top + list.length * rowH + 24;
            svg.setAttribute("viewBox", `0 0 ${W} ${H}`);
            svg.innerHTML = "";
            const D0 = -15, D1 = 9;
            const x = (d) => nameW + 10 + ((W - nameW - 10 - padR) * (Math.max(D0, Math.min(D1, d)) - D0)) / (D1 - D0);
            const origin = -state.h;
            // What is known at the origin lies to the left of it.
            node("rect", { x: x(D0), y: top - 6, width: x(origin) - x(D0), height: list.length * rowH + 6, fill: c.train, "fill-opacity": 0.1 }, svg);
            for (const [d, name, col] of [[origin, "forecast made", c.train], [0, "target day", c.fg]]) {
                node("line", { x1: x(d), x2: x(d), y1: top - 10, y2: top + list.length * rowH, stroke: col, "stroke-width": 2 }, svg);
                node("text", { x: x(d), y: top - 14, "text-anchor": d === 0 ? "start" : "end", "font-size": 10, "font-weight": 700, fill: col }, svg, name);
            }
            let ok = 0;
            list.forEach(([name, ref, known], i) => {
                const y = top + i * rowH + (narrow ? 22 : rowH / 2);
                const legit = known <= origin;
                ok += legit;
                const col = legit ? c.train : c.accent;
                node("text", narrow ? { x: 4, y: y - 10, "font-size": 10, fill: c.fg } : { x: nameW, y: y + 3.5, "text-anchor": "end", "font-size": 10.5, fill: c.fg }, svg, name);
                node("line", { x1: x(ref), x2: x(known), y1: y, y2: y, stroke: col, "stroke-width": 1.5, "stroke-dasharray": "3 3" }, svg);
                node("circle", { cx: x(ref), cy: y, r: 3.5, fill: "none", stroke: col, "stroke-width": 1.5 }, svg);
                node("rect", { x: x(known) - 4.5, y: y - 4.5, width: 9, height: 9, fill: col }, svg);
                node("text", { x: W - 2, y: y + 3.5, "text-anchor": "end", "font-size": 10, "font-weight": 700, fill: col }, svg, legit ? "ok" : "leak");
            });
            for (let d = -14; d <= 8; d += 2) node("text", { x: x(d), y: H - 6, "text-anchor": "middle", "font-size": 9.5, fill: c.fgLight }, svg, d);
            legend(legendBox, [
                ["known when the forecast is made: legitimate", c.train],
                ["known only later: using it is leakage", c.accent],
                ["circle: the day the value refers to; square: the day it becomes known", c.fgLight],
            ]);
            svg.setAttribute("aria-label", `Seven candidate features on a timeline for a forecast made ${state.h} days before the target day.`);
            caption.textContent =
                `A forecast made ${state.h} day${state.h > 1 ? "s" : ""} before the target day may use ${ok} of these ${list.length} features. ` +
                (state.h > 1 ? "Yesterday's sales are a fine feature for a one-day horizon and a leak for any longer one. " : "") +
                (-10 + state.delay > origin ? "The price index refers to a period long past, but it is published after the forecast is made, so the forecaster does not have it." : "What decides is the day a value becomes known, not the day it refers to.");
        }
        render();
        return lifecycle(el, api, render, stop);
    });

    // ==========================================================================
    // split-leak
    // ==========================================================================
    dlWidgets.register("split-leak", (el, api) => {
        const N = 400;
        const SHOW = 90;
        const state = { rho: 0.95, seed: 3, runs: 0, sumLeak: 0, sumHonest: 0, playing: false };
        let xs = [], test = [], leakPred = [], honestPred = [], cur = { leak: 0, honest: 0 };
        function simulate() {
            const g = rng(state.seed);
            xs = [0];
            const s = Math.sqrt(1 - state.rho * state.rho);
            for (let t = 1; t < N; t++) xs.push(state.rho * xs[t - 1] + s * g.normal());
            // Random split: a quarter of the points become test points.
            test = xs.map((_, t) => t > 0 && t < N - 1 && g.u() < 0.25);
            leakPred = [];
            honestPred = [];
            const eL = [], eH = [];
            for (let t = 1; t < N - 1; t++) {
                if (!test[t]) continue;
                let a = t - 1, b = t + 1;
                while (a > 0 && test[a]) a--;
                while (b < N - 1 && test[b]) b++;
                // Leaky: the nearest training points on both sides. Honest: the last value before t.
                leakPred[t] = 0.5 * (xs[a] + xs[b]);
                honestPred[t] = xs[t - 1];
                eL.push(Math.abs(xs[t] - leakPred[t]));
                eH.push(Math.abs(xs[t] - honestPred[t]));
            }
            cur = { leak: mean(eL), honest: mean(eH) };
            state.runs++;
            state.sumLeak += cur.leak;
            state.sumHonest += cur.honest;
        }
        const reset = () => {
            stop();
            state.runs = state.sumLeak = state.sumHonest = 0;
            simulate();
            render();
        };
        el.innerHTML = "";
        const controls = html("div", "dl-widget__controls", el);
        slider(controls, "Autocorrelation ρ", 0, 0.99, 0.01, state.rho, 2, (v) => { state.rho = v; reset(); });
        button(controls, "New series", () => { stop(); state.seed++; simulate(); render(); });
        const play = button(controls, "Play", () => (state.playing ? stop() : start()));
        const svg = node("svg", { role: "img" }, el);
        const legendBox = html("div", "dl-widget__legend", el);
        const caption = html("p", "dl-widget__caption", el);
        caption.setAttribute("aria-live", "polite");
        let timer = 0;
        function start() {
            state.playing = true;
            play.textContent = "Pause";
            const tick = () => {
                state.seed++;
                simulate();
                render();
                if (state.runs >= 200) return stop();
                timer = setTimeout(tick, state.runs < 6 ? 600 : 70);
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
            const padL = 8, padR = 8;
            const topH = 170, gap = 34, barH = 16;
            const H = topH + gap + 2 * (barH + 20) + 4;
            svg.setAttribute("viewBox", `0 0 ${W} ${H}`);
            svg.innerHTML = "";
            const seg = xs.slice(0, SHOW);
            const lo = Math.min(...seg) - 0.3, hi = Math.max(...seg) + 0.3;
            const x = (t) => padL + ((W - padL - padR) * t) / (SHOW - 1);
            const y = (v) => 6 + (topH - 12) * (1 - (v - lo) / (hi - lo));
            node("polyline", { points: polyline(seg.map((v, t) => [x(t), y(v)])), fill: "none", stroke: c.fgLighter, "stroke-width": 1.2 }, svg);
            seg.forEach((v, t) => {
                if (test[t]) {
                    node("line", { x1: x(t), x2: x(t), y1: y(v), y2: y(leakPred[t]), stroke: c.accent, "stroke-width": 1.5 }, svg);
                    node("circle", { cx: x(t), cy: y(leakPred[t]), r: 2.6, fill: c.accent }, svg);
                    node("circle", { cx: x(t), cy: y(honestPred[t]), r: 2.6, fill: c.train }, svg);
                    node("circle", { cx: x(t), cy: y(v), r: 3.6, fill: "none", stroke: c.fg, "stroke-width": 1.5 }, svg);
                } else {
                    node("circle", { cx: x(t), cy: y(v), r: 1.8, fill: c.fgLight }, svg);
                }
            });
            node("text", { x: padL, y: topH + 12, "font-size": 10, fill: c.fgLight }, svg, `first ${SHOW} of ${N} time steps`);
            const full = W - padL - padR;
            const top = Math.max(cur.honest, cur.leak, state.sumHonest / state.runs) * 1.1;
            let yb = topH + gap;
            for (const [name, v, avg, col] of [
                ["Random split, model sees both neighbours", cur.leak, state.sumLeak / state.runs, c.accent],
                ["Forecasting forward, model sees the past only", cur.honest, state.sumHonest / state.runs, c.train],
            ]) {
                node("text", { x: padL, y: yb - 4, "font-size": 10, fill: c.fg }, svg, `${name}: MAE ${v.toFixed(3)}` + (state.runs > 1 ? ` (average of ${state.runs} series: ${avg.toFixed(3)})` : ""));
                node("rect", { x: padL, y: yb, width: full, height: barH, fill: c.fgLightest }, svg);
                node("rect", { x: padL, y: yb, width: (full * v) / top, height: barH, fill: col, "fill-opacity": 0.75 }, svg);
                yb += barH + 20;
            }
            legend(legendBox, [
                ["training points", c.fgLight],
                ["test points (open circles)", c.fg],
                ["prediction from the neighbours on both sides", c.accent],
                ["prediction from the previous value only", c.train],
            ]);
            const ratio = cur.honest / cur.leak;
            svg.setAttribute("aria-label", `An autocorrelated series with randomly chosen test points and two predictions for each.`);
            caption.textContent =
                `Same series, same test points, same kind of model. With access to the value after each test point the error is ${cur.leak.toFixed(3)}; without it, ${cur.honest.toFixed(3)}` +
                (ratio > 1.05 ? `, ${ratio.toFixed(2)} times larger. The random split reports an accuracy that no forecast made in real time can have.` : ". With little autocorrelation the neighbours carry no information, and the leak does no harm.");
        }
        simulate();
        render();
        return lifecycle(el, api, render, stop);
    });

    // ==========================================================================
    // selection-leak
    // ==========================================================================
    dlWidgets.register("selection-leak", (el, api) => {
        const state = { n: 100, p: 200, leak: [], clean: [], playing: false };
        let g = rng(19);
        function corr(a, b, from, to) {
            let sa = 0, sb = 0, saa = 0, sbb = 0, sab = 0;
            const m = to - from;
            for (let i = from; i < to; i++) {
                sa += a[i]; sb += b[i]; saa += a[i] * a[i]; sbb += b[i] * b[i]; sab += a[i] * b[i];
            }
            return (sab - (sa * sb) / m) / Math.sqrt((saa - (sa * sa) / m) * (sbb - (sb * sb) / m));
        }
        function experiment() {
            const { n, p } = state;
            const half = n >> 1;
            const y = Float64Array.from({ length: n }, () => g.normal());
            let bestAll = -1, bestTrain = -1, testOfAll = 0, testOfTrain = 0;
            for (let j = 0; j < p; j++) {
                // Every feature is pure noise: none of them can predict y.
                const f = Float64Array.from({ length: n }, () => g.normal());
                const cTrain = Math.abs(corr(f, y, 0, half));
                const cAll = Math.abs(corr(f, y, 0, n));
                const cTest = Math.abs(corr(f, y, half, n));
                if (cAll > bestAll) { bestAll = cAll; testOfAll = cTest; }
                if (cTrain > bestTrain) { bestTrain = cTrain; testOfTrain = cTest; }
            }
            state.leak.push(testOfAll);
            state.clean.push(testOfTrain);
        }
        const reset = () => {
            stop();
            state.leak = [];
            state.clean = [];
            experiment();
            render();
        };
        el.innerHTML = "";
        const controls = html("div", "dl-widget__controls", el);
        slider(controls, "Samples n", 40, 400, 20, state.n, 0, (v) => { state.n = v; reset(); });
        slider(controls, "Noise features p", 5, 500, 5, state.p, 0, (v) => { state.p = v; reset(); });
        button(controls, "New experiment", () => { stop(); experiment(); render(); });
        const play = button(controls, "Play", () => (state.playing ? stop() : start()));
        const svg = node("svg", { role: "img" }, el);
        const legendBox = html("div", "dl-widget__legend", el);
        const caption = html("p", "dl-widget__caption", el);
        caption.setAttribute("aria-live", "polite");
        let timer = 0;
        function start() {
            state.playing = true;
            play.textContent = "Pause";
            const tick = () => {
                const batch = state.leak.length < 6 ? 1 : 4;
                for (let i = 0; i < batch; i++) experiment();
                render();
                if (state.leak.length >= 300) return stop();
                timer = setTimeout(tick, state.leak.length < 6 ? 500 : 50);
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
            const padL = 10, padR = 14, rowH = 78;
            const H = 2 * rowH + 34;
            svg.setAttribute("viewBox", `0 0 ${W} ${H}`);
            svg.innerHTML = "";
            const xmax = Math.max(0.5, ...state.leak, ...state.clean) * 1.05;
            const x = (v) => padL + ((W - padL - padR) * v) / xmax;
            [["Feature chosen on all the data, scored on the test half", state.leak, c.accent], ["Feature chosen on the training half, scored on the test half", state.clean, c.train]].forEach(([name, vals, col], k) => {
                const y0 = 16 + k * rowH;
                node("text", { x: padL, y: y0, "font-size": 10.5, fill: c.fg }, svg, `${name}: mean |r| = ${mean(vals).toFixed(3)}`);
                node("line", { x1: padL, x2: W - padR, y1: y0 + 50, y2: y0 + 50, stroke: c.fgLighter }, svg);
                vals.forEach((v, i) => node("circle", { cx: x(v), cy: y0 + 46 - ((i * 5) % 34), r: i === vals.length - 1 ? 4.5 : 2.4, fill: col, "fill-opacity": i === vals.length - 1 ? 1 : 0.5 }, svg));
                node("line", { x1: x(mean(vals)), x2: x(mean(vals)), y1: y0 + 8, y2: y0 + 54, stroke: c.fg, "stroke-width": 2 }, svg);
            });
            for (let v = 0; v <= xmax; v += 0.1) node("text", { x: x(v), y: H - 6, "text-anchor": "middle", "font-size": 10, fill: c.fgLight }, svg, v.toFixed(1));
            node("text", { x: W - padR, y: H - 20, "text-anchor": "end", "font-size": 10, fill: c.fgLight }, svg, "|correlation| with the target on the test half");
            legend(legendBox, [
                ["selection saw the test half", c.accent],
                ["selection saw the training half only", c.train],
                ["mean over the experiments", c.fg],
            ]);
            svg.setAttribute("aria-label", `Test correlation of the best of ${state.p} noise features chosen with and without the test data, over ${state.leak.length} experiments.`);
            caption.textContent =
                `${state.leak.length} experiment${state.leak.length > 1 ? "s" : ""}, each with ${state.p} features of pure noise and ${state.n} samples, half of them for testing. ` +
                `No feature has any relation to the target. Chosen cleanly, the best feature has a test correlation of ${mean(state.clean).toFixed(2)} on average, which is what chance gives for ${state.n >> 1} test samples. ` +
                `Chosen on all the data, it has ${mean(state.leak).toFixed(2)}: the test half helped to pick it, so it looks predictive on the test half.`;
        }
        experiment();
        render();
        return lifecycle(el, api, render, stop);
    });

    // ==========================================================================
    // pretrain-overlap
    // ==========================================================================
    dlWidgets.register("pretrain-overlap", (el, api) => {
        const CTX = 48, HOR = 16, NB = 40, NC = 150;
        const state = { share: 0, rival: 0.8, playing: false };
        const g = rng(2024);
        function makeSeries(len) {
            const period = 8 + 22 * g.u(), amp = 0.5 + g.u(), ph = 2 * Math.PI * g.u(), slope = (g.u() - 0.5) * 0.02;
            return Array.from({ length: len }, (_, t) => amp * Math.sin((2 * Math.PI * t) / period + ph) + slope * t + 0.15 * g.normal());
        }
        const bench = Array.from({ length: NB }, () => makeSeries(CTX + HOR));
        // The clean corpus: other series from the same generator, cut into windows.
        const windows = [];
        for (let i = 0; i < NC; i++) {
            const s = makeSeries(CTX + HOR + 56);
            for (let o = 0; o <= 56; o += 4) windows.push(s.slice(o, o + CTX + HOR));
        }
        /** The memorizing model: return what followed the most similar context in the corpus. */
        function recall(context) {
            let best = Infinity, out = null;
            for (const w of windows) {
                let d = 0;
                for (let t = 0; t < CTX && d < best; t++) d += (w[t] - context[t]) * (w[t] - context[t]);
                if (d < best) { best = d; out = w; }
            }
            return out.slice(CTX);
        }
        const unseen = bench.map((s) => recall(s.slice(0, CTX)));
        const mae = (s, f) => mean(f.map((v, t) => Math.abs(v - s[CTX + t])));
        const errUnseen = bench.map((s, i) => mae(s, unseen[i]));
        const clean = mean(errUnseen);
        // Which benchmark series enter the corpus as the leaked share grows: a fixed random order.
        const order = bench.map((_, i) => i).sort((a, b) => ((a * 7919) % NB) - ((b * 7919) % NB));

        el.innerHTML = "";
        const controls = html("div", "dl-widget__controls", el);
        const sS = slider(controls, "Benchmark series in the pretraining data (%)", 0, 100, 5, state.share, 0, (v) => { state.share = v; stop(); render(); });
        slider(controls, "Rival's error (× the memorizer's honest error)", 0.5, 1, 0.05, state.rival, 2, (v) => { state.rival = v; stop(); render(); });
        const play = button(controls, "Play", () => (state.playing ? stop() : start()));
        const svg = node("svg", { role: "img" }, el);
        const legendBox = html("div", "dl-widget__legend", el);
        const caption = html("p", "dl-widget__caption", el);
        caption.setAttribute("aria-live", "polite");
        let timer = 0;
        function start() {
            state.playing = true;
            play.textContent = "Pause";
            state.share = -5;
            const tick = () => {
                state.share += 5;
                sS.set(state.share);
                render();
                if (state.share >= 100) return stop();
                timer = setTimeout(tick, 350);
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
            const padL = 8, padR = 8;
            const topH = 150, gap = 40, botH = 110, padB = 28;
            const H = topH + gap + botH + padB;
            svg.setAttribute("viewBox", `0 0 ${W} ${H}`);
            svg.innerHTML = "";
            const nLeak = Math.round((state.share / 100) * NB);
            const leaked = new Set(order.slice(0, nLeak));
            // A series that is in the corpus is recalled exactly.
            const score = mean(bench.map((_, i) => (leaked.has(i) ? 0 : errUnseen[i])));
            const rival = state.rival * clean;

            // Top: one benchmark series, with the forecast when it was not seen and when it was.
            const ex = order[0];
            const s = bench[ex];
            const lo = Math.min(...s, ...unseen[ex]) - 0.2, hi = Math.max(...s, ...unseen[ex]) + 0.2;
            const x = (t) => padL + ((W - padL - padR) * t) / (CTX + HOR - 1);
            const y = (v) => 14 + (topH - 20) * (1 - (v - lo) / (hi - lo));
            node("rect", { x: x(CTX - 0.5), y: 4, width: x(CTX + HOR - 1) - x(CTX - 0.5), height: topH - 4, fill: c.fgLightest }, svg);
            node("text", { x: x(CTX), y: 13, "font-size": 10, fill: c.fgLight }, svg, "test horizon");
            node("polyline", { points: polyline(s.map((v, t) => [x(t), y(v)])), fill: "none", stroke: c.fg, "stroke-width": 1.5 }, svg);
            const seen = leaked.has(ex);
            const f = seen ? s.slice(CTX) : unseen[ex];
            node("polyline", { points: polyline(f.map((v, t) => [x(CTX + t), y(v)])), fill: "none", stroke: seen ? c.accent : c.train, "stroke-width": 2.4, "stroke-dasharray": seen ? "" : "5 3" }, svg);
            node("text", { x: padL, y: topH + 13, "font-size": 10, fill: c.fgLight }, svg, `one benchmark series: ${seen ? "it is in the pretraining data, and the forecast is a recall" : "not in the pretraining data"}`);

            // Bottom: the benchmark score over the leaked share.
            const y0 = topH + gap;
            const xb = (p) => padL + 26 + ((W - padL - padR - 30) * p) / 100;
            const yb = (v) => y0 + botH * (1 - v / (clean * 1.15));
            node("line", { x1: xb(0), x2: xb(100), y1: yb(0), y2: yb(0), stroke: c.fgLighter }, svg);
            const curve = [];
            for (let p = 0; p <= 100; p += 5) {
                const set = new Set(order.slice(0, Math.round((p / 100) * NB)));
                curve.push([xb(p), yb(mean(bench.map((_, i) => (set.has(i) ? 0 : errUnseen[i]))))]);
            }
            node("polyline", { points: polyline(curve), fill: "none", stroke: c.accent, "stroke-width": 2.2 }, svg);
            node("line", { x1: xb(0), x2: xb(100), y1: yb(clean), y2: yb(clean), stroke: c.accent, "stroke-dasharray": "5 4" }, svg);
            node("text", { x: xb(100), y: yb(clean) - 4, "text-anchor": "end", "font-size": 10, fill: c.accent }, svg, "memorizer on series it has never seen");
            node("line", { x1: xb(0), x2: xb(100), y1: yb(rival), y2: yb(rival), stroke: c.train, "stroke-width": 2 }, svg);
            node("text", { x: xb(100), y: yb(rival) + 12, "text-anchor": "end", "font-size": 10, fill: c.train }, svg, "rival that saw none of the benchmark");
            node("circle", { cx: xb(state.share), cy: yb(score), r: 5, fill: c.accent }, svg);
            for (const p of [0, 25, 50, 75, 100]) node("text", { x: xb(p), y: yb(0) + 13, "text-anchor": "middle", "font-size": 10, fill: c.fgLight }, svg, p + "%");
            node("text", { x: xb(50), y: H - 3, "text-anchor": "middle", "font-size": 10, fill: c.fgLight }, svg, "share of the benchmark that is in the pretraining data");
            node("text", { x: padL, y: y0 + 4, "font-size": 10, fill: c.fgLight }, svg, "MAE");

            legend(legendBox, [
                ["benchmark series", c.fg],
                ["memorizing model: its benchmark score", c.accent],
                ["its forecast for a series it has not seen", c.train],
            ]);
            svg.setAttribute("aria-label", `Benchmark error of a memorizing model when ${state.share} percent of the benchmark series are in its pretraining data.`);
            caption.textContent =
                `${nLeak} of ${NB} benchmark series are in the pretraining data. Benchmark MAE of the memorizer: ${score.toFixed(3)}; on unseen series it is ${clean.toFixed(3)}. The rival scores ${rival.toFixed(3)}. ` +
                (score < rival
                    ? "The benchmark ranks the memorizer first, although the rival is the better model on series that neither has seen."
                    : "The benchmark still ranks the rival first; raise the share to see the ranking flip.");
        }
        render();
        return lifecycle(el, api, render, stop);
    });
})();
