/*
 * Four explainers for the page "Is the Difference Real?".
 *
 *   run-noise       two models whose scores vary from run to run: how often
 *                   the worse one wins a comparison of k-run averages
 *   dm-test         the Diebold-Mariano test on a simulated loss differential,
 *                   against a z-test that ignores serial correlation
 *   signed-rank     the Wilcoxon signed-ranks test on the 14 data sets of
 *                   Demšar (2006), Table 2, next to the paired t statistic
 *   multiple-tests  many comparisons at once: the family-wise error with no
 *                   correction, with Bonferroni's and with Holm's
 *
 * Everything is simulated in the browser from seeded random numbers.
 */
(function () {
    const { Phi, PhiInv, phi, rng, node, html, slider, seg, button, legend, widthOf, lifecycle, polyline, range } = window.dlWidgetKit;
    const pTwoSided = (z) => 2 * (1 - Phi(Math.abs(z)));

    // ==========================================================================
    // run-noise
    // ==========================================================================
    dlWidgets.register("run-noise", (el, api) => {
        const state = { delta: 0.05, sigma: 0.1, k: 3, playing: false };
        const MEAN_B = 1;
        let r = rng(11);
        let runsA = [], runsB = [], diffs = [];
        const draw = () => {
            runsA = Array.from({ length: state.k }, () => MEAN_B - state.delta + state.sigma * r.normal());
            runsB = Array.from({ length: state.k }, () => MEAN_B + state.sigma * r.normal());
            const mean = (a) => a.reduce((s, v) => s + v, 0) / a.length;
            diffs.push(mean(runsB) - mean(runsA));
        };
        const reset = () => {
            stop();
            diffs = [];
            draw();
            render();
        };
        el.innerHTML = "";
        const controls = html("div", "dl-widget__controls", el);
        slider(controls, "True advantage of A", 0, 0.3, 0.01, state.delta, 2, (v) => { state.delta = v; reset(); });
        slider(controls, "Run-to-run spread σ", 0.02, 0.3, 0.01, state.sigma, 2, (v) => { state.sigma = v; reset(); });
        slider(controls, "Runs per model k", 1, 30, 1, state.k, 0, (v) => { state.k = v; reset(); });
        button(controls, "New experiment", () => { stop(); draw(); render(); });
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
                draw();
                render();
                if (diffs.length >= 400) return stop();
                timer = setTimeout(tick, diffs.length < 8 ? 500 : 60);
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
            const { delta, sigma, k } = state;
            const padL = 12, padR = 12;
            const topH = 120, gap = 40, botH = 120, padB = 30;
            const H = topH + gap + botH + padB;
            svg.setAttribute("viewBox", `0 0 ${W} ${H}`);
            svg.innerHTML = "";

            // Top: the distribution of a single run for each model, and this experiment's runs.
            const lo = MEAN_B - 0.3 - 3 * 0.3, hi = MEAN_B + 3 * 0.3;
            const x = (v) => padL + ((W - padL - padR) * (Math.max(lo, Math.min(hi, v)) - lo)) / (hi - lo);
            const xs = range(lo, hi, 220);
            const dmax = phi(0) / sigma;
            const yT = (d) => 8 + (topH - 30) * (1 - d / dmax);
            const base = topH - 22;
            node("line", { x1: padL, x2: W - padR, y1: base, y2: base, stroke: c.fgLighter }, svg);
            for (const [mu, col, runs, row, name] of [[MEAN_B - delta, c.train, runsA, 0, "A"], [MEAN_B, c.accent, runsB, 1, "B"]]) {
                node("polyline", { points: polyline(xs.map((v) => [x(v), yT(phi((v - mu) / sigma) / sigma)])), fill: "none", stroke: col, "stroke-width": 2 }, svg);
                const y = base + 7 + row * 9;
                for (const v of runs) node("circle", { cx: x(v), cy: y, r: 3, fill: col, "fill-opacity": 0.8 }, svg);
                const m = runs.reduce((s, v) => s + v, 0) / runs.length;
                node("line", { x1: x(m), x2: x(m), y1: 10, y2: base, stroke: col, "stroke-dasharray": "4 3" }, svg);
                node("text", { x: x(m), y: 8, "text-anchor": "middle", "font-size": 10, fill: col }, svg, `mean of ${name}`);
            }
            for (const v of [0.4, 0.7, 1.0, 1.3, 1.6]) node("text", { x: x(v), y: topH + 12, "text-anchor": "middle", "font-size": 10, fill: c.fgLight }, svg, v.toFixed(1));
            node("text", { x: W - padR, y: topH + 12, "text-anchor": "end", "font-size": 10, fill: c.fgLight }, svg, "error of one run");

            // Bottom: the difference of the two averages, over repeated experiments.
            const se = sigma * Math.sqrt(2 / k);
            const dl = Math.min(-0.25, delta - 4 * se), dh = Math.max(0.45, delta + 4 * se);
            const xd = (v) => padL + ((W - padL - padR) * (Math.max(dl, Math.min(dh, v)) - dl)) / (dh - dl);
            const y0 = topH + gap;
            const ds = range(dl, dh, 240);
            const yD = (d) => y0 + botH * (1 - (d * se) / phi(0));
            const wrong = ds.filter((v) => v <= 0);
            node("polygon", { points: polyline(wrong.map((v) => [xd(v), yD(phi((v - delta) / se) / se)]).concat([[xd(0), y0 + botH], [xd(dl), y0 + botH]])), fill: c.accent, "fill-opacity": 0.3 }, svg);
            node("polyline", { points: polyline(ds.map((v) => [xd(v), yD(phi((v - delta) / se) / se)])), fill: "none", stroke: c.fg, "stroke-width": 1.6 }, svg);
            node("line", { x1: padL, x2: W - padR, y1: y0 + botH, y2: y0 + botH, stroke: c.fgLighter }, svg);
            node("line", { x1: xd(0), x2: xd(0), y1: y0, y2: y0 + botH + 4, stroke: c.fg, "stroke-dasharray": "4 3" }, svg);
            node("text", { x: xd(0), y: y0 + botH + 14, "text-anchor": "middle", "font-size": 10, fill: c.fg }, svg, "0");
            node("text", { x: xd(0) - 6, y: y0 + 10, "text-anchor": "end", "font-size": 10, fill: c.accent }, svg, "B looks better");
            node("text", { x: xd(0) + 6, y: y0 + 10, "font-size": 10, fill: c.train }, svg, "A looks better");
            diffs.forEach((d, i) => {
                const last = i === diffs.length - 1;
                node("circle", { cx: xd(d), cy: y0 + botH - 5 - ((i * 7) % (botH - 30)), r: last ? 4.5 : 2.4, fill: d <= 0 ? c.accent : c.train, "fill-opacity": last ? 1 : 0.55 }, svg);
            });
            node("text", { x: (padL + W - padR) / 2, y: H - 3, "text-anchor": "middle", "font-size": 10, fill: c.fgLight }, svg, "average error of B minus average error of A, one dot per experiment");

            legend(legendBox, [
                ["model A (truly better when the advantage is above 0)", c.train],
                ["model B, and experiments in which B looks better", c.accent],
                ["distribution of the difference of the two averages", c.fg],
            ]);
            const pRun = Phi(delta / (sigma * Math.SQRT2));
            const pWrong = Phi(-delta / se);
            const nWrong = diffs.filter((d) => d <= 0).length;
            svg.setAttribute("aria-label", `Two models with a true difference of ${delta.toFixed(2)} and run-to-run spread ${sigma.toFixed(2)}, compared with ${k} runs each.`);
            caption.textContent =
                `A single run of A beats a single run of B with probability ${(100 * pRun).toFixed(0)}%. ` +
                `With ${k} run${k > 1 ? "s" : ""} per model, the standard error of the difference is ${se.toFixed(3)}, ` +
                (delta > 0
                    ? `and the worse model has the better average in ${(100 * pWrong).toFixed(1)}% of experiments. `
                    : "and the models are equal, so each of them wins half of the experiments. ") +
                `So far: B looked better in ${nWrong} of ${diffs.length} experiment${diffs.length > 1 ? "s" : ""}. ` +
                `A difference has to exceed ${(1.645 * se).toFixed(3)} before a one-sided z-test at the 5% level calls it.`;
        }
        draw();
        render();
        return lifecycle(el, api, render, stop);
    });

    // ==========================================================================
    // dm-test
    // ==========================================================================
    dlWidgets.register("dm-test", (el, api) => {
        const state = { mu: 0, h: 4, T: 120, seed: 5, reps: 0, rejNaive: 0, rejDM: 0, playing: false };
        /** Loss differential of h-step forecasts: mean mu plus a moving average of h shocks, variance 1. */
        function series(g, mu, h, T) {
            const e = Array.from({ length: T + h }, () => g.normal());
            const d = new Array(T);
            for (let t = 0; t < T; t++) {
                let s = 0;
                for (let j = 0; j < h; j++) s += e[t + j];
                d[t] = mu + s / Math.sqrt(h);
            }
            return d;
        }
        function test(d, h) {
            const T = d.length;
            const mean = d.reduce((s, v) => s + v, 0) / T;
            const gamma = (lag) => {
                let s = 0;
                for (let t = lag; t < T; t++) s += (d[t] - mean) * (d[t - lag] - mean);
                return s / T;
            };
            const g0 = gamma(0);
            let lrv = g0;
            for (let lag = 1; lag < h; lag++) lrv += 2 * gamma(lag);
            const seNaive = Math.sqrt(g0 / T);
            // Diebold and Mariano: a negative estimate is treated as zero and the null is rejected.
            const seDM = lrv > 0 ? Math.sqrt(lrv / T) : 0;
            return { mean, seNaive, seDM, zNaive: mean / seNaive, zDM: seDM ? mean / seDM : Infinity };
        }
        let d = [];
        const fresh = () => {
            stop();
            state.reps = state.rejNaive = state.rejDM = 0;
            d = series(rng(state.seed), state.mu, state.h, state.T);
            render();
        };
        el.innerHTML = "";
        const controls = html("div", "dl-widget__controls", el);
        slider(controls, "True mean difference", -0.5, 0.5, 0.05, state.mu, 2, (v) => { state.mu = v; fresh(); });
        slider(controls, "Horizon h", 1, 8, 1, state.h, 0, (v) => { state.h = v; fresh(); });
        slider(controls, "Forecasts T", 30, 300, 10, state.T, 0, (v) => { state.T = v; fresh(); });
        button(controls, "New sample", () => { state.seed += 1; const keep = [state.reps, state.rejNaive, state.rejDM]; fresh(); [state.reps, state.rejNaive, state.rejDM] = keep; render(); });
        const play = button(controls, "Repeat 2000 times", () => (state.playing ? stop() : start()));
        const svg = node("svg", { role: "img" }, el);
        const legendBox = html("div", "dl-widget__legend", el);
        const caption = html("p", "dl-widget__caption", el);
        caption.setAttribute("aria-live", "polite");
        const note = html("p", "dl-widget__caption", el);

        let timer = 0;
        let g = null;
        function start() {
            state.playing = true;
            play.textContent = "Pause";
            if (state.reps >= 2000) state.reps = state.rejNaive = state.rejDM = 0;
            if (!state.reps) g = rng(4242);
            const tick = () => {
                for (let i = 0; i < 40 && state.reps < 2000; i++) {
                    const res = test(series(g, state.mu, state.h, state.T), state.h);
                    state.reps++;
                    state.rejNaive += Math.abs(res.zNaive) > 1.96;
                    state.rejDM += Math.abs(res.zDM) > 1.96;
                }
                render();
                if (state.reps >= 2000) return stop();
                timer = setTimeout(tick, 60);
            };
            tick();
        }
        function stop() {
            clearTimeout(timer);
            state.playing = false;
            play.textContent = "Repeat 2000 times";
        }

        let W = 640;
        function render() {
            const c = api.colors();
            W = widthOf(el);
            const padL = 30, padR = 10;
            const topH = 150, gap = 34, barH = 16;
            const H = topH + gap + 2 * (barH + 20) + 6;
            svg.setAttribute("viewBox", `0 0 ${W} ${H}`);
            svg.innerHTML = "";
            const res = test(d, state.h);
            const T = d.length;
            const x = (t) => padL + ((W - padL - padR) * t) / (T - 1);
            const y = (v) => 6 + (topH - 12) * (1 - (Math.max(-3.5, Math.min(3.5, v)) + 3.5) / 7);
            for (const v of [-2, 0, 2]) {
                node("line", { x1: padL, x2: W - padR, y1: y(v), y2: y(v), stroke: v ? c.fgLightest : c.fgLighter }, svg);
                node("text", { x: padL - 5, y: y(v) + 3, "text-anchor": "end", "font-size": 10, fill: c.fgLight }, svg, v);
            }
            // Bars above zero: forecast 1 lost more at that time; below: forecast 2 lost more.
            const bw = Math.max(1, (W - padL - padR) / T - 1);
            d.forEach((v, t) => node("rect", { x: x(t) - bw / 2, width: bw, y: Math.min(y(0), y(v)), height: Math.abs(y(v) - y(0)), fill: v > 0 ? c.accent : c.train, "fill-opacity": 0.6 }, svg));
            node("line", { x1: padL, x2: W - padR, y1: y(res.mean), y2: y(res.mean), stroke: c.fg, "stroke-width": 2 }, svg);
            node("text", { x: W - padR, y: y(res.mean) - 5, "text-anchor": "end", "font-size": 11, fill: c.fg }, svg, `mean = ${res.mean.toFixed(3)}`);
            node("text", { x: padL, y: topH + 12, "font-size": 10, fill: c.fgLight }, svg, "loss differential d over time");

            // Rejection rates over the repeated experiments.
            let yb = topH + gap;
            const full = W - padL - padR;
            const scale = (p) => full * Math.min(1, p / 0.5);
            for (const [name, k, col] of [["z-test that ignores serial correlation", state.rejNaive, c.accent], ["Diebold-Mariano test", state.rejDM, c.train]]) {
                const p = state.reps ? k / state.reps : 0;
                node("text", { x: padL, y: yb - 4, "font-size": 10, fill: c.fg }, svg, state.reps ? `${name}: rejects in ${(100 * p).toFixed(1)}% of ${state.reps} samples` : `${name}: press "Repeat 2000 times"`);
                node("rect", { x: padL, y: yb, width: full, height: barH, fill: c.fgLightest }, svg);
                node("rect", { x: padL, y: yb, width: scale(p), height: barH, fill: col, "fill-opacity": 0.75 }, svg);
                node("line", { x1: padL + scale(0.05), x2: padL + scale(0.05), y1: yb - 2, y2: yb + barH + 2, stroke: c.fg, "stroke-width": 1.5 }, svg);
                yb += barH + 20;
            }
            legend(legendBox, [
                ["forecast 1 had the larger loss", c.accent],
                ["forecast 2 had the larger loss", c.train],
                ["mean loss differential; and the 5% mark on the bars", c.fg],
            ]);
            svg.setAttribute("aria-label", `Loss differential of ${T} forecasts at horizon ${state.h}, with its mean and two test statistics.`);
            caption.textContent =
                `This sample: mean ${res.mean.toFixed(3)}. Ignoring serial correlation: standard error ${res.seNaive.toFixed(3)}, z = ${res.zNaive.toFixed(2)}, p = ${pTwoSided(res.zNaive).toFixed(3)}. ` +
                `Diebold-Mariano with ${state.h - 1} autocovariance${state.h === 2 ? "" : "s"}: standard error ${res.seDM.toFixed(3)}, DM = ${Number.isFinite(res.zDM) ? res.zDM.toFixed(2) : "∞"}, p = ${pTwoSided(res.zDM).toFixed(3)}.`;
            note.textContent = state.mu === 0
                ? "The true mean difference is 0, so every rejection is a false alarm, and a test at the 5% level should reject about 5% of the time."
                : "The true mean difference is not 0, so a rejection is a correct detection here; set it to 0 to see the false alarms.";
        }
        fresh();
        return lifecycle(el, api, render, stop);
    });

    // ==========================================================================
    // signed-rank
    // ==========================================================================
    dlWidgets.register("signed-rank", (el, api) => {
        // AUC of C4.5 and of C4.5 with m tuned, Demšar (2006), Table 2.
        const DATA = [
            ["adult (sample)", 0.763, 0.768], ["breast cancer", 0.599, 0.591], ["breast cancer wisconsin", 0.954, 0.971],
            ["cmc", 0.628, 0.661], ["ionosphere", 0.882, 0.888], ["iris", 0.936, 0.931], ["liver disorders", 0.661, 0.668],
            ["lung cancer", 0.583, 0.583], ["lymphography", 0.775, 0.838], ["mushroom", 1.0, 1.0], ["primary tumor", 0.94, 0.962],
            ["rheum", 0.619, 0.666], ["voting", 0.972, 0.981], ["wine", 0.957, 0.978],
        ];
        const N = DATA.length;
        const CRIT = 21; // exact critical value for N = 14 at alpha = 0.05, as given by Demšar
        const state = { flip: new Array(N).fill(false), scale: 1, shown: N, playing: false };
        function compute() {
            const big = DATA.reduce((b, row, i) => (Math.abs(row[2] - row[1]) > Math.abs(DATA[b][2] - DATA[b][1]) ? i : b), 0);
            const rows = DATA.map(([name, a, b], i) => {
                let d = Math.round((b - a) * 1000) / 1000;
                if (i === big) d *= state.scale;
                if (state.flip[i]) d = -d;
                return { name, d, i };
            });
            const order = rows.slice().sort((p, q) => Math.abs(p.d) - Math.abs(q.d));
            for (let s = 0; s < N; ) {
                let e = s;
                while (e + 1 < N && Math.abs(Math.abs(order[e + 1].d) - Math.abs(order[s].d)) < 1e-9) e++;
                for (let j = s; j <= e; j++) order[j].rank = (s + e) / 2 + 1;
                s = e + 1;
            }
            return order;
        }
        el.innerHTML = "";
        const controls = html("div", "dl-widget__controls", el);
        slider(controls, "Largest difference ×", 1, 10, 0.5, state.scale, 1, (v) => { state.scale = v; stop(); state.shown = N; render(); });
        const play = button(controls, "Play", () => (state.playing ? stop() : start()));
        button(controls, "Reset", () => { stop(); state.flip.fill(false); state.shown = N; render(); });
        const svg = node("svg", { role: "img" }, el);
        const legendBox = html("div", "dl-widget__legend", el);
        const caption = html("p", "dl-widget__caption", el);
        caption.setAttribute("aria-live", "polite");

        let timer = 0;
        function start() {
            state.playing = true;
            play.textContent = "Pause";
            state.shown = 0;
            const tick = () => {
                state.shown++;
                render();
                if (state.shown >= N) return stop();
                timer = setTimeout(tick, 650);
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
            const order = compute();
            const rowH = 20;
            const nameW = W < 480 ? 108 : 170;
            const rankW = 40;
            const H = N * rowH + 30;
            svg.setAttribute("viewBox", `0 0 ${W} ${H}`);
            svg.innerHTML = "";
            const x0 = nameW + (W - nameW - rankW) * 0.3;
            const maxAbs = Math.max(...order.map((o) => Math.abs(o.d)));
            const unit = Math.min((W - rankW - x0 - 4) / maxAbs, (x0 - nameW) / Math.max(0.0001, ...order.filter((o) => o.d < 0).map((o) => -o.d)));
            let rPlus = 0, rMinus = 0, zeros = 0;
            order.forEach((o, k) => {
                const y = 18 + k * rowH;
                const on = k < state.shown;
                const col = o.d > 0 ? c.train : o.d < 0 ? c.accent : c.fgLighter;
                const g = node("g", { style: "cursor:pointer", opacity: on ? 1 : 0.25 }, svg);
                node("rect", { x: 0, y, width: W, height: rowH, fill: "transparent" }, g);
                node("text", { x: nameW - 6, y: y + 13, "text-anchor": "end", "font-size": 10, fill: c.fg }, g, o.name);
                node("rect", { x: o.d >= 0 ? x0 : x0 + o.d * unit, y: y + 3, width: Math.max(1.5, Math.abs(o.d) * unit), height: rowH - 7, fill: col, "fill-opacity": 0.75 }, g);
                node("text", { x: o.d >= 0 ? x0 - 4 : x0 + 4, y: y + 13, "text-anchor": o.d >= 0 ? "end" : "start", "font-size": 9, fill: c.fgLight }, g, (o.d > 0 ? "+" : "") + o.d.toFixed(3));
                node("text", { x: W - 4, y: y + 13, "text-anchor": "end", "font-size": 10, "font-weight": 700, fill: col }, g, on ? o.rank : "");
                g.addEventListener("click", () => { stop(); state.shown = N; state.flip[o.i] = !state.flip[o.i]; render(); });
                if (on) {
                    if (o.d > 0) rPlus += o.rank;
                    else if (o.d < 0) rMinus += o.rank;
                    else zeros += o.rank;
                }
            });
            node("line", { x1: x0, x2: x0, y1: 14, y2: 18 + N * rowH, stroke: c.fg }, svg);
            node("text", { x: W - 4, y: 10, "text-anchor": "end", "font-size": 10, fill: c.fgLight }, svg, "rank");
            node("text", { x: x0 + 4, y: 10, "font-size": 10, fill: c.fgLight }, svg, "second method better →");
            // Ranks of zero differences are split evenly between the two sums.
            rPlus += zeros / 2;
            rMinus += zeros / 2;
            const T = Math.min(rPlus, rMinus);
            const z = (T - (N * (N + 1)) / 4) / Math.sqrt((N * (N + 1) * (2 * N + 1)) / 24);
            const ds = order.map((o) => o.d);
            const mean = ds.reduce((s, v) => s + v, 0) / N;
            const sd = Math.sqrt(ds.reduce((s, v) => s + (v - mean) * (v - mean), 0) / (N - 1));
            const t = mean / (sd / Math.sqrt(N));
            legend(legendBox, [
                ["second method better: rank goes to R+", c.train],
                ["first method better: rank goes to R−", c.accent],
                ["no difference: rank split between the two", c.fgLighter],
            ]);
            svg.setAttribute("aria-label", "Differences between two classifiers on 14 data sets, ordered by absolute size, with their ranks.");
            caption.textContent = state.shown < N
                ? `Ranking the differences by absolute size: ${state.shown} of ${N} ranked. R+ = ${rPlus}, R− = ${rMinus}.`
                : `R+ = ${rPlus}, R− = ${rMinus}, so T = ${T}. For 14 data sets the difference is significant at the 5% level when T is ${CRIT} or less: ` +
                  `${T <= CRIT ? "significant" : "not significant"} (normal approximation z = ${z.toFixed(2)}). ` +
                  `Paired t statistic: ${t.toFixed(2)} (mean difference ${mean.toFixed(4)}, standard deviation ${sd.toFixed(4)}). ` +
                  (state.scale > 1 ? "Enlarging the largest difference leaves every rank, and so the Wilcoxon result, unchanged; the t statistic moves. " : "") +
                  "Click a row to flip the sign of its difference.";
        }
        render();
        return lifecycle(el, api, render, stop);
    });

    // ==========================================================================
    // multiple-tests
    // ==========================================================================
    dlWidgets.register("multiple-tests", (el, api) => {
        const ALPHA = 0.05;
        const state = { m: 10, real: 0, method: 0, n: 0, any: [0, 0, 0], found: [0, 0, 0], playing: false };
        const METHODS = ["No correction", "Bonferroni", "Holm"];
        let r = rng(77);
        let ps = [];
        /** Which hypotheses each method rejects, for one set of p-values. */
        function reject(p) {
            const m = p.length;
            const none = p.map((v) => v < ALPHA);
            const bonf = p.map((v) => v < ALPHA / m);
            const holm = new Array(m).fill(false);
            const idx = p.map((v, i) => i).sort((a, b) => p[a] - p[b]);
            for (let k = 0; k < m; k++) {
                if (p[idx[k]] < ALPHA / (m - k)) holm[idx[k]] = true;
                else break;
            }
            return [none, bonf, holm];
        }
        function experiment() {
            // The first `real` comparisons have a true difference (z around 3); the rest have none.
            ps = Array.from({ length: state.m }, (_, i) => pTwoSided(r.normal() + (i < state.real ? 3 : 0)));
            const rej = reject(ps);
            state.n++;
            rej.forEach((list, k) => {
                state.any[k] += list.some((v, i) => v && i >= state.real);
                state.found[k] += list.filter((v, i) => v && i < state.real).length;
            });
        }
        const reset = () => {
            stop();
            state.real = Math.min(state.real, state.m);
            realS.set(state.real);
            state.n = 0;
            state.any = [0, 0, 0];
            state.found = [0, 0, 0];
            experiment();
            render();
        };
        el.innerHTML = "";
        const controls = html("div", "dl-widget__controls", el);
        slider(controls, "Comparisons m", 1, 40, 1, state.m, 0, (v) => { state.m = v; reset(); });
        const realS = slider(controls, "With a real difference", 0, 10, 1, state.real, 0, (v) => { state.real = v; reset(); });
        const method = seg(controls, "Correction", METHODS, (i) => { state.method = i; render(); });
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
                const batch = state.n < 10 ? 1 : 5;
                for (let i = 0; i < batch; i++) experiment();
                render();
                if (state.n >= 1000) return stop();
                timer = setTimeout(tick, state.n < 10 ? 450 : 50);
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
            method.select(state.method);
            const m = state.m;
            const rej = reject(ps)[state.method];
            const perRow = W < 480 ? 8 : 10;
            const cell = Math.min(58, (W - 8) / perRow);
            const rows = Math.ceil(m / perRow);
            const gridH = rows * (cell * 0.62 + 4);
            const barH = 14;
            const H = gridH + 34 + 3 * (barH + 18);
            svg.setAttribute("viewBox", `0 0 ${W} ${H}`);
            svg.innerHTML = "";
            ps.forEach((p, i) => {
                const cx = 4 + (i % perRow) * cell;
                const cy = 2 + Math.floor(i / perRow) * (cell * 0.62 + 4);
                const real = i < state.real;
                const fill = rej[i] ? (real ? c.train : c.accent) : "none";
                node("rect", { x: cx, y: cy, width: cell - 4, height: cell * 0.62, rx: 3, fill, "fill-opacity": 0.8, stroke: real ? c.train : c.fgLighter, "stroke-width": real ? 2 : 1 }, svg);
                node("text", { x: cx + (cell - 4) / 2, y: cy + cell * 0.31 + 3.5, "text-anchor": "middle", "font-size": cell < 40 ? 8.5 : 10, fill: rej[i] ? c.bg : c.fg }, svg, p < 0.001 ? "<.001" : p.toFixed(3).slice(1));
            });
            // Share of experiments with at least one false discovery, per method.
            let y = gridH + 30;
            const full = W - 8;
            const theory = 1 - Math.pow(1 - ALPHA, m - state.real);
            METHODS.forEach((name, k) => {
                const share = state.n ? state.any[k] / state.n : 0;
                const power = state.real ? state.found[k] / (state.n * state.real) : null;
                node("text", { x: 4, y: y - 4, "font-size": 10, "font-weight": k === state.method ? 700 : 400, fill: c.fg }, svg,
                    `${name}: at least one false discovery in ${(100 * share).toFixed(1)}% of experiments` + (power === null ? "" : `; real differences found: ${(100 * power).toFixed(0)}%`));
                node("rect", { x: 4, y, width: full, height: barH, fill: c.fgLightest }, svg);
                node("rect", { x: 4, y, width: full * share, height: barH, fill: c.accent, "fill-opacity": k === state.method ? 0.85 : 0.45 }, svg);
                node("line", { x1: 4 + full * ALPHA, x2: 4 + full * ALPHA, y1: y - 2, y2: y + barH + 2, stroke: c.fg, "stroke-width": 1.5 }, svg);
                y += barH + 18;
            });
            legend(legendBox, [
                ["declared significant although there is no difference", c.accent],
                ["a real difference (outlined), filled when detected", c.train],
                ["the 5% mark on the bars", c.fg],
            ]);
            const nFalse = rej.filter((v, i) => v && i >= state.real).length;
            svg.setAttribute("aria-label", `${m} p-values from one experiment and the share of experiments with a false discovery under three procedures.`);
            caption.textContent =
                `This experiment, ${METHODS[state.method].toLowerCase()}: ${nFalse} false discover${nFalse === 1 ? "y" : "ies"} among ${m - state.real} comparison${m - state.real === 1 ? "" : "s"} with no real difference. ` +
                `Over ${state.n} experiment${state.n > 1 ? "s" : ""} the bars show how often at least one occurs. ` +
                `For ${m - state.real} independent tests at the 5% level without correction the expected share is ${(100 * theory).toFixed(0)}%.`;
        }
        experiment();
        render();
        return lifecycle(el, api, render, stop);
    });

    // ==========================================================================
    // resampled-t: correlated split estimates and the corrected resampled t-test
    // ==========================================================================
    dlWidgets.register("resampled-t", (el, api) => {
        const JS = [3, 5, 8, 10, 15, 20, 30, 50, 100];
        const REPS = 3000;
        const state = { frac: 0.2, j: 3, playing: false };
        // Two-sided 5% critical value of Student's t (Cornish-Fisher expansion around the normal quantile).
        const tcrit = (nu) => {
            const z = 1.959964;
            return z + (z ** 3 + z) / (4 * nu) + (5 * z ** 5 + 16 * z ** 3 + 3 * z) / (96 * nu * nu) + (3 * z ** 7 + 19 * z ** 5 + 17 * z ** 3 - 15 * z) / (384 * nu ** 3);
        };
        let rates = [];
        function simulate() {
            const g = rng(4711);
            const rho = state.frac; // Nadeau and Bengio's stand-in: the share of the data used for testing
            rates = JS.map((J) => {
                let naive = 0, corr = 0;
                const crit = tcrit(J - 1);
                for (let r = 0; r < REPS; r++) {
                    const z0 = g.normal();
                    let s = 0, ss = 0;
                    for (let j = 0; j < J; j++) {
                        const x = Math.sqrt(rho) * z0 + Math.sqrt(1 - rho) * g.normal();
                        s += x;
                        ss += x * x;
                    }
                    const m = s / J, v = (ss - J * m * m) / (J - 1);
                    naive += Math.abs(m / Math.sqrt(v / J)) > crit;
                    corr += Math.abs(m / Math.sqrt(v * (1 / J + rho / (1 - rho)))) > crit;
                }
                return [naive / REPS, corr / REPS];
            });
        }
        el.innerHTML = "";
        const controls = html("div", "dl-widget__controls", el);
        slider(controls, "Share of data used for testing", 0.05, 0.5, 0.05, state.frac, 2, (v) => { state.frac = v; stop(); simulate(); render(); });
        const jS = slider(controls, "Number of splits J", 0, JS.length - 1, 1, state.j, 0, (v) => { state.j = v; stop(); render(); });
        const play = button(controls, "Play", () => (state.playing ? stop() : start()));
        const svg = node("svg", { role: "img" }, el);
        const legendBox = html("div", "dl-widget__legend", el);
        const caption = html("p", "dl-widget__caption", el);
        caption.setAttribute("aria-live", "polite");
        let timer = 0;
        function start() {
            state.playing = true;
            play.textContent = "Pause";
            state.j = -1;
            const tick = () => {
                state.j += 1;
                jS.set(state.j);
                render();
                if (state.j >= JS.length - 1) return stop();
                timer = setTimeout(tick, 800);
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
            const padL = 40, padR = 12, H = 230, padB = 32;
            svg.setAttribute("viewBox", `0 0 ${W} ${H}`);
            svg.innerHTML = "";
            const x = (i) => padL + ((W - padL - padR) * i) / (JS.length - 1);
            const ymax = Math.max(0.3, ...rates.map((r) => r[0])) * 1.08;
            const y = (v) => 8 + (H - padB - 8) * (1 - v / ymax);
            for (const v of [0, 0.05, 0.2, 0.4, 0.6, 0.8].filter((q) => q <= ymax)) {
                node("line", { x1: padL, x2: W - padR, y1: y(v), y2: y(v), stroke: v === 0.05 ? c.fg : c.fgLightest, "stroke-dasharray": v === 0.05 ? "5 4" : "" }, svg);
                node("text", { x: padL - 5, y: y(v) + 3, "text-anchor": "end", "font-size": 10, fill: c.fgLight }, svg, `${Math.round(v * 100)}%`);
            }
            const upto = state.j + 1;
            for (const [k, col] of [[0, c.accent], [1, c.train]]) {
                node("polyline", { points: polyline(rates.map((r, i) => [x(i), y(r[k])])), fill: "none", stroke: col, "stroke-width": 1, "stroke-opacity": 0.35 }, svg);
                node("polyline", { points: polyline(rates.slice(0, upto).map((r, i) => [x(i), y(r[k])])), fill: "none", stroke: col, "stroke-width": 2.4 }, svg);
                node("circle", { cx: x(state.j), cy: y(rates[state.j][k]), r: 4.5, fill: col }, svg);
            }
            JS.forEach((J, i) => node("text", { x: x(i), y: H - padB + 14, "text-anchor": "middle", "font-size": 10, fill: c.fgLight }, svg, J));
            node("text", { x: (padL + W - padR) / 2, y: H - 3, "text-anchor": "middle", "font-size": 10, fill: c.fgLight }, svg, "number of random train/test splits J");
            legend(legendBox, [
                ["ordinary t-test on the J split results", c.accent],
                ["corrected resampled t-test", c.train],
                ["the 5% it should be", c.fg],
            ]);
            const J = JS[state.j], r = rates[state.j];
            svg.setAttribute("aria-label", "False alarm rates of an ordinary and a corrected t-test on the results of repeated random splits, by number of splits.");
            caption.textContent =
                `Two algorithms of equal quality, compared on ${J} random splits with ${Math.round(state.frac * 100)}% of the data held out each time. ` +
                `The ordinary t-test declares a difference in ${(100 * r[0]).toFixed(0)}% of experiments, the corrected test in ${(100 * r[1]).toFixed(0)}%; 5% is the target. ` +
                "More splits make the ordinary test worse, because they add repetitions and no new data.";
        }
        simulate();
        render();
        return lifecycle(el, api, render, stop);
    });

    // ==========================================================================
    // winners-curse: the best of many models that are all no better than the benchmark
    // ==========================================================================
    dlWidgets.register("winners-curse", (el, api) => {
        const REPS = 3000, LMAX = 200;
        const state = { l: 20, r: 0, playing: false };
        const g = rng(2000);
        const Z0 = Float64Array.from({ length: REPS }, () => g.normal());
        const Z = Array.from({ length: REPS }, () => Float64Array.from({ length: LMAX }, () => g.normal()));
        function best() {
            const a = Math.sqrt(state.r), b = Math.sqrt(1 - state.r);
            const out = new Float64Array(REPS);
            for (let i = 0; i < REPS; i++) {
                let m = -Infinity;
                for (let k = 0; k < state.l; k++) m = Math.max(m, a * Z0[i] + b * Z[i][k]);
                out[i] = m;
            }
            return out;
        }
        el.innerHTML = "";
        const controls = html("div", "dl-widget__controls", el);
        const lS = slider(controls, "Models tried", 1, LMAX, 1, state.l, 0, (v) => { state.l = v; stop(); render(); });
        slider(controls, "Correlation between models", 0, 0.95, 0.05, state.r, 2, (v) => { state.r = v; stop(); render(); });
        const play = button(controls, "Play", () => (state.playing ? stop() : start()));
        const svg = node("svg", { role: "img" }, el);
        const legendBox = html("div", "dl-widget__legend", el);
        const caption = html("p", "dl-widget__caption", el);
        caption.setAttribute("aria-live", "polite");
        let timer = 0;
        function start() {
            state.playing = true;
            play.textContent = "Pause";
            state.l = 0;
            const tick = () => {
                state.l = Math.min(LMAX, state.l + (state.l < 10 ? 1 : state.l < 50 ? 5 : 15));
                lS.set(state.l);
                render();
                if (state.l >= LMAX) return stop();
                timer = setTimeout(tick, 320);
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
            const vals = best();
            const padL = 12, padR = 12, H = 210, padB = 30;
            svg.setAttribute("viewBox", `0 0 ${W} ${H}`);
            svg.innerHTML = "";
            const LO = -3.5, HI = 5, BINS = 51;
            const counts = new Array(BINS).fill(0);
            let sig = 0, sum = 0;
            for (const v of vals) {
                counts[Math.max(0, Math.min(BINS - 1, Math.floor(((v - LO) / (HI - LO)) * BINS)))]++;
                sig += v > 1.645;
                sum += v;
            }
            const bw = (HI - LO) / BINS;
            const dens = counts.map((k) => k / (REPS * bw));
            const ymax = Math.max(0.42, ...dens) * 1.05;
            const x = (v) => padL + ((W - padL - padR) * (v - LO)) / (HI - LO);
            const y = (d) => 8 + (H - padB - 8) * (1 - d / ymax);
            dens.forEach((d, i) => {
                const v0 = LO + i * bw;
                node("rect", { x: x(v0) + 0.5, width: x(v0 + bw) - x(v0) - 1, y: y(d), height: y(0) - y(d), fill: v0 + bw / 2 > 1.645 ? c.accent : c.train, "fill-opacity": 0.6 }, svg);
            });
            const curve = [];
            for (let v = LO; v <= HI; v += 0.05) curve.push([x(v), y(phi(v))]);
            node("polyline", { points: polyline(curve), fill: "none", stroke: c.fg, "stroke-width": 1.6, "stroke-dasharray": "5 3" }, svg);
            node("line", { x1: x(1.645), x2: x(1.645), y1: 6, y2: H - padB + 4, stroke: c.fg }, svg);
            node("text", { x: x(1.645) + 4, y: 16, "font-size": 10, fill: c.fg }, svg, "5% threshold for one model");
            for (let v = -3; v <= 5; v++) node("text", { x: x(v), y: H - padB + 14, "text-anchor": "middle", "font-size": 10, fill: c.fgLight }, svg, v);
            node("text", { x: (padL + W - padR) / 2, y: H - 3, "text-anchor": "middle", "font-size": 10, fill: c.fgLight }, svg, "test statistic of the best model against the benchmark");
            legend(legendBox, [
                ["best of the models tried, over repeated searches", c.train],
                ["searches whose winner looks significant", c.accent],
                ["what one model chosen in advance would show", c.fg],
            ]);
            const mean = sum / REPS;
            svg.setAttribute("aria-label", `Distribution of the test statistic of the best of ${state.l} models that are all no better than the benchmark.`);
            caption.textContent =
                `${state.l} model${state.l > 1 ? "s" : ""}, none better than the benchmark. The best one has an average test statistic of ${mean.toFixed(2)}, and in ${((100 * sig) / REPS).toFixed(0)}% of searches it passes a one-sided 5% test meant for a single model. ` +
                (state.r > 0.6 ? "Highly correlated models are almost one model tried many times, so the inflation is small." : state.l === 1 ? "With one model chosen in advance the rate is the nominal 5%." : "The right reference for the winner is the distribution of the best of all models tried, not of one.");
        }
        render();
        return lifecycle(el, api, render, stop);
    });

    // ==========================================================================
    // type-m: what significant results look like when the power is low (after Gelman and Carlin 2014)
    // ==========================================================================
    dlWidgets.register("type-m", (el, api) => {
        const state = { d: 1, playing: false };
        el.innerHTML = "";
        const controls = html("div", "dl-widget__controls", el);
        const dS = slider(controls, "True difference, in standard errors", 0.1, 4, 0.05, state.d, 2, (v) => { state.d = v; stop(); render(); });
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
                state.d = Math.round((4 - 3.9 * u) * 20) / 20;
                dS.set(state.d);
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
            const d = state.d, Zc = 1.96;
            const padL = 12, padR = 12, H = 210, padB = 30;
            svg.setAttribute("viewBox", `0 0 ${W} ${H}`);
            svg.innerHTML = "";
            const LO = -4, HI = 8;
            const x = (v) => padL + ((W - padL - padR) * (v - LO)) / (HI - LO);
            const y = (v) => 8 + (H - padB - 8) * (1 - v / 0.42);
            const xs = range(LO, HI, 300);
            const area = (a, b, col) => {
                const seg = xs.filter((v) => v >= a && v <= b);
                if (seg.length < 2) return;
                node("polygon", { points: polyline(seg.map((v) => [x(v), y(phi(v - d))]).concat([[x(seg[seg.length - 1]), y(0)], [x(seg[0]), y(0)]])), fill: col, "fill-opacity": 0.45 }, svg);
            };
            area(Zc, HI, c.train);
            area(LO, -Zc, c.accent);
            node("polyline", { points: polyline(xs.map((v) => [x(v), y(phi(v - d))])), fill: "none", stroke: c.fg, "stroke-width": 1.8 }, svg);
            node("line", { x1: padL, x2: W - padR, y1: y(0), y2: y(0), stroke: c.fgLighter }, svg);
            for (const v of [-Zc, Zc]) node("line", { x1: x(v), x2: x(v), y1: 20, y2: y(0), stroke: c.fgLight, "stroke-dasharray": "4 3" }, svg);
            node("line", { x1: x(d), x2: x(d), y1: 8, y2: y(0) + 4, stroke: c.fg, "stroke-width": 2 }, svg);
            node("text", { x: x(d) + 4, y: 16, "font-size": 10, fill: c.fg }, svg, "true difference");
            // Probabilities and the mean absolute size of the significant estimates.
            const pPos = 1 - Phi(Zc - d), pNeg = Phi(-Zc - d);
            const power = pPos + pNeg;
            const ePos = d * pPos + phi(Zc - d), eNeg = -d * pNeg + phi(-Zc - d);
            const exaggeration = (ePos + eNeg) / power / d;
            const meanSig = (ePos + eNeg) / power;
            node("line", { x1: x(meanSig), x2: x(meanSig), y1: 30, y2: y(0) + 4, stroke: c.train, "stroke-width": 2, "stroke-dasharray": "2 2" }, svg);
            node("text", { x: x(meanSig) + 4, y: 40, "font-size": 10, fill: c.train }, svg, "average significant estimate");
            for (let v = -4; v <= 8; v += 2) node("text", { x: x(v), y: H - padB + 14, "text-anchor": "middle", "font-size": 10, fill: c.fgLight }, svg, v);
            node("text", { x: (padL + W - padR) / 2, y: H - 3, "text-anchor": "middle", "font-size": 10, fill: c.fgLight }, svg, "estimated difference, in standard errors");
            legend(legendBox, [
                ["estimates over repeated experiments", c.fg],
                ["significant, with the right sign", c.train],
                ["significant, with the wrong sign", c.accent],
            ]);
            svg.setAttribute("aria-label", `Sampling distribution of an estimated difference whose true value is ${d.toFixed(2)} standard errors, with the significant regions shaded.`);
            caption.textContent =
                `Power ${(100 * power).toFixed(0)}%: that share of experiments reaches significance. Those that do report a difference ${exaggeration.toFixed(2)} times the true one on average, and ${((100 * pNeg) / power).toFixed(1)}% of them have the wrong sign. ` +
                (power < 0.5 ? "At low power, a significant result is mostly a lucky draw, and it overstates the effect." : "With enough power, significant results are close to the truth.");
        }
        render();
        return lifecycle(el, api, render, stop);
    });
})();
