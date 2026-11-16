/*
 * Four explainers for the page "Evaluating Probabilistic Forecasts".
 *
 *   pit-explorer    forecasters against nature: the PIT histogram, the
 *                   coverage and width of central intervals, and the CRPS
 *   proper-score    the expected score of a reported N(m, s²) when the truth
 *                   is N(0, 1), for the log score, the CRPS and the linear score
 *   quantile-loss   the pinball loss of a quoted quantile on a small sample
 *   interval-score  a quoted central interval against N(0, 1) observations
 *
 * The simulation in pit-explorer follows the scenario of Gneiting, Balabdaoui
 * and Raftery (2007), Table 1: nature draws mu_t ~ N(0, 1) and the observation
 * from N(mu_t, 1). All forecasts are mixtures of normals, so every quantity is
 * computed in closed form or by bisection; nothing is fetched.
 *
 *     <div class="dl-widget" data-widget="pit-explorer" markdown>
 *     Fallback text for readers without JavaScript.
 *     </div>
 */
(function () {
    const {
        NS, SQRT2PI, SQRTPI, phi, Phi, PhiInv, absMean, rng,
        mixCdf, mixPdf, mixQuantile, mixCrps,
        node, html, slider, seg, button, legend, widthOf, lifecycle, polyline, range,
    } = window.dlWidgetKit;

    // ==========================================================================
    // pit-explorer
    // ==========================================================================
    dlWidgets.register("pit-explorer", (el, api) => {
        const N = 4000;
        const BINS = 20;
        const X0 = -6;
        const X1 = 6;
        const state = { who: 0, bias: 0, spread: 1, n: N, playing: false };
        const FORECASTERS = [
            ["Your forecaster", null, ""],
            ["Ideal", (mu) => [[1, mu, 1]], "The forecast is the distribution the observation is drawn from."],
            ["Climatological", () => [[1, 0, Math.SQRT2]], "The same N(0, 2) every time, whatever the current state is."],
            ["Unfocused", (mu, r) => [[0.5, mu, 1], [0.5, mu + (r.u() < 0.5 ? -1 : 1), 1]], "Half of the weight is on the right distribution, half on a copy shifted by +1 or −1."],
            ["Hamill's", (mu, r) => {
                const k = Math.floor(r.u() * 3);
                return [[1, mu + [0.5, -0.5, 0][k], [1, 1, 1.3][k]]];
            }, "Every single forecast is biased up, biased down or too wide, picked at random."],
        ];
        // Nature is the same for every forecaster: mu_t ~ N(0, 1), x_t ~ N(mu_t, 1).
        const nature = rng(20071);
        const MU = [];
        const X = [];
        for (let t = 0; t < N; t++) {
            MU.push(nature.normal());
            X.push(MU[t] + nature.normal());
        }
        let draws = [];
        function simulate() {
            const r = rng(977);
            const f = FORECASTERS[state.who][1] || ((mu) => [[1, mu + state.bias, state.spread]]);
            const widthCache = new Map();
            let c50 = 0, c90 = 0, w90 = 0, crps = 0;
            draws = [];
            for (let t = 0; t < N; t++) {
                const mix = f(MU[t], r);
                const pit = mixCdf(mix, X[t]);
                // The 90% width only depends on the shape of the mixture, not on where it sits.
                const key = mix.map(([w, m, s]) => `${w}:${(m - mix[0][1]).toFixed(3)}:${s}`).join("|");
                if (!widthCache.has(key)) widthCache.set(key, mixQuantile(mix, 0.95) - mixQuantile(mix, 0.05));
                c50 += pit >= 0.25 && pit <= 0.75;
                c90 += pit >= 0.05 && pit <= 0.95;
                w90 += widthCache.get(key);
                crps += mixCrps(mix, X[t]);
                draws.push({ mix, pit, c50, c90, w90, crps });
            }
        }

        el.innerHTML = "";
        const controls = html("div", "dl-widget__controls", el);
        const who = seg(controls, "Forecaster", FORECASTERS.map((f) => f[0]), (i) => {
            state.who = i;
            restart();
        });
        const custom = () => {
            state.who = 0;
            restart();
        };
        slider(controls, "Bias", -1.5, 1.5, 0.05, state.bias, 2, (v) => { state.bias = v; custom(); });
        slider(controls, "Spread", 0.4, 2.5, 0.05, state.spread, 2, (v) => { state.spread = v; custom(); });
        const play = button(controls, "Play", () => setPlaying(!state.playing));
        const svg = node("svg", { role: "img" }, el);
        const legendBox = html("div", "dl-widget__legend", el);
        const stats = html("p", "dl-widget__caption", el);
        stats.setAttribute("aria-live", "polite");
        const caption = html("p", "dl-widget__caption", el);

        let timer = 0;
        function setPlaying(on) {
            clearTimeout(timer);
            state.playing = on;
            play.textContent = on ? "Pause" : "Play";
            if (on) {
                if (state.n >= N) state.n = 0;
                tick();
            }
        }
        function tick() {
            if (!state.playing) return;
            const n = state.n;
            state.n = Math.min(N, n < 8 ? n + 1 : n + Math.ceil(n * 0.18));
            render();
            if (state.n >= N) return setPlaying(false);
            timer = setTimeout(tick, n < 8 ? 900 : n < 60 ? 220 : 90);
        }
        function restart() {
            setPlaying(false);
            state.n = N;
            simulate();
            render();
        }

        let W = 640;
        function render() {
            const c = api.colors();
            W = widthOf(el);
            const padL = 34, padR = 10;
            const topH = 118, gap = 30, botH = W < 480 ? 150 : 170, padB = 30;
            const H = topH + gap + botH + padB;
            svg.setAttribute("viewBox", `0 0 ${W} ${H}`);
            svg.innerHTML = "";
            who.select(state.who);
            const n = state.n;
            const cur = draws[Math.max(0, n - 1)];
            const t = Math.max(0, n - 1);

            // Top: the latest forecast, its observation and the PIT value.
            const x = (v) => padL + ((W - padL - padR) * (v - X0)) / (X1 - X0);
            const xs = range(X0, X1, 200);
            const dens = xs.map((v) => mixPdf(cur.mix, v));
            const dmax = Math.max(0.42, ...dens);
            const yTop = (d) => 10 + (topH - 10) * (1 - d / dmax);
            node("line", { x1: padL, x2: W - padR, y1: topH, y2: topH, stroke: c.fgLighter }, svg);
            if (n > 0) {
                const left = xs.filter((v) => v <= X[t]);
                const area = left.map((v) => [x(v), yTop(mixPdf(cur.mix, v))]);
                area.push([x(Math.min(X1, Math.max(X0, X[t]))), yTop(mixPdf(cur.mix, X[t]))], [x(Math.min(X1, Math.max(X0, X[t]))), topH], [x(X0), topH]);
                node("polygon", { points: polyline(area), fill: c.train, "fill-opacity": 0.3 }, svg);
                node("polyline", { points: polyline(xs.map((v, i) => [x(v), yTop(dens[i])])), fill: "none", stroke: c.train, "stroke-width": 2 }, svg);
                const xo = x(Math.min(X1, Math.max(X0, X[t])));
                node("line", { x1: xo, x2: xo, y1: 6, y2: topH, stroke: c.accent, "stroke-width": 2 }, svg);
                node("circle", { cx: xo, cy: topH, r: 4, fill: c.accent }, svg);
                const right = xo < W * 0.6;
                node("text", { x: xo + (right ? 7 : -7), y: 16, "text-anchor": right ? "start" : "end", "font-size": 11, fill: c.fg }, svg,
                    `forecast ${n} of ${N}: PIT = ${cur.pit.toFixed(2)}`);
            } else {
                node("text", { x: W / 2, y: topH / 2, "text-anchor": "middle", "font-size": 11, fill: c.fgLight }, svg, "no forecasts yet");
            }
            for (let v = -6; v <= 6; v += 2) node("text", { x: x(v), y: topH + 13, "text-anchor": "middle", "font-size": 10, fill: c.fgLight }, svg, v);

            // Bottom: histogram of the PIT values so far.
            const y0 = topH + gap;
            const counts = new Array(BINS).fill(0);
            for (let i = 0; i < n; i++) counts[Math.min(BINS - 1, Math.floor(draws[i].pit * BINS))]++;
            const rel = counts.map((k) => (n ? (k / n) * BINS : 0));
            const ymax = Math.min(4, Math.max(2, ...rel)) * 1.05;
            const xb = (p) => padL + (W - padL - padR) * p;
            const yb = (v) => y0 + botH * (1 - v / ymax);
            for (const v of [0, 1, 2]) {
                node("line", { x1: padL, x2: W - padR, y1: yb(v), y2: yb(v), stroke: v === 1 ? c.fg : c.fgLightest, "stroke-dasharray": v === 1 ? "5 4" : "" }, svg);
                node("text", { x: padL - 6, y: yb(v) + 3, "text-anchor": "end", "font-size": 10, fill: c.fgLight }, svg, v);
            }
            const curBin = n ? Math.min(BINS - 1, Math.floor(cur.pit * BINS)) : -1;
            rel.forEach((v, i) => {
                node("rect", {
                    x: xb(i / BINS) + 1, width: Math.max(1, xb((i + 1) / BINS) - xb(i / BINS) - 2),
                    y: yb(Math.min(v, ymax)), height: Math.max(0, yb(0) - yb(Math.min(v, ymax))),
                    fill: i === curBin && state.playing ? c.accent : c.train, "fill-opacity": i === curBin && state.playing ? 0.9 : 0.6,
                }, svg);
            });
            for (const p of [0, 0.25, 0.5, 0.75, 1]) node("text", { x: xb(p), y: yb(0) + 13, "text-anchor": "middle", "font-size": 10, fill: c.fgLight }, svg, p);
            node("text", { x: xb(0.5), y: H - 3, "text-anchor": "middle", "font-size": 10, fill: c.fgLight }, svg, "PIT value");
            node("text", { x: W - padR, y: yb(1) - 4, "text-anchor": "end", "font-size": 10, fill: c.fg }, svg, "flat = calibrated");

            legend(legendBox, [
                ["forecast density, shaded up to the observation", c.train],
                ["observation", c.accent],
                ["share of PIT values per bin (1 = uniform)", c.train],
            ]);
            svg.setAttribute("aria-label", `PIT histogram of ${n} forecasts by ${FORECASTERS[state.who][0]}.`);
            if (n) {
                stats.textContent =
                    `After ${n} forecasts: coverage of the 50% interval ${((100 * cur.c50) / n).toFixed(1)}%, of the 90% interval ${((100 * cur.c90) / n).toFixed(1)}%; ` +
                    `average width of the 90% interval ${(cur.w90 / n).toFixed(2)}; mean CRPS ${(cur.crps / n).toFixed(3)}.`;
            } else {
                stats.textContent = "Press Play to issue forecasts one at a time.";
            }
            let note = FORECASTERS[state.who][2];
            if (state.who === 0) {
                const parts = [];
                if (state.spread > 1.12) parts.push("the forecasts are too wide, so observations pile up in the middle (a hump)");
                if (state.spread < 0.9) parts.push("the forecasts are too narrow, so observations land in the tails (a U shape)");
                if (Math.abs(state.bias) > 0.12) parts.push(`the forecasts sit too ${state.bias > 0 ? "high" : "low"}, so the histogram slopes to one side`);
                note = parts.length
                    ? "Your forecaster: " + parts.join("; ") + "."
                    : "Your forecaster matches the ideal one: bias 0 and spread 1. Move the sliders to break it.";
            } else {
                note += " The histogram is close to flat, as it is for the other three; compare the interval width and the CRPS.";
            }
            caption.textContent = note;
        }

        simulate();
        render();
        return lifecycle(el, api, render, () => clearTimeout(timer));
    });

    // ==========================================================================
    // proper-score
    // ==========================================================================
    dlWidgets.register("proper-score", (el, api) => {
        const S0 = 0.2;
        const S1 = 3;
        const state = { m: 0, s: 1.6, playing: false };
        // Expected scores when the observation is N(0, 1) and the report is N(m, s²). Lower is better.
        const SCORES = [
            ["Log score", "proper", (m, s) => Math.log(s * SQRT2PI) + (1 + m * m) / (2 * s * s)],
            ["CRPS", "proper", (m, s) => absMean(m, 1 + s * s) - s / SQRTPI],
            ["Linear score", "not proper", (m, s) => -Math.exp((-m * m) / (2 * (1 + s * s))) / Math.sqrt(2 * Math.PI * (1 + s * s))],
        ];
        el.innerHTML = "";
        const controls = html("div", "dl-widget__controls", el);
        const mS = slider(controls, "Reported mean m", -1.5, 1.5, 0.05, state.m, 2, (v) => { state.m = v; stop(); render(); });
        const sS = slider(controls, "Reported spread s", S0, S1, 0.02, state.s, 2, (v) => { state.s = v; stop(); render(); });
        const play = button(controls, "Play", () => (state.playing ? stop() : start()));
        button(controls, "Honest report", () => {
            stop();
            state.m = 0;
            state.s = 1;
            mS.set(0);
            sS.set(1);
            render();
        });
        const svg = node("svg", { role: "img" }, el);
        const legendBox = html("div", "dl-widget__legend", el);
        const caption = html("p", "dl-widget__caption", el);
        caption.setAttribute("aria-live", "polite");
        const clipId = "dl-ps-" + Math.random().toString(36).slice(2, 8);

        let raf = 0;
        let t0 = 0;
        function start() {
            state.playing = true;
            play.textContent = "Pause";
            t0 = performance.now();
            const from = S1;
            const frame = (now) => {
                const u = Math.min(1, (now - t0) / 7000);
                state.s = from + (S0 - from) * u;
                sS.set(state.s);
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
            const wide = W >= 560;
            const padL = 36, padR = 10;
            const topH = 104;
            const cols = wide ? 3 : 1;
            const pw = (W - padL - padR - (cols - 1) * 44) / cols;
            const ph = wide ? 150 : 120;
            const rowH = ph + 46;
            const H = topH + 34 + (wide ? rowH : 3 * rowH);
            svg.setAttribute("viewBox", `0 0 ${W} ${H}`);
            svg.innerHTML = "";

            // Top: the true density and the reported one.
            const x = (v) => padL + ((W - padL - padR) * (v + 5)) / 10;
            const xs = range(-5, 5, 200);
            const dmax = Math.max(0.42, phi(0) / state.s);
            const cap = Math.min(dmax, 1.3);
            const yT = (d) => 8 + (topH - 8) * (1 - Math.min(d, cap) / cap);
            const truth = xs.map((v) => [x(v), yT(phi(v))]);
            node("polygon", { points: polyline(truth.concat([[x(5), topH], [x(-5), topH]])), fill: c.fgLighter, "fill-opacity": 0.35 }, svg);
            node("polyline", { points: polyline(xs.map((v) => [x(v), yT(phi((v - state.m) / state.s) / state.s)])), fill: "none", stroke: c.accent, "stroke-width": 2.2 }, svg);
            node("line", { x1: padL, x2: W - padR, y1: topH, y2: topH, stroke: c.fgLighter }, svg);
            for (let v = -4; v <= 4; v += 2) node("text", { x: x(v), y: topH + 13, "text-anchor": "middle", "font-size": 10, fill: c.fgLight }, svg, v);

            const defs = node("defs", {}, svg);
            const ss = range(S0, S1, 140);
            const out = [];
            SCORES.forEach(([name, kind, f], k) => {
                const px = padL + (wide ? k * (pw + 44) : 0);
                const py = topH + 34 + (wide ? 0 : k * rowH) + 16;
                const vals = ss.map((s) => f(state.m, s));
                const honest = f(0, 1);
                const now = f(state.m, state.s);
                let iMin = 0;
                vals.forEach((v, i) => { if (v < vals[iMin]) iMin = i; });
                const lo = Math.min(vals[iMin], honest);
                const hi = k === 0 ? lo + 2.2 : Math.max(...vals, honest);
                const pad = (hi - lo) * 0.08 || 0.1;
                const sx = (s) => px + (pw * (s - S0)) / (S1 - S0);
                const sy = (v) => py + ph * (1 - (v - (lo - pad)) / (hi - lo + 2 * pad));
                node("rect", { x: px, y: py, width: pw, height: ph }, node("clipPath", { id: `${clipId}-${k}` }, defs));
                node("rect", { x: px, y: py, width: pw, height: ph, fill: "none", stroke: c.fgLightest }, svg);
                node("text", { x: px, y: py - 5, "font-size": 11, "font-weight": 700, fill: c.fg }, svg, name);
                node("text", { x: px + pw, y: py - 5, "text-anchor": "end", "font-size": 10, fill: kind === "proper" ? c.fgLight : c.accent }, svg, kind);
                // Where the truth's own spread is.
                node("line", { x1: sx(1), x2: sx(1), y1: py, y2: py + ph, stroke: c.fgLighter, "stroke-dasharray": "3 3" }, svg);
                const g = node("g", { "clip-path": `url(#${clipId}-${k})` }, svg);
                node("polyline", { points: polyline(ss.map((s, i) => [sx(s), sy(vals[i])])), fill: "none", stroke: c.train, "stroke-width": 2.2 }, g);
                // Lowest point of the curve, and the current report.
                node("circle", { cx: sx(ss[iMin]), cy: sy(vals[iMin]), r: 5, fill: "none", stroke: c.fg, "stroke-width": 1.5 }, svg);
                node("circle", { cx: sx(state.s), cy: Math.max(py, Math.min(py + ph, sy(now))), r: 4, fill: c.accent }, svg);
                for (const s of [0.5, 1, 2, 3]) node("text", { x: sx(s), y: py + ph + 12, "text-anchor": "middle", "font-size": 10, fill: c.fgLight }, svg, s);
                node("text", { x: px + pw / 2, y: py + ph + 25, "text-anchor": "middle", "font-size": 10, fill: c.fgLight }, svg, "reported spread s");
                node("text", { x: px - 4, y: sy(vals[iMin]) + 3, "text-anchor": "end", "font-size": 9, fill: c.fgLight }, svg, vals[iMin].toFixed(2));
                out.push({ name, now, honest, sMin: ss[iMin], edge: iMin === 0 });
            });

            legend(legendBox, [
                ["true distribution N(0, 1)", c.fgLighter],
                ["your report N(m, s²), and its expected score", c.accent],
                ["expected score over s (lower is better)", c.train],
                ["lowest point (open circle)", c.fg],
            ]);
            svg.setAttribute("aria-label", `Expected scores of the report N(${state.m.toFixed(2)}, ${state.s.toFixed(2)} squared) when the truth is N(0, 1).`);
            const [lg, cr, li] = out;
            caption.textContent =
                `Expected score of your report against the honest report N(0, 1): log score ${lg.now.toFixed(3)} vs ${lg.honest.toFixed(3)}, ` +
                `CRPS ${cr.now.toFixed(3)} vs ${cr.honest.toFixed(3)}, linear score ${li.now.toFixed(3)} vs ${li.honest.toFixed(3)}. ` +
                (li.now < li.honest - 1e-4
                    ? "The linear score rates your report better than the truth: it can be gamed. "
                    : "") +
                (Math.abs(state.m) < 0.03
                    ? `With the right mean, the log score and the CRPS are lowest at s = 1, the true spread; the linear score ${li.edge ? "keeps falling as s shrinks" : "is lowest at s = " + li.sMin.toFixed(2)}.`
                    : `With a wrong mean, the best spread to report is wider than 1: ${lg.sMin.toFixed(2)} for the log score and ${cr.sMin.toFixed(2)} for the CRPS. Neither reaches the honest report's score.`);
        }
        render();
        return lifecycle(el, api, render, stop);
    });

    // ==========================================================================
    // quantile-loss
    // ==========================================================================
    dlWidgets.register("quantile-loss", (el, api) => {
        const DATA = [1.2, 1.9, 2.3, 2.6, 3.0, 3.1, 3.6, 4.0, 4.4, 4.9, 5.5, 6.3, 7.4, 8.8, 11.0];
        const X0 = 0;
        const X1 = 12;
        const state = { alpha: 0.9, r: 4, playing: false };
        const loss = (a, r, x) => (x > r ? a * (x - r) : (1 - a) * (r - x));
        const mean = (a, r) => DATA.reduce((s, x) => s + loss(a, r, x), 0) / DATA.length;
        function best(a) {
            let b = DATA[0];
            for (const x of DATA) if (mean(a, x) < mean(a, b) - 1e-12) b = x;
            return b;
        }
        el.innerHTML = "";
        const controls = html("div", "dl-widget__controls", el);
        slider(controls, "Level α", 0.05, 0.95, 0.05, state.alpha, 2, (v) => { state.alpha = v; stop(); render(); });
        const rS = slider(controls, "Quoted quantile r", X0, X1, 0.05, state.r, 2, (v) => { state.r = v; stop(); render(); });
        const play = button(controls, "Let it settle", () => (state.playing ? stop() : start()));
        const svg = node("svg", { role: "img" }, el);
        const legendBox = html("div", "dl-widget__legend", el);
        const caption = html("p", "dl-widget__caption", el);
        caption.setAttribute("aria-live", "polite");

        let raf = 0;
        function start() {
            state.playing = true;
            play.textContent = "Pause";
            // Each observation above r pulls it up with weight α, each one below pulls it
            // down with 1 - α; r comes to rest where the pulls balance.
            const from = state.r;
            const target = best(state.alpha);
            const t0 = performance.now();
            const duration = 600 + 350 * Math.abs(target - from);
            const frame = (now) => {
                const u = Math.min(1, (now - t0) / duration);
                state.r = from + (target - from) * (1 - (1 - u) * (1 - u));
                rS.set(state.r);
                render();
                if (u < 1 && state.playing) raf = requestAnimationFrame(frame);
                else stop();
            };
            raf = requestAnimationFrame(frame);
        }
        function stop() {
            cancelAnimationFrame(raf);
            state.playing = false;
            play.textContent = "Let it settle";
        }

        let W = 640;
        function render() {
            const c = api.colors();
            W = widthOf(el);
            const padL = 36, padR = 10;
            const topH = W < 480 ? 150 : 170, gap = 34, botH = 120, padB = 30;
            const H = topH + gap + botH + padB;
            svg.setAttribute("viewBox", `0 0 ${W} ${H}`);
            svg.innerHTML = "";
            const a = state.alpha;
            const r = state.r;
            const x = (v) => padL + ((W - padL - padR) * (v - X0)) / (X1 - X0);

            // Top: the loss as a function of the observation, for the quoted r.
            const lmax = 6;
            const yT = (v) => 8 + (topH - 8) * (1 - Math.min(v, lmax) / lmax);
            for (const v of [0, 2, 4, 6]) {
                node("line", { x1: padL, x2: W - padR, y1: yT(v), y2: yT(v), stroke: c.fgLightest }, svg);
                node("text", { x: padL - 6, y: yT(v) + 3, "text-anchor": "end", "font-size": 10, fill: c.fgLight }, svg, v);
            }
            const xl = Math.max(X0, r - lmax / (1 - a));
            const xr = Math.min(X1, r + lmax / a);
            node("polyline", { points: polyline([[x(xl), yT(loss(a, r, xl))], [x(r), yT(0)], [x(xr), yT(loss(a, r, xr))]]), fill: "none", stroke: c.fg, "stroke-width": 1.6 }, svg);
            node("line", { x1: x(r), x2: x(r), y1: 8, y2: topH + 6, stroke: c.fg, "stroke-dasharray": "4 3" }, svg);
            node("text", { x: x(r) + (r > 9 ? -5 : 5), y: 16, "text-anchor": r > 9 ? "end" : "start", "font-size": 11, fill: c.fg }, svg, "r");
            for (const d of DATA) {
                const col = d > r ? c.accent : c.train;
                const l = loss(a, r, d);
                node("line", { x1: x(d), x2: x(d), y1: yT(0), y2: yT(l), stroke: col, "stroke-width": 2, "stroke-opacity": 0.75 }, svg);
                node("circle", { cx: x(d), cy: yT(0), r: 3.5, fill: col }, svg);
            }
            for (let v = 0; v <= 12; v += 2) node("text", { x: x(v), y: topH + 15, "text-anchor": "middle", "font-size": 10, fill: c.fgLight }, svg, v);

            // Bottom: the average loss over every r that could be quoted.
            const y0 = topH + gap;
            const rs = range(X0, X1, 240);
            const ms = rs.map((v) => mean(a, v));
            const b = best(a);
            const mmax = Math.max(...ms) * 1.05;
            const yB = (v) => y0 + botH * (1 - v / mmax);
            node("line", { x1: padL, x2: W - padR, y1: yB(0), y2: yB(0), stroke: c.fgLighter }, svg);
            node("line", { x1: x(b), x2: x(b), y1: y0, y2: yB(0), stroke: c.fgLighter, "stroke-dasharray": "3 3" }, svg);
            node("polyline", { points: polyline(rs.map((v, i) => [x(v), yB(ms[i])])), fill: "none", stroke: c.train, "stroke-width": 2.2 }, svg);
            node("circle", { cx: x(b), cy: yB(mean(a, b)), r: 5, fill: "none", stroke: c.fg, "stroke-width": 1.5 }, svg);
            node("circle", { cx: x(r), cy: yB(mean(a, r)), r: 4, fill: c.accent }, svg);
            node("text", { x: padL - 6, y: yB(0) + 3, "text-anchor": "end", "font-size": 10, fill: c.fgLight }, svg, 0);
            node("text", { x: padL - 6, y: y0 + 8, "text-anchor": "end", "font-size": 10, fill: c.fgLight }, svg, mmax.toFixed(1));
            for (let v = 0; v <= 12; v += 2) node("text", { x: x(v), y: yB(0) + 13, "text-anchor": "middle", "font-size": 10, fill: c.fgLight }, svg, v);
            node("text", { x: (padL + W - padR) / 2, y: H - 3, "text-anchor": "middle", "font-size": 10, fill: c.fgLight }, svg, "quoted quantile r");

            legend(legendBox, [
                ["loss for an observation, given r (top)", c.fg],
                ["observations at or below r: slope 1 − α", c.train],
                ["observations above r: slope α", c.accent],
                ["average loss over r (bottom); open circle = lowest", c.train],
            ]);
            const below = DATA.filter((d) => d <= r).length;
            svg.setAttribute("aria-label", `Pinball loss at level ${a.toFixed(2)} for the quoted quantile ${r.toFixed(2)} on 15 observations.`);
            caption.textContent =
                `Average loss ${mean(a, r).toFixed(3)}; the lowest possible is ${mean(a, b).toFixed(3)}, at r = ${b.toFixed(1)}. ` +
                `${below} of ${DATA.length} observations (${((100 * below) / DATA.length).toFixed(0)}%) are at or below r; the level asks for ${(100 * a).toFixed(0)}%. ` +
                (Math.abs(below / DATA.length - a) < 0.5 / DATA.length + 1e-9 || r === b
                    ? "The pulls from the two sides balance, which is what makes r the sample's α-quantile."
                    : below / DATA.length < a
                    ? "Too few are below, so moving r up lowers the loss."
                    : "Too many are below, so moving r down lowers the loss.");
        }
        render();
        return lifecycle(el, api, render, stop);
    });

    // ==========================================================================
    // interval-score
    // ==========================================================================
    dlWidgets.register("interval-score", (el, api) => {
        const LEVELS = [0.5, 0.8, 0.9, 0.95];
        const N = 200;
        const SHOWN = 60;
        const state = { level: 2, l: -1, u: 1, n: SHOWN, playing: false };
        const r = rng(431);
        const OBS = Array.from({ length: N }, () => r.normal());
        const JIT = Array.from({ length: N }, () => r.u());
        const alpha = () => 1 - LEVELS[state.level];
        const z = () => PhiInv(1 - alpha() / 2);
        const score = (a, l, u, x) => u - l + (2 / a) * Math.max(0, l - x) + (2 / a) * Math.max(0, x - u);
        // Expected parts of the score when the observation is N(0, 1).
        const parts = (a, l, u) => [u - l, (2 / a) * (l * Phi(l) + phi(l)), (2 / a) * (phi(u) - u * (1 - Phi(u)))];

        el.innerHTML = "";
        const controls = html("div", "dl-widget__controls", el);
        const level = seg(controls, "Nominal coverage", LEVELS.map((v) => `${100 * v}%`), (i) => { state.level = i; render(); });
        const lS = slider(controls, "Lower l", -4, 4, 0.05, state.l, 2, (v) => { state.l = Math.min(v, state.u - 0.1); lS.set(state.l); render(); });
        const uS = slider(controls, "Upper u", -4, 4, 0.05, state.u, 2, (v) => { state.u = Math.max(v, state.l + 0.1); uS.set(state.u); render(); });
        const presets = html("div", "dl-widget__seg", controls);
        presets.setAttribute("role", "group");
        presets.setAttribute("aria-label", "Examples");
        presets.style.flexWrap = "wrap";
        for (const [name, f] of [
            ["True quantiles", () => [-z(), z()]],
            ["Too narrow", () => [-0.5 * z(), 0.5 * z()]],
            ["Too wide", () => [-Math.min(4, 1.7 * z()), Math.min(4, 1.7 * z())]],
            ["One-sided", () => [PhiInv(alpha()), 4]],
        ]) {
            const b = html("button", "", presets, name);
            b.type = "button";
            b.addEventListener("click", () => {
                [state.l, state.u] = f().map((v) => Math.round(v * 100) / 100);
                lS.set(state.l);
                uS.set(state.u);
                render();
            });
        }
        const play = button(controls, "Play", () => (state.playing ? stop() : start()));
        const svg = node("svg", { role: "img" }, el);
        const legendBox = html("div", "dl-widget__legend", el);
        const caption = html("p", "dl-widget__caption", el);
        caption.setAttribute("aria-live", "polite");

        let timer = 0;
        function start() {
            state.playing = true;
            play.textContent = "Pause";
            if (state.n >= N || state.n === SHOWN) state.n = 0;
            const tick = () => {
                state.n += 1;
                render();
                if (state.n >= N) return stop();
                timer = setTimeout(tick, state.n < 10 ? 450 : 60);
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
            level.select(state.level);
            const a = alpha();
            const { l, u, n } = state;
            const padL = 12, padR = 12;
            const topH = 120, stripH = 64, gap = 30, barH = 16, padB = 8;
            const H = topH + stripH + gap + 2 * (barH + 18) + padB;
            svg.setAttribute("viewBox", `0 0 ${W} ${H}`);
            svg.innerHTML = "";
            const x = (v) => padL + ((W - padL - padR) * (Math.max(-4.4, Math.min(4.4, v)) + 4.4)) / 8.8;
            const yT = (d) => 8 + (topH - 8) * (1 - d / 0.42);

            // Top: the true density, with the quoted interval as a band.
            node("rect", { x: x(l), y: 4, width: x(u) - x(l), height: topH + stripH - 4, fill: c.train, "fill-opacity": 0.14 }, svg);
            const xs = range(-4.4, 4.4, 180);
            node("polyline", { points: polyline(xs.map((v) => [x(v), yT(phi(v))])), fill: "none", stroke: c.fgLight, "stroke-width": 1.6 }, svg);
            for (const [v, name] of [[l, "l"], [u, "u"]]) {
                node("line", { x1: x(v), x2: x(v), y1: 4, y2: topH + stripH, stroke: c.train, "stroke-width": 2 }, svg);
                node("text", { x: x(v), y: topH + stripH + 12, "text-anchor": "middle", "font-size": 11, fill: c.fg }, svg, name);
            }
            node("line", { x1: padL, x2: W - padR, y1: topH, y2: topH, stroke: c.fgLighter }, svg);
            // Strip: observations; a miss is joined to the endpoint it missed.
            let hits = 0;
            let total = 0;
            for (let i = 0; i < n; i++) {
                const o = OBS[i];
                const y = topH + 8 + JIT[i] * (stripH - 16);
                const miss = o < l || o > u;
                hits += !miss;
                total += score(a, l, u, o);
                if (miss) node("line", { x1: x(o), x2: x(o < l ? l : u), y1: y, y2: y, stroke: c.accent, "stroke-width": 1.4, "stroke-opacity": 0.8 }, svg);
                node("circle", { cx: x(o), cy: y, r: i === n - 1 && state.playing ? 4.5 : 2.6, fill: miss ? c.accent : c.fgLight }, svg);
            }

            // Bars: the expected score split into width and the two penalties.
            const mine = parts(a, l, u);
            const opt = parts(a, -z(), z());
            const sum = (p) => p[0] + p[1] + p[2];
            const scale = (W - padL - padR - 4) / Math.max(sum(mine), sum(opt));
            let y = topH + stripH + gap;
            for (const [name, p] of [["Your interval", mine], ["True quantiles", opt]]) {
                node("text", { x: padL, y: y - 4, "font-size": 10, fill: c.fg }, svg, `${name}: expected score ${sum(p).toFixed(2)} = width ${p[0].toFixed(2)} + penalties ${(p[1] + p[2]).toFixed(2)}`);
                let px = padL;
                [c.train, c.accent, c.accent].forEach((col, k) => {
                    const w = p[k] * scale;
                    node("rect", { x: px, y, width: Math.max(0, w), height: barH, fill: col, "fill-opacity": k === 0 ? 0.6 : k === 1 ? 0.9 : 0.55 }, svg);
                    px += w;
                });
                y += barH + 18;
            }

            legend(legendBox, [
                ["quoted interval, and its width in the score", c.train],
                ["observations that miss, and their penalty", c.accent],
                ["observations inside", c.fgLight],
            ]);
            const cover = Phi(u) - Phi(l);
            svg.setAttribute("aria-label", `Interval from ${l.toFixed(2)} to ${u.toFixed(2)} at nominal coverage ${100 * (1 - a)} percent against standard normal observations.`);
            caption.textContent =
                `Nominal coverage ${(100 * (1 - a)).toFixed(0)}%, true coverage of your interval ${(100 * cover).toFixed(1)}%. ` +
                (n ? `In the ${n} observations shown: ${hits} inside (${((100 * hits) / n).toFixed(0)}%), mean interval score ${(total / n).toFixed(2)}. ` : "") +
                `A miss costs ${(2 / a).toFixed(0)} times its distance from the interval. ` +
                (sum(mine) - sum(opt) < 0.005
                    ? "This is the lowest expected score any interval can get."
                    : Math.abs(cover - (1 - a)) < 0.006
                    ? "The coverage is right, yet the score is worse than for the true quantiles: coverage alone does not show it."
                    : cover < 1 - a
                    ? "The interval covers too little: the penalties outweigh what the narrow width saves."
                    : "The interval covers too much: the extra width costs more than the avoided penalties.");
        }
        render();
        return lifecycle(el, api, render, stop);
    });

    // ==========================================================================
    // crps-family: the CRPS as the average pinball loss over all quantile levels
    // ==========================================================================
    dlWidgets.register("crps-family", (el, api) => {
        const state = { mu: 0.8, sigma: 1, k: 9, playing: false };
        const pin2 = (a, q, y) => 2 * (y < q ? (1 - a) * (q - y) : a * (y - q)); // twice the pinball loss
        const exact = (mu, s) => {
            const z = -mu / s;
            return s * (z * (2 * Phi(z) - 1) + 2 * phi(z) - 1 / SQRTPI);
        };
        el.innerHTML = "";
        const controls = html("div", "dl-widget__controls", el);
        slider(controls, "Forecast mean", -2, 2, 0.05, state.mu, 2, (v) => { state.mu = v; stop(); render(); });
        slider(controls, "Forecast spread", 0.2, 2.5, 0.05, state.sigma, 2, (v) => { state.sigma = v; stop(); render(); });
        const kS = slider(controls, "Quantile levels K", 1, 99, 1, state.k, 0, (v) => { state.k = v; stop(); render(); });
        const play = button(controls, "Play", () => (state.playing ? stop() : start()));
        const svg = node("svg", { role: "img" }, el);
        const legendBox = html("div", "dl-widget__legend", el);
        const caption = html("p", "dl-widget__caption", el);
        caption.setAttribute("aria-live", "polite");
        let timer = 0;
        function start() {
            state.playing = true;
            play.textContent = "Pause";
            state.k = 0;
            const tick = () => {
                state.k = Math.min(99, state.k + (state.k < 12 ? 1 : 6));
                kS.set(state.k);
                render();
                if (state.k >= 99) return stop();
                timer = setTimeout(tick, state.k < 12 ? 450 : 160);
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
            const { mu, sigma, k } = state;
            const padL = 36, padR = 10, H = 240, padB = 30;
            svg.setAttribute("viewBox", `0 0 ${W} ${H}`);
            svg.innerHTML = "";
            const q = (a) => mu + sigma * PhiInv(a);
            const as = range(0.005, 0.995, 198);
            const vals = as.map((a) => pin2(a, q(a), 0));
            const ymax = Math.max(...vals) * 1.1;
            const x = (a) => padL + (W - padL - padR) * a;
            const y = (v) => 8 + (H - padB - 8) * (1 - v / ymax);
            node("polygon", { points: polyline(as.map((a, i) => [x(a), y(vals[i])]).concat([[x(0.995), y(0)], [x(0.005), y(0)]])), fill: c.train, "fill-opacity": 0.18 }, svg);
            node("polyline", { points: polyline(as.map((a, i) => [x(a), y(vals[i])])), fill: "none", stroke: c.train, "stroke-width": 2 }, svg);
            // The K levels a quantile forecast would report, and their losses.
            let grid = 0;
            for (let j = 1; j <= k; j++) {
                const a = j / (k + 1);
                const v = pin2(a, q(a), 0);
                grid += v / k;
                node("line", { x1: x(a), x2: x(a), y1: y(0), y2: y(v), stroke: c.accent, "stroke-width": k > 40 ? 1 : 2 }, svg);
                if (k <= 30) node("circle", { cx: x(a), cy: y(v), r: 3, fill: c.accent }, svg);
            }
            node("line", { x1: padL, x2: W - padR, y1: y(0), y2: y(0), stroke: c.fgLighter }, svg);
            const full = exact(mu, sigma);
            node("line", { x1: padL, x2: W - padR, y1: y(full), y2: y(full), stroke: c.fg, "stroke-dasharray": "5 4" }, svg);
            node("text", { x: W - padR, y: y(full) - 4, "text-anchor": "end", "font-size": 10, fill: c.fg }, svg, `CRPS = ${full.toFixed(3)} (the mean height of the curve)`);
            for (const a of [0, 0.25, 0.5, 0.75, 1]) node("text", { x: x(a), y: H - padB + 14, "text-anchor": "middle", "font-size": 10, fill: c.fgLight }, svg, a);
            for (const v of [0, ymax / 2.2]) node("text", { x: padL - 5, y: y(v) + 3, "text-anchor": "end", "font-size": 10, fill: c.fgLight }, svg, v.toFixed(1));
            node("text", { x: (padL + W - padR) / 2, y: H - 3, "text-anchor": "middle", "font-size": 10, fill: c.fgLight }, svg, "quantile level α");
            legend(legendBox, [
                ["twice the pinball loss of the forecast's α-quantile, for every level", c.train],
                [`the ${k} level${k > 1 ? "s" : ""} of a quantile forecast`, c.accent],
                ["exact CRPS", c.fg],
            ]);
            svg.setAttribute("aria-label", `Pinball loss by quantile level for a normal forecast with mean ${mu.toFixed(2)} and spread ${sigma.toFixed(2)} against the observation 0.`);
            const rel = (100 * (grid - full)) / full;
            caption.textContent =
                `The observation is 0. Exact CRPS ${full.toFixed(3)}. The average over ${k} equally spaced level${k > 1 ? "s" : ""} is ${grid.toFixed(3)}, ${Math.abs(rel).toFixed(1)}% ${rel >= 0 ? "above" : "below"} it. ` +
                (k === 1 ? "With one level, the median, this is the absolute error of the median." : k <= 9 ? "Nine levels is the grid many deep forecasting papers report under the name CRPS; it leaves out what happens in the tails." : "With many levels the average converges to the CRPS.");
        }
        render();
        return lifecycle(el, api, render, stop);
    });

    // ==========================================================================
    // multi-score: what the marginal CRPS, CRPS-sum and energy score see of a wrong correlation
    // ==========================================================================
    dlWidgets.register("multi-score", (el, api) => {
        const M = 1500;
        const state = { rho: 0.6, model: -0.4, playing: false };
        const g = rng(321);
        const Z = Array.from({ length: M }, () => [g.normal(), g.normal(), g.normal(), g.normal(), g.normal(), g.normal()]);
        const pair = (z1, z2, r) => [z1, r * z1 + Math.sqrt(1 - r * r) * z2];
        /** Expected scores when the data has correlation rho and the forecast correlation m; unit variances, zero means. */
        function scores(rho, m) {
            const sumScore = Math.sqrt(2 / Math.PI) * Math.sqrt(2 + 2 * m + 2 + 2 * rho) - Math.sqrt(2 + 2 * m) / SQRTPI;
            let a = 0, b = 0;
            for (const z of Z) {
                const y = pair(z[0], z[1], rho), x1 = pair(z[2], z[3], m), x2 = pair(z[4], z[5], m);
                a += Math.hypot(x1[0] - y[0], x1[1] - y[1]);
                b += Math.hypot(x1[0] - x2[0], x1[1] - x2[1]);
            }
            return { marginal: 1 / SQRTPI, sum: sumScore, energy: a / M - b / (2 * M) };
        }
        el.innerHTML = "";
        const controls = html("div", "dl-widget__controls", el);
        slider(controls, "Correlation in the data", -0.95, 0.95, 0.05, state.rho, 2, (v) => { state.rho = v; stop(); render(); });
        const mS = slider(controls, "Correlation in the forecast", -0.95, 0.95, 0.05, state.model, 2, (v) => { state.model = v; stop(); render(); });
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
                const u = Math.min(1, (now - t0) / 7000);
                state.model = Math.round((-0.95 + 1.9 * u) * 20) / 20;
                mS.set(state.model);
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
            const wide = W >= 560;
            const side = wide ? 150 : 120;
            const padL = 8, padR = 8;
            const cols = wide ? 3 : 1;
            const pw = wide ? (W - padL - padR - side - 30 - 2 * 34) / 3 : W - padL - padR - 30, ph = 120;
            const H = wide ? side + 30 : side + 20 + 3 * (ph + 44);
            svg.setAttribute("viewBox", `0 0 ${W} ${H}`);
            svg.innerHTML = "";
            // Scatter of the data and of the forecast.
            const sx = (v) => padL + side / 2 + (v * side) / 7, sy = (v) => 10 + side / 2 - (v * side) / 7;
            node("rect", { x: padL, y: 10, width: side, height: side, fill: "none", stroke: c.fgLightest }, svg);
            for (let i = 0; i < 220; i++) {
                const z = Z[i];
                const d = pair(z[0], z[1], state.rho), f = pair(z[2], z[3], state.model);
                node("circle", { cx: sx(f[0]), cy: sy(f[1]), r: 1.8, fill: c.accent, "fill-opacity": 0.55 }, svg);
                node("circle", { cx: sx(d[0]), cy: sy(d[1]), r: 1.8, fill: c.train, "fill-opacity": 0.55 }, svg);
            }
            node("text", { x: padL + side / 2, y: side + 24, "text-anchor": "middle", "font-size": 10, fill: c.fgLight }, svg, "series 1 against series 2");
            const ms = range(-0.95, 0.95, 38);
            const table = ms.map((m) => scores(state.rho, m));
            const now = scores(state.rho, state.model), best = scores(state.rho, state.rho);
            [["Mean of the two CRPS", "marginal"], ["CRPS of the sum", "sum"], ["Energy score", "energy"]].forEach(([name, key], k) => {
                const px = wide ? padL + side + 30 + k * (pw + 34) + 26 : padL + 30;
                const py = wide ? 26 : side + 44 + k * (ph + 44);
                const vals = table.map((s) => s[key]);
                const lo = Math.min(...vals), hi = Math.max(...vals, lo + 0.02);
                const pad = (hi - lo) * 0.15 + 0.01;
                const x = (m) => px + (pw * (m + 0.95)) / 1.9;
                const y = (v) => py + ph * (1 - (v - lo + pad) / (hi - lo + 2 * pad));
                node("rect", { x: px, y: py, width: pw, height: ph, fill: "none", stroke: c.fgLightest }, svg);
                node("text", { x: px, y: py - 5, "font-size": 11, "font-weight": 700, fill: c.fg }, svg, name);
                node("line", { x1: x(state.rho), x2: x(state.rho), y1: py, y2: py + ph, stroke: c.train, "stroke-dasharray": "3 3" }, svg);
                node("polyline", { points: polyline(ms.map((m, i) => [x(m), y(vals[i])])), fill: "none", stroke: c.fg, "stroke-width": 2 }, svg);
                node("circle", { cx: x(state.model), cy: y(now[key]), r: 4.5, fill: c.accent }, svg);
                for (const m of [-0.9, 0, 0.9]) node("text", { x: x(m), y: py + ph + 12, "text-anchor": "middle", "font-size": 10, fill: c.fgLight }, svg, m);
                node("text", { x: px - 4, y: y(lo) + 3, "text-anchor": "end", "font-size": 9, fill: c.fgLight }, svg, lo.toFixed(2));
                node("text", { x: px - 4, y: y(hi) + 3, "text-anchor": "end", "font-size": 9, fill: c.fgLight }, svg, hi.toFixed(2));
                node("text", { x: px + pw / 2, y: py + ph + 25, "text-anchor": "middle", "font-size": 10, fill: c.fgLight }, svg, "correlation in the forecast");
            });
            legend(legendBox, [
                ["data, and its correlation (dashed line)", c.train],
                ["forecast, and its expected score", c.accent],
                ["expected score over all forecast correlations (lower is better)", c.fg],
            ]);
            const pct = (k) => (100 * (now[k] - best[k])) / best[k];
            svg.setAttribute("aria-label", "Two correlated series, a forecast with another correlation, and three expected scores as functions of the forecast's correlation.");
            caption.textContent =
                `Both series are forecast perfectly one at a time; only the correlation between them is ${Math.abs(state.model - state.rho) < 0.03 ? "right" : "wrong"}. ` +
                `Against the forecast with the right correlation, the mean of the two CRPS is ${pct("marginal").toFixed(0)}% worse, the CRPS of the sum ${pct("sum").toFixed(0)}% worse and the energy score ${Math.max(0, pct("energy")).toFixed(0)}% worse. ` +
                "The marginal CRPS can not see dependence at all; the two joint scores see it, but faintly compared with how wrong the forecast is.";
        }
        render();
        return lifecycle(el, api, render, stop);
    });

    // ==========================================================================
    // aci-coverage: fixed and adaptive conformal intervals under a change in volatility
    // ==========================================================================
    dlWidgets.register("aci-coverage", (el, api) => {
        const T = 600, CAL = 100, ALPHA = 0.1;
        const state = { gamma: 0.02, t: T, playing: false };
        const g = rng(909);
        // Forecast errors whose spread triples for a while, as in a turbulent period.
        const sdAt = (t) => (t >= 250 && t < 420 ? 3 : 1);
        const E = Array.from({ length: T }, (_, t) => Math.abs(sdAt(t) * g.normal()));
        function run(gamma) {
            let a = ALPHA;
            const half = [], miss = [];
            for (let t = CAL; t < T; t++) {
                // Calibration scores: the last CAL absolute errors, as known just before time t.
                const scores = E.slice(t - CAL, t).sort((u, v) => u - v);
                const level = Math.min(1, Math.max(0, 1 - a));
                const h = level >= 1 ? Infinity : scores[Math.min(CAL - 1, Math.max(0, Math.ceil(level * (CAL + 1)) - 1))];
                half.push(h);
                const err = E[t] > h ? 1 : 0;
                miss.push(err);
                a += gamma * (ALPHA - err);
            }
            return { half, miss };
        }
        el.innerHTML = "";
        const controls = html("div", "dl-widget__controls", el);
        slider(controls, "Step size γ", 0, 0.1, 0.005, state.gamma, 3, (v) => { state.gamma = v; render(); });
        const play = button(controls, "Play", () => (state.playing ? stop() : start()));
        const svg = node("svg", { role: "img" }, el);
        const legendBox = html("div", "dl-widget__legend", el);
        const caption = html("p", "dl-widget__caption", el);
        caption.setAttribute("aria-live", "polite");
        let timer = 0;
        function start() {
            state.playing = true;
            play.textContent = "Pause";
            state.t = CAL + 10;
            const tick = () => {
                state.t = Math.min(T, state.t + 6);
                render();
                if (state.t >= T) return stop();
                timer = setTimeout(tick, 50);
            };
            tick();
        }
        function stop() {
            clearTimeout(timer);
            state.playing = false;
            state.t = T;
            play.textContent = "Play";
            render();
        }
        let W = 640;
        function render() {
            const c = api.colors();
            W = widthOf(el);
            const fixed = run(0), adapt = run(state.gamma);
            const n = state.t - CAL;
            const padL = 30, padR = 8, topH = 170, gap = 34, botH = 90, padB = 26;
            const H = topH + gap + botH + padB;
            svg.setAttribute("viewBox", `0 0 ${W} ${H}`);
            svg.innerHTML = "";
            const x = (t) => padL + ((W - padL - padR) * (t - CAL)) / (T - CAL);
            const y = (v) => 6 + (topH - 6) * (1 - Math.min(v, 9) / 9);
            node("rect", { x: x(250), y: 6, width: x(420) - x(250), height: topH - 6, fill: c.fgLightest }, svg);
            node("text", { x: x(252), y: 16, "font-size": 10, fill: c.fgLight }, svg, "errors three times larger");
            for (let t = CAL; t < state.t; t++) node("circle", { cx: x(t), cy: y(E[t]), r: 1.6, fill: E[t] > adapt.half[t - CAL] ? c.accent : c.fgLight }, svg);
            for (const [res, col, w] of [[fixed, c.fg, 1.4], [adapt, c.train, 2.2]]) {
                node("polyline", { points: polyline(res.half.slice(0, n).map((h, i) => [x(CAL + i), y(h)])), fill: "none", stroke: col, "stroke-width": w }, svg);
            }
            for (const v of [0, 3, 6, 9]) node("text", { x: padL - 5, y: y(v) + 3, "text-anchor": "end", "font-size": 10, fill: c.fgLight }, svg, v);
            // Coverage over the last 50 forecasts.
            const y0 = topH + gap;
            const yb = (v) => y0 + botH * (1 - (v - 0.5) / 0.5);
            for (const v of [0.5, 0.9, 1]) {
                node("line", { x1: padL, x2: W - padR, y1: yb(v), y2: yb(v), stroke: v === 0.9 ? c.fgLight : c.fgLightest, "stroke-dasharray": v === 0.9 ? "5 4" : "" }, svg);
                node("text", { x: padL - 5, y: yb(v) + 3, "text-anchor": "end", "font-size": 10, fill: c.fgLight }, svg, `${v * 100}%`);
            }
            const rolling = (miss) => miss.slice(0, n).map((_, i) => {
                const s = miss.slice(Math.max(0, i - 49), i + 1);
                return 1 - s.reduce((p, v) => p + v, 0) / s.length;
            });
            for (const [res, col, w] of [[fixed, c.fg, 1.4], [adapt, c.train, 2.2]]) {
                node("polyline", { points: polyline(rolling(res.miss).map((v, i) => [x(CAL + i), yb(Math.max(0.5, v))])), fill: "none", stroke: col, "stroke-width": w }, svg);
            }
            node("text", { x: padL, y: y0 - 6, "font-size": 10, fill: c.fgLight }, svg, "coverage over the last 50 forecasts (target 90%)");
            node("text", { x: (padL + W - padR) / 2, y: H - 4, "text-anchor": "middle", "font-size": 10, fill: c.fgLight }, svg, "time");
            legend(legendBox, [
                ["absolute forecast errors (orange: outside the adaptive interval)", c.fgLight],
                ["half-width with a fixed level", c.fg],
                ["half-width with the adaptive level", c.train],
            ]);
            const cov = (r) => 100 * (1 - r.miss.slice(0, n).reduce((p, v) => p + v, 0) / n);
            const wid = (r) => {
                const f = r.half.slice(0, n).filter(Number.isFinite);
                return f.reduce((p, v) => p + v, 0) / f.length;
            };
            const shock = (r) => 100 * (1 - r.miss.slice(150, 200).reduce((p, v) => p + v, 0) / 50);
            svg.setAttribute("aria-label", "Prediction interval half-widths and rolling coverage for a fixed and an adaptive conformal method when the error spread triples.");
            caption.textContent =
                `Target coverage 90%. Fixed level: ${cov(fixed).toFixed(1)}% overall, mean half-width ${wid(fixed).toFixed(2)}. Adaptive with γ = ${state.gamma.toFixed(3)}: ${cov(adapt).toFixed(1)}%, mean half-width ${wid(adapt).toFixed(2)}. ` +
                (n >= 200 ? `In the first 50 forecasts after the errors grow, coverage is ${shock(fixed).toFixed(0)}% with the fixed level and ${shock(adapt).toFixed(0)}% with the adaptive one. ` : "") +
                (state.gamma === 0 ? "With γ = 0 the two methods are the same." : state.gamma > 0.06 ? "A large step reacts fast and makes the width jumpy." : "The adaptive level widens the interval after each miss and narrows it after each hit.");
        }
        render();
        return lifecycle(el, api, render, stop);
    });
})();
