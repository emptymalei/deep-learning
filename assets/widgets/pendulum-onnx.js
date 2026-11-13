/*
 * Pendulum forecaster: a small PyTorch MLP, exported to ONNX, running in the
 * reader's browser with ONNX Runtime Web (WebAssembly, one thread).
 *
 *     <div class="dl-widget" data-widget="pendulum-onnx" markdown>
 *     *Interactive demo: enable JavaScript to run the pendulum forecaster in your browser.*
 *     </div>
 *
 * The model and its settings (k, h, dt, g, L range, damping, metrics) come from
 * assets/models/pendulum-mlp.{onnx,json}, both written by
 * scripts/pendulum_onnx.py. The page simulates the pendulum with the same RK4
 * integrator, feeds the first k angles to the model and plots the forecast
 * against the truth. ONNX Runtime is fetched from jsDelivr only when the widget
 * comes into view.
 */
(function () {
    "use strict";

    const ORT_VERSION = "1.30.0";
    const ORT_DIST = "https://cdn.jsdelivr.net/npm/onnxruntime-web@" + ORT_VERSION + "/dist/";
    const ORT_SCRIPT = ORT_DIST + "ort.wasm.min.js";
    const MODEL_PATH = "assets/models/pendulum-mlp.onnx";
    const META_PATH = "assets/models/pendulum-mlp.json";
    const SVG_NS = "http://www.w3.org/2000/svg";
    const DEG = 180 / Math.PI;
    const ANIMATION_SPEED = 4; // simulated seconds per real second
    const DEBOUNCE_MS = 60;

    let uid = 0;

    /**
     * RK4 for theta'' = -(g/L) sin(theta) - 2 beta theta', starting from rest.
     * Returns every integrator step; sample i of the model's series is step
     * i * rk4_substeps. Must stay identical to simulate() in
     * scripts/pendulum_onnx.py.
     */
    function simulate(theta0, length, nSteps, meta) {
        const dt = meta.rk4_dt;
        const beta = meta.damping;
        const w2 = meta.g / length;
        const accel = (th, om) => -w2 * Math.sin(th) - 2.0 * beta * om;
        const out = new Float64Array(nSteps);
        let th = theta0;
        let om = 0;
        out[0] = th;
        for (let i = 1; i < nSteps; i++) {
            const k1t = om;
            const k1w = accel(th, om);
            const k2t = om + 0.5 * dt * k1w;
            const k2w = accel(th + 0.5 * dt * k1t, k2t);
            const k3t = om + 0.5 * dt * k2w;
            const k3w = accel(th + 0.5 * dt * k2t, k3t);
            const k4t = om + dt * k3w;
            const k4w = accel(th + dt * k3t, k4t);
            th = th + (dt / 6.0) * (k1t + 2 * k2t + 2 * k3t + k4t);
            om = om + (dt / 6.0) * (k1w + 2 * k2w + 2 * k3w + k4w);
            out[i] = th;
        }
        return out;
    }

    /** Fine trajectory (every RK4 step) plus the model's samples. */
    function trajectory(theta0, length, nSamples, meta) {
        const sub = meta.rk4_substeps;
        const fine = simulate(theta0, length, (nSamples - 1) * sub + 1, meta);
        const samples = new Float64Array(nSamples);
        for (let i = 0; i < nSamples; i++) samples[i] = fine[i * sub];
        return { fine, samples };
    }

    /** Single shot (rounds = 1) or autoregressive forecast, in radians. */
    async function forecast(ort, session, context, k, h, rounds) {
        let ctx = Float32Array.from(context);
        const out = new Float32Array(h * rounds);
        for (let r = 0; r < rounds; r++) {
            const feeds = { context: new ort.Tensor("float32", ctx, [1, k]) };
            const result = await session.run(feeds);
            const f = result.forecast.data;
            out.set(f, r * h);
            const joined = new Float32Array(k + h);
            joined.set(ctx, 0);
            joined.set(f, k);
            ctx = joined.slice(h); // the last k values: forecasts fed back in
        }
        return out;
    }

    function rmse(a, b) {
        let s = 0;
        for (let i = 0; i < a.length; i++) s += (a[i] - b[i]) ** 2;
        return Math.sqrt(s / a.length);
    }

    /** Catmull-Rom interpolation of an evenly sampled series at index u. */
    function sampleAt(arr, u) {
        const n = arr.length;
        if (u <= 0) return arr[0];
        if (u >= n - 1) return arr[n - 1];
        const i = Math.floor(u);
        const f = u - i;
        const p0 = arr[Math.max(i - 1, 0)];
        const p1 = arr[i];
        const p2 = arr[i + 1];
        const p3 = arr[Math.min(i + 2, n - 1)];
        return (
            0.5 *
            (2 * p1 +
                (-p0 + p2) * f +
                (2 * p0 - 5 * p1 + 4 * p2 - p3) * f * f +
                (-p0 + 3 * p1 - 3 * p2 + p3) * f * f * f)
        );
    }

    function fmtDeg(x) {
        if (!isFinite(x)) return "n/a";
        if (x < 1) return x.toFixed(2) + "°";
        if (x < 10) return x.toFixed(1) + "°";
        return x.toFixed(0) + "°";
    }

    function svg(tag, attrs, parent) {
        const node = document.createElementNS(SVG_NS, tag);
        for (const key in attrs) node.setAttribute(key, attrs[key]);
        if (parent) parent.appendChild(node);
        return node;
    }

    function html(tag, props, parent) {
        const node = document.createElement(tag);
        Object.assign(node, props || {});
        if (parent) parent.appendChild(node);
        return node;
    }

    function niceStep(range, maxTicks, steps) {
        for (const s of steps) if (range / s <= maxTicks) return s;
        return steps[steps.length - 1];
    }

    async function fetchOk(url, what) {
        const res = await fetch(url);
        if (!res.ok) throw new Error(what + " not found (HTTP " + res.status + ")");
        return res;
    }

    /** ONNX Runtime Web, loaded once per page. */
    let ortPromise = null;
    function loadOrt(api) {
        if (!ortPromise) {
            ortPromise = api.loadScript(ORT_SCRIPT).then(() => {
                const ort = window.ort;
                if (!ort || !ort.InferenceSession) {
                    throw new Error("ONNX Runtime Web did not initialise");
                }
                ort.env.wasm.wasmPaths = ORT_DIST;
                ort.env.wasm.numThreads = 1; // no cross-origin isolation needed
                ort.env.wasm.proxy = false;
                return ort;
            });
        }
        return ortPromise;
    }

    dlWidgets.register("pendulum-onnx", (el, api) => {
        const id = "dl-pendulum-" + ++uid;
        const reduced = api.reducedMotion();
        let colors = api.colors();
        let meta = null;
        let ort = null;
        let session = null;
        let loadFailed = false;
        let disposed = false;
        const cleanups = [];

        const state = { theta0Deg: 60, length: 100, mode: "single" };
        let current = null; // last computed { samples, fine, pred, rounds, ms }

        // Animation state (see below).
        let rafId = 0;
        let animT = 0; // simulated seconds
        let lastFrame = 0;
        let xScale = null; // time (s) -> chart x, set by draw()

        // DOM nodes, created by build() once the metadata has loaded.
        let controls, chartBox, chartSvg, bobSvg, legend, readout, segButtons;
        let cursorLine = null;
        let bobParts = null;

        // Debounced, serialised recomputation (see refresh()).
        let timer = null;
        let running = false;
        let again = false;

        el.replaceChildren();
        const status = html("p", { className: "dl-widget__caption", textContent: "Loading model…" }, el);
        status.setAttribute("role", "status");
        status.style.margin = "0 0 0.4rem";

        // ------------------------------------------------------------------
        // Lifecycle: start loading when the widget is near the viewport.
        // ------------------------------------------------------------------
        let visible = true;
        let started = false;
        let io = null;
        if ("IntersectionObserver" in window) {
            visible = false;
            io = new IntersectionObserver(
                (entries) => {
                    visible = entries.some((e) => e.isIntersecting);
                    if (visible && !started) start();
                    if (visible) kickAnimation();
                },
                { rootMargin: "300px 0px" }
            );
            io.observe(el);
        } else {
            start();
        }

        async function start() {
            started = true;
            try {
                meta = await (await fetchOk(api.resolve(META_PATH), "Model metadata")).json();
            } catch (err) {
                return fail(err, "meta");
            }
            if (disposed) return;
            build();
            refresh();
            try {
                const [ortLib, bytes] = await Promise.all([
                    loadOrt(api),
                    fetchOk(api.resolve(MODEL_PATH), "Model file")
                        .then((r) => r.arrayBuffer())
                        .then((b) => new Uint8Array(b)),
                ]);
                ort = ortLib;
                const s = await ort.InferenceSession.create(bytes, {
                    executionProviders: ["wasm"],
                    graphOptimizationLevel: "all",
                });
                if (!s.inputNames.includes("context") || !s.outputNames.includes("forecast")) {
                    throw new Error("unexpected model inputs/outputs: " + s.inputNames + " -> " + s.outputNames);
                }
                if (disposed) {
                    release(s);
                    return;
                }
                session = s;
                status.textContent = "";
                status.hidden = true;
                refresh();
            } catch (err) {
                fail(err, "model");
            }
        }

        function release(s) {
            Promise.resolve()
                .then(() => s.release && s.release())
                .catch(() => {});
        }

        /** kind: "meta" (nothing works), "model" (no forecast) or "run". */
        function fail(err, kind) {
            console.error("[dl-widget] pendulum-onnx", err);
            if (disposed) return;
            loadFailed = true;
            session = null;
            if (readout) readout.textContent = "";
            const reason = String((err && err.message) || err).replace(/\.$/, "");
            status.hidden = false;
            status.textContent =
                (kind === "run" ? "The model failed to run: " : "Could not load the model: ") +
                reason +
                (kind === "meta" ? "." : ". The chart still shows the simulated pendulum, without a forecast.");
            if (kind === "meta") el.classList.add("dl-widget--error");
        }

        // ------------------------------------------------------------------
        // DOM
        // ------------------------------------------------------------------
        function slider(label, min, max, value, unit, onInput) {
            const wrap = html("label", {}, controls);
            wrap.append(label + " ");
            const input = html("input", { type: "range", min, max, step: 1, value }, wrap);
            const out = html("output", { textContent: value + unit }, wrap);
            out.style.minWidth = "3.2em";
            input.addEventListener("input", () => {
                out.textContent = input.value + unit;
                onInput(Number(input.value));
            });
            return out;
        }

        function build() {
            const k = meta.k;
            const h = meta.h;
            const rounds = meta.autoregressive_rounds;
            const [tMin, tMax] = meta.theta0_range_deg;
            const [lMin, lMax] = meta.L_range;
            state.theta0Deg = Math.min(Math.max(state.theta0Deg, tMin), tMax);
            state.length = Math.min(Math.max(state.length, lMin), lMax);

            controls = html("div", { className: "dl-widget__controls" });
            el.insertBefore(controls, status);
            slider("Initial angle θ₀", tMin, tMax, state.theta0Deg, "°", (v) => {
                state.theta0Deg = v;
                schedule();
            });
            slider("Length L", lMin, lMax, state.length, " m", (v) => {
                state.length = v;
                schedule();
            });

            const segWrap = html("span", {}, controls);
            segWrap.style.display = "inline-flex";
            segWrap.style.alignItems = "center";
            segWrap.style.gap = "0.4rem";
            segWrap.append("Rollout ");
            const seg = html("span", { className: "dl-widget__seg" }, segWrap);
            seg.setAttribute("role", "group");
            seg.setAttribute("aria-label", "Rollout");
            segButtons = [
                ["single", "Single shot · " + h + " steps"],
                ["auto", "Autoregressive · " + rounds + "×" + h],
            ].map(([mode, text]) => {
                const b = html("button", { type: "button", textContent: text }, seg);
                b.dataset.mode = mode;
                b.setAttribute("aria-pressed", String(state.mode === mode));
                b.addEventListener("click", () => {
                    state.mode = mode;
                    segButtons.forEach((x) => x.setAttribute("aria-pressed", String(x.dataset.mode === mode)));
                    schedule();
                });
                return b;
            });

            const figure = html("div", {}, el);
            Object.assign(figure.style, {
                display: "flex",
                flexWrap: "wrap",
                alignItems: "center",
                justifyContent: "center",
                gap: "0.25rem 0.75rem",
            });
            chartBox = html("div", {}, figure);
            Object.assign(chartBox.style, { flex: "1 1 18rem", minWidth: "0" });
            chartSvg = svg("svg", { role: "img" }, chartBox);

            const bobBox = html("div", {}, figure);
            Object.assign(bobBox.style, { flex: "0 0 auto", width: "7rem" });
            bobSvg = svg("svg", { viewBox: "-64 -64 128 128", "aria-hidden": "true" }, bobBox);

            legend = html("div", { className: "dl-widget__legend" }, el);
            readout = html("p", { className: "dl-widget__caption" }, el);
            readout.setAttribute("aria-live", "polite");
            const m = meta.metrics || {};
            const arch = meta.architecture || {};
            html(
                "p",
                {
                    className: "dl-widget__caption",
                    textContent:
                        "The model is an MLP (" +
                        k +
                        " angles in, " +
                        h +
                        " out; hidden layers " +
                        (arch.hidden_widths || []).join(" → ") +
                        "; " +
                        (arch.n_params || 0).toLocaleString("en") +
                        " parameters) trained offline in PyTorch on simulated pendulums and exported to ONNX. " +
                        "It runs here with ONNX Runtime Web; nothing is sent to a server. " +
                        "Samples are " +
                        meta.dt.toFixed(2) +
                        " s apart. On held-out pendulums its " +
                        h +
                        "-step RMSE is " +
                        fmtDeg((m.val_rmse_rad || NaN) * DEG) +
                        ", against " +
                        fmtDeg((m.val_naive_rmse_rad || NaN) * DEG) +
                        " for repeating the last value.",
                },
                el
            );

            if ("ResizeObserver" in window) {
                let lastWidth = 0;
                const ro = new ResizeObserver(() => {
                    const w = chartBox.clientWidth;
                    if (w && Math.abs(w - lastWidth) > 1 && current) {
                        lastWidth = w;
                        draw();
                    }
                });
                ro.observe(chartBox);
                cleanups.push(() => ro.disconnect());
            }
        }

        // ------------------------------------------------------------------
        // Compute: simulate, run the model, draw. Runs are serialised.
        // ------------------------------------------------------------------
        function schedule() {
            clearTimeout(timer);
            timer = setTimeout(refresh, DEBOUNCE_MS);
        }

        async function refresh() {
            if (!meta || disposed) return;
            if (running) {
                again = true;
                return;
            }
            running = true;
            try {
                do {
                    again = false;
                    await compute();
                } while (again && !disposed);
            } catch (err) {
                fail(err, "run");
            } finally {
                running = false;
            }
        }

        async function compute() {
            const k = meta.k;
            const h = meta.h;
            const n = k + h * meta.autoregressive_rounds;
            const rounds = state.mode === "auto" ? meta.autoregressive_rounds : 1;
            const { fine, samples } = trajectory(state.theta0Deg / DEG, state.length, n, meta);
            let pred = null;
            let ms = 0;
            if (session) {
                const t0 = performance.now();
                pred = await forecast(ort, session, samples.subarray(0, k), k, h, rounds);
                ms = performance.now() - t0;
            }
            if (disposed) return;
            current = { fine, samples, pred, rounds, ms };
            draw();
            kickAnimation();
        }

        // ------------------------------------------------------------------
        // Drawing
        // ------------------------------------------------------------------
        function draw() {
            if (!current) return;
            const c = colors;
            const { fine, samples, pred, rounds, ms } = current;
            const k = meta.k;
            const h = meta.h;
            const dt = meta.dt;
            const sub = meta.rk4_substeps;
            const n = samples.length;
            const tEnd = (n - 1) * dt;

            const W = Math.max(Math.round(chartBox.clientWidth || 600), 240);
            const H = Math.round(Math.min(Math.max(W * 0.45, 200), 270));
            const m = { l: 40, r: 8, t: 16, b: 30 };
            const pw = W - m.l - m.r;
            const ph = H - m.t - m.b;

            let yMax = state.theta0Deg;
            if (pred) {
                let p = 0;
                for (const v of pred) p = Math.max(p, Math.abs(v * DEG));
                yMax = Math.max(yMax, Math.min(p, 1.5 * state.theta0Deg, 200));
            }
            yMax = Math.max(yMax * 1.12, 3);
            const x = (t) => m.l + (t / tEnd) * pw;
            const y = (deg) => m.t + ph / 2 - (deg / yMax) * (ph / 2);

            chartSvg.replaceChildren();
            chartSvg.setAttribute("viewBox", "0 0 " + W + " " + H);
            chartSvg.setAttribute("font-size", "11");
            chartSvg.setAttribute(
                "aria-label",
                "Pendulum angle over " +
                    tEnd.toFixed(0) +
                    " seconds: the simulated trajectory, the " +
                    k +
                    "-sample context window, and the model forecast."
            );
            const clipId = id + "-clip";
            const defs = svg("defs", {}, chartSvg);
            svg("rect", { x: m.l, y: m.t, width: pw, height: ph }, svg("clipPath", { id: clipId }, defs));

            // Context window.
            const tCtx = (k - 1) * dt;
            svg("rect", { x: x(0), y: m.t, width: x(tCtx) - x(0), height: ph, fill: c.train, "fill-opacity": 0.13 }, chartSvg);
            svg("text", { x: x(0) + 4, y: m.t - 4, fill: c.fgLight }, chartSvg).textContent = "context";
            svg("text", { x: x(tCtx) + 4, y: m.t - 4, fill: c.fgLight }, chartSvg).textContent =
                rounds > 1 && pw > 360 ? "forecast, fed back every " + h + " steps" : "forecast";

            // Axes and grid.
            const yStep = niceStep(yMax, 3, [5, 10, 15, 30, 45, 60, 90]);
            for (let v = -Math.floor(yMax / yStep) * yStep; v <= yMax; v += yStep) {
                svg("line", { x1: m.l, x2: m.l + pw, y1: y(v), y2: y(v), stroke: v === 0 ? c.fgLighter : c.fgLightest }, chartSvg);
                svg("text", { x: m.l - 5, y: y(v) + 4, fill: c.fgLight, "text-anchor": "end" }, chartSvg).textContent =
                    (v < 0 ? "\u2212" + -v : v) + "°";
            }
            const xStep = niceStep(tEnd, Math.max(2, Math.floor(pw / 70)), [5, 10, 15, 20, 30]);
            for (let t = 0; t <= tEnd + 1e-9; t += xStep) {
                svg("line", { x1: x(t), x2: x(t), y1: m.t + ph, y2: m.t + ph + 4, stroke: c.fgLighter }, chartSvg);
                svg("text", { x: x(t), y: m.t + ph + 16, fill: c.fgLight, "text-anchor": "middle" }, chartSvg).textContent = t;
            }
            svg("text", { x: m.l + pw, y: m.t + ph + 28, fill: c.fgLight, "text-anchor": "end" }, chartSvg).textContent = "time (s)";

            // Where forecasts are fed back in.
            if (rounds > 1) {
                for (let r = 1; r < rounds; r++) {
                    const t = (k - 1 + r * h) * dt;
                    svg("line", { x1: x(t), x2: x(t), y1: m.t, y2: m.t + ph, stroke: c.fgLighter, "stroke-dasharray": "2 3" }, chartSvg);
                }
            }

            const plot = svg("g", { "clip-path": "url(#" + clipId + ")" }, chartSvg);

            // Truth, at the integrator's resolution.
            const fineDt = dt / sub;
            const step = Math.max(1, Math.floor(fine.length / (pw * 1.5)));
            let d = "";
            for (let i = 0; i < fine.length; i += step) {
                d += (i ? "L" : "M") + x(i * fineDt).toFixed(1) + " " + y(fine[i] * DEG).toFixed(1);
            }
            d += "L" + x(tEnd).toFixed(1) + " " + y(fine[fine.length - 1] * DEG).toFixed(1);
            svg("path", { d, fill: "none", stroke: c.fg, "stroke-width": 1.6, "stroke-linejoin": "round" }, plot);

            // The k samples the model sees.
            for (let i = 0; i < k; i++) {
                svg("circle", { cx: x(i * dt), cy: y(samples[i] * DEG), r: 2.2, fill: c.train }, plot);
            }

            // Forecast, starting from the last context value.
            if (pred) {
                let p = "M" + x(tCtx).toFixed(1) + " " + y(samples[k - 1] * DEG).toFixed(1);
                for (let i = 0; i < pred.length; i++) {
                    p += "L" + x((k + i) * dt).toFixed(1) + " " + y(pred[i] * DEG).toFixed(1);
                }
                svg("path", {
                    d: p,
                    fill: "none",
                    stroke: c.accent,
                    "stroke-width": 2.2,
                    "stroke-dasharray": "6 4",
                    "stroke-linejoin": "round",
                }, plot);
            }

            cursorLine = reduced
                ? null
                : svg("line", { x1: x(0), x2: x(0), y1: m.t, y2: m.t + ph, stroke: c.fgLighter, "stroke-width": 1 }, chartSvg);
            xScale = x;

            // Legend and readout.
            legend.replaceChildren();
            [
                [c.fg, "true angle (RK4 simulation)"],
                [c.train, "context: the " + k + " samples the model sees"],
                [c.accent, "model forecast (dashed)"],
            ].forEach(([color, text]) => {
                const s = html("span", { textContent: text }, legend);
                s.style.setProperty("--swatch", color);
            });

            if (pred) {
                const truth = samples.subarray(k, k + pred.length);
                const naive = new Float64Array(pred.length).fill(samples[k - 1]);
                readout.textContent =
                    "RMSE over the " +
                    pred.length +
                    " forecast steps: " +
                    fmtDeg(rmse(pred, truth) * DEG) +
                    " (repeating the last value: " +
                    fmtDeg(rmse(naive, truth) * DEG) +
                    "). Inference took " +
                    (ms < 1 ? ms.toFixed(2) : ms.toFixed(1)) +
                    " ms in your browser.";
            } else {
                readout.textContent = loadFailed ? "" : "The forecast appears once the model has loaded.";
            }

            drawBob();
            if (reduced || !rafId) renderFrame(reduced ? 0 : animT);
        }

        function drawBob() {
            const c = colors;
            const [lMin, lMax] = meta.L_range;
            const R = 30 + 26 * ((state.length - lMin) / Math.max(lMax - lMin, 1e-9));
            const a = state.theta0Deg / DEG;
            bobSvg.replaceChildren();
            // Reachable arc, +/- theta0.
            const arc =
                "M" + (R * Math.sin(-a)).toFixed(2) + " " + (R * Math.cos(-a)).toFixed(2) +
                "A" + R + " " + R + " 0 " + (a > Math.PI / 2 ? 1 : 0) + " 0 " +
                (R * Math.sin(a)).toFixed(2) + " " + (R * Math.cos(a)).toFixed(2);
            svg("path", { d: arc, fill: "none", stroke: c.fgLightest, "stroke-width": 1.5 }, bobSvg);
            svg("line", { x1: -10, x2: 10, y1: 0, y2: 0, stroke: c.fgLighter, "stroke-width": 2 }, bobSvg);
            const ghostRod = svg("line", { x1: 0, y1: 0, stroke: c.accent, "stroke-width": 1.5, "stroke-dasharray": "3 2" }, bobSvg);
            const ghost = svg("circle", { r: 6, fill: "none", stroke: c.accent, "stroke-width": 2 }, bobSvg);
            const rod = svg("line", { x1: 0, y1: 0, stroke: c.fg, "stroke-width": 1.5 }, bobSvg);
            const bob = svg("circle", { r: 6, fill: c.fg }, bobSvg);
            svg("circle", { r: 2, fill: c.fgLight }, bobSvg);
            bobParts = { R, rod, bob, ghostRod, ghost };
        }

        // ------------------------------------------------------------------
        // Animation: a time cursor on the chart and a swinging pendulum.
        // ------------------------------------------------------------------
        function renderFrame(t) {
            if (!current || !bobParts) return;
            const { samples, pred } = current;
            const k = meta.k;
            const dt = meta.dt;
            const u = t / dt; // sample index
            const place = (rod, ball, theta, show) => {
                const px = bobParts.R * Math.sin(theta);
                const py = bobParts.R * Math.cos(theta);
                rod.setAttribute("x2", px.toFixed(2));
                rod.setAttribute("y2", py.toFixed(2));
                ball.setAttribute("cx", px.toFixed(2));
                ball.setAttribute("cy", py.toFixed(2));
                rod.style.display = ball.style.display = show ? "" : "none";
            };
            place(bobParts.rod, bobParts.bob, sampleAt(current.fine, t / (dt / meta.rk4_substeps)), true);
            const showGhost = !!pred && u >= k - 1 && u <= k - 1 + pred.length;
            if (showGhost) {
                const series = new Float64Array(pred.length + 1);
                series[0] = samples[k - 1];
                series.set(pred, 1);
                place(bobParts.ghostRod, bobParts.ghost, sampleAt(series, u - (k - 1)), true);
            } else {
                place(bobParts.ghostRod, bobParts.ghost, 0, false);
            }
            if (cursorLine) {
                const cx = xScale(Math.min(t, (samples.length - 1) * dt)).toFixed(1);
                cursorLine.setAttribute("x1", cx);
                cursorLine.setAttribute("x2", cx);
            }
        }

        function frame(now) {
            rafId = 0;
            if (disposed || !el.isConnected || !visible || !current) return;
            const dtReal = lastFrame ? Math.min((now - lastFrame) / 1000, 0.1) : 0;
            lastFrame = now;
            const tEnd = (current.samples.length - 1) * meta.dt;
            const hold = 1.5 * ANIMATION_SPEED; // pause at the end, in simulated seconds
            animT += dtReal * ANIMATION_SPEED;
            if (animT > tEnd + hold) animT = 0;
            renderFrame(Math.min(animT, tEnd));
            rafId = requestAnimationFrame(frame);
        }

        function kickAnimation() {
            if (reduced || rafId || disposed || !visible || !current) return;
            lastFrame = 0;
            rafId = requestAnimationFrame(frame);
        }

        // ------------------------------------------------------------------
        // Theme and cleanup
        // ------------------------------------------------------------------
        cleanups.push(
            api.onThemeChange((c) => {
                if (!el.isConnected) return;
                colors = c;
                draw();
            })
        );

        return () => {
            disposed = true;
            clearTimeout(timer);
            if (rafId) cancelAnimationFrame(rafId);
            if (io) io.disconnect();
            cleanups.forEach((fn) => fn());
            if (session) release(session);
        };
    });
})();
