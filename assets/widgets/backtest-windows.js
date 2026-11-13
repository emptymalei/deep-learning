/*
 * Backtesting windows: sliding vs expanding train sets, with an optional gap.
 *
 *     <div class="dl-widget" data-widget="backtest-windows" markdown>
 *     ![Sliding Blocks](../assets/timeseries-evaluation.forecasting/block-slide.png)
 *     </div>
 *
 * Optional data attributes: data-steps (36), data-train (12), data-test (4),
 * data-gap (0), data-mode ("sliding" | "expanding").
 */
dlWidgets.register("backtest-windows", (el, api) => {
    const NS = "http://www.w3.org/2000/svg";
    const N = parseInt(el.dataset.steps || "36", 10);
    const state = {
        mode: el.dataset.mode === "expanding" ? "expanding" : "sliding",
        train: parseInt(el.dataset.train || "12", 10),
        test: parseInt(el.dataset.test || "4", 10),
        gap: parseInt(el.dataset.gap || "0", 10),
        fold: 0,
        playing: false,
    };

    // A deterministic toy series: trend + seasonality + a little noise.
    const series = Array.from({ length: N }, (_, t) => {
        const noise = Math.sin(t * 12.9898) * 43758.5453;
        return 2 + 0.05 * t + Math.sin((2 * Math.PI * t) / 12) + 0.25 * (noise - Math.floor(noise) - 0.5);
    });

    function folds() {
        const out = [];
        for (let i = 0; ; i++) {
            const testStart = state.train + state.gap + i * state.test;
            const testEnd = testStart + state.test - 1;
            if (testEnd > N - 1) break;
            const trainEnd = testStart - state.gap - 1;
            const trainStart = state.mode === "sliding" ? trainEnd - state.train + 1 : 0;
            out.push({ trainStart, trainEnd, testStart, testEnd });
        }
        return out;
    }

    function role(f, t) {
        if (t >= f.trainStart && t <= f.trainEnd) return "train";
        if (t > f.trainEnd && t < f.testStart) return "gap";
        if (t >= f.testStart && t <= f.testEnd) return "test";
        return "unused";
    }

    // ---- DOM ---------------------------------------------------------------
    el.innerHTML = "";

    const controls = document.createElement("div");
    controls.className = "dl-widget__controls";

    const seg = document.createElement("div");
    seg.className = "dl-widget__seg";
    seg.setAttribute("role", "group");
    seg.setAttribute("aria-label", "Train window");
    const modeButtons = ["sliding", "expanding"].map((m) => {
        const b = document.createElement("button");
        b.type = "button";
        b.textContent = m === "sliding" ? "Sliding" : "Expanding";
        b.addEventListener("click", () => {
            state.mode = m;
            state.fold = Math.min(state.fold, folds().length - 1);
            render();
        });
        seg.appendChild(b);
        return [m, b];
    });

    function slider(label, key, min, max) {
        const wrap = document.createElement("label");
        const span = document.createElement("span");
        const input = document.createElement("input");
        const out = document.createElement("output");
        input.type = "range";
        input.min = min;
        input.max = max;
        input.value = state[key];
        span.textContent = label;
        out.textContent = state[key];
        input.addEventListener("input", () => {
            state[key] = parseInt(input.value, 10);
            out.textContent = input.value;
            state.fold = Math.max(0, Math.min(state.fold, folds().length - 1));
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
    stepBtn.textContent = "Next fold";
    stepBtn.addEventListener("click", () => {
        setPlaying(false);
        advance();
    });

    controls.append(
        seg,
        slider("Train", "train", 4, 20),
        slider("Gap", "gap", 0, 4),
        slider("Test", "test", 1, 8),
        playBtn,
        stepBtn
    );

    const svg = document.createElementNS(NS, "svg");
    svg.setAttribute("role", "img");

    const caption = document.createElement("p");
    caption.className = "dl-widget__caption";
    caption.setAttribute("aria-live", "polite");

    const legend = document.createElement("div");
    legend.className = "dl-widget__legend";

    el.append(controls, svg, legend, caption);

    // ---- Drawing -----------------------------------------------------------
    function node(name, attrs, parent) {
        const n = document.createElementNS(NS, name);
        for (const k in attrs) n.setAttribute(k, attrs[k]);
        if (parent) parent.appendChild(n);
        return n;
    }

    function render() {
        const c = api.colors();
        const fs = folds();
        if (fs.length === 0) {
            svg.replaceChildren();
            caption.textContent = "No room for a test window — reduce the train, gap or test size.";
            return;
        }
        state.fold = Math.max(0, Math.min(state.fold, fs.length - 1));
        const cur = fs[state.fold];
        const fill = { train: c.train, gap: c.gap, test: c.test, unused: c.fgLightest };

        // Draw at the container's real width so labels stay legible on phones.
        const W = Math.max(320, Math.min(720, Math.round(el.clientWidth - 2) || 720));
        const padL = W < 480 ? 44 : 58, padR = 8;
        const cell = (W - padL - padR) / N;
        const chartH = 120, rowH = Math.max(10, Math.min(16, 260 / fs.length));
        const rowsTop = chartH + 26;
        const H = rowsTop + fs.length * rowH + 6;

        svg.setAttribute("viewBox", `0 0 ${W} ${H}`);
        svg.setAttribute(
            "aria-label",
            `${state.mode} window backtest, fold ${state.fold + 1} of ${fs.length}`
        );
        svg.replaceChildren();

        // Series panel with the current fold shaded.
        const min = Math.min(...series), max = Math.max(...series);
        const x = (t) => padL + (t + 0.5) * cell;
        const y = (v) => 10 + (1 - (v - min) / (max - min)) * (chartH - 20);
        const band = (a, b, color, op) =>
            node("rect", {
                x: padL + a * cell, y: 0, width: (b - a + 1) * cell, height: chartH,
                fill: color, opacity: op, class: "bw-anim",
            }, svg);
        band(cur.trainStart, cur.trainEnd, c.train, 0.14);
        if (state.gap > 0) band(cur.trainEnd + 1, cur.testStart - 1, c.gap, 0.18);
        band(cur.testStart, cur.testEnd, c.test, 0.18);

        node("polyline", {
            points: series.map((v, t) => `${x(t)},${y(v)}`).join(" "),
            fill: "none", stroke: c.fgLight, "stroke-width": 1.5,
        }, svg);
        series.forEach((v, t) => {
            const r = role(cur, t);
            node("circle", {
                cx: x(t), cy: y(v), r: r === "unused" ? 2 : 3.2,
                fill: r === "unused" ? c.fgLighter : fill[r], class: "bw-anim",
            }, svg);
        });
        const lbl = (txt, yy) =>
            node("text", { x: padL - 8, y: yy, "text-anchor": "end", "font-size": 11, fill: c.fgLight }, svg)
                .textContent = txt;
        lbl("series", chartH / 2 + 4);
        node("line", { x1: padL, x2: W - padR, y1: chartH + 0.5, y2: chartH + 0.5, stroke: c.fgLightest }, svg);

        // One row of blocks per fold, like the static figures.
        fs.forEach((f, i) => {
            const top = rowsTop + i * rowH;
            const isCur = i === state.fold;
            const g = node("g", {
                opacity: isCur ? 1 : i < state.fold ? 0.45 : 0.22,
                class: "bw-anim", style: "cursor:pointer",
            }, svg);
            g.addEventListener("click", () => {
                setPlaying(false);
                state.fold = i;
                render();
            });
            for (let t = 0; t < N; t++) {
                const r = role(f, t);
                node("rect", {
                    x: padL + t * cell + 0.75, y: top + 1, width: cell - 1.5, height: rowH - 3,
                    rx: 1.5, fill: fill[r], opacity: r === "unused" ? 1 : 0.9,
                }, g);
            }
            node("text", {
                x: padL - 8, y: top + rowH / 2 + 3.5, "text-anchor": "end", "font-size": 10,
                fill: isCur ? c.fg : c.fgLight, "font-weight": isCur ? 700 : 400,
            }, g).textContent = `fold ${i + 1}`;
        });

        // Controls, legend, caption.
        modeButtons.forEach(([m, b]) => b.setAttribute("aria-pressed", String(m === state.mode)));
        playBtn.textContent = state.playing ? "Pause" : "Play";
        legend.innerHTML = "";
        [["train", c.train], ["gap", c.gap], ["test", c.test]].forEach(([name, color]) => {
            if (name === "gap" && state.gap === 0) return;
            const s = document.createElement("span");
            s.style.setProperty("--swatch", color);
            s.textContent = name;
            legend.appendChild(s);
        });
        const trainLen = cur.trainEnd - cur.trainStart + 1;
        caption.textContent =
            `Fold ${state.fold + 1} of ${fs.length}: train on t=${cur.trainStart}–${cur.trainEnd} ` +
            `(${trainLen} steps)` +
            (state.gap ? `, skip ${state.gap}` : "") +
            `, test on t=${cur.testStart}–${cur.testEnd}. ` +
            (state.mode === "sliding"
                ? "The train window keeps its length and slides forward."
                : "The train window keeps its start and grows.");
    }

    // ---- Animation ---------------------------------------------------------
    let timer = null;
    function advance() {
        const n = folds().length;
        state.fold = n ? (state.fold + 1) % n : 0;
        render();
    }
    function setPlaying(on) {
        state.playing = on;
        clearInterval(timer);
        timer = on ? setInterval(advance, 1100) : null;
        render();
    }

    render();

    // Autoplay only while visible, and never when the reader prefers reduced motion.
    let io = null;
    if (!api.reducedMotion() && "IntersectionObserver" in window) {
        let autostarted = false;
        io = new IntersectionObserver((entries) => {
            const visible = entries.some((e) => e.isIntersecting);
            if (visible && !autostarted) {
                autostarted = true;
                setPlaying(true);
            } else if (!visible && state.playing) {
                setPlaying(false);
                autostarted = false;
            }
        }, { threshold: 0.4 });
        io.observe(el);
    }

    // Redraw when the column width changes (rotation, sidebar toggles).
    let lastWidth = el.clientWidth;
    const ro = "ResizeObserver" in window
        ? new ResizeObserver(() => {
            if (Math.abs(el.clientWidth - lastWidth) > 8) {
                lastWidth = el.clientWidth;
                requestAnimationFrame(render);
            }
        })
        : null;
    if (ro) ro.observe(el);

    const offTheme = api.onThemeChange(render);
    return () => {
        clearInterval(timer);
        if (io) io.disconnect();
        if (ro) ro.disconnect();
        offTheme();
    };
});
