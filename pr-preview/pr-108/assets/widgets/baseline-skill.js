/*
 * Baselines and skill: a series with a weekly pattern, a model's forecast for
 * the next two weeks, and two baselines (naive: the last value; seasonal
 * naive: the value one week earlier). Two sliders set how strong the weekly
 * pattern is and how large the model's errors are. The widget shows
 *
 *   - the MAE of the model and of both baselines on the test period,
 *   - the skill score 1 − MAE / MAE_baseline against each baseline,
 *   - MASE with the naive and with the seasonal naive scaling, both computed
 *     on the training period.
 *
 * The same model can look skilful against the naive forecast and not at all
 * against the seasonal naive one.
 *
 *     <div class="dl-widget" data-widget="baseline-skill" markdown>
 *     Fallback text for readers without JavaScript.
 *     </div>
 */
dlWidgets.register("baseline-skill", (el, api) => {
    const NS = "http://www.w3.org/2000/svg";
    const M = 7, TRAIN = 56, TEST = 14, SHOWN = 21; // period, lengths, training days drawn
    const PATTERN = [0, 0.3, 0.5, 0.4, 0.9, 1.6, 1.2].map((v) => v - 0.7); // one week, mean zero
    const state = { season: 2, error: 0.6 };
    const PRESETS = [
        ["Weekly pattern, decent model", { season: 2, error: 0.6 }],
        ["No pattern", { season: 0, error: 0.6 }],
        ["Model no better than last week", { season: 2, error: 0.8 }],
    ];
    // Fixed pseudo-random numbers in [-1, 1], so the picture is the same on every visit.
    const jitter = (i, salt) => {
        const x = Math.sin((i + 1) * 12.9898 + salt * 78.233) * 43758.5453;
        return 2 * (x - Math.floor(x)) - 1;
    };

    function compute() {
        const y = Array.from({ length: TRAIN + TEST }, (_, t) => 10 + 0.02 * t + state.season * PATTERN[t % M] + 0.35 * jitter(t, 1));
        const train = y.slice(0, TRAIN), test = y.slice(TRAIN);
        const naive = test.map(() => train[TRAIN - 1]);
        const seasonal = test.map((_, j) => train[TRAIN + j - M * (Math.floor(j / M) + 1)]);
        const model = test.map((v, j) => v + state.error * jitter(j, 2));
        const mae = (f) => f.reduce((s, v, j) => s + Math.abs(test[j] - v), 0) / TEST;
        let s1 = 0, sm = 0;
        for (let t = 1; t < TRAIN; t++) s1 += Math.abs(train[t] - train[t - 1]);
        for (let t = M; t < TRAIN; t++) sm += Math.abs(train[t] - train[t - M]);
        const scale1 = s1 / (TRAIN - 1), scaleM = sm / (TRAIN - M);
        const e = { model: mae(model), naive: mae(naive), seasonal: mae(seasonal) };
        return {
            y, train, test, naive, seasonal, model, e,
            skillNaive: 1 - e.model / e.naive,
            skillSeasonal: 1 - e.model / e.seasonal,
            mase: e.model / scale1,
            maseSeasonal: e.model / scaleM,
        };
    }

    el.innerHTML = "";
    const style = document.createElement("style");
    style.textContent = `
.dl-skill { display: grid; grid-template-columns: repeat(auto-fit, minmax(6.2rem, 1fr)); gap: .4rem; margin: .5rem 0 .2rem; }
.dl-skill > div { min-width: 0; padding: .3rem .45rem; border-radius: .3rem; background: var(--md-default-fg-color--lightest); line-height: 1.25; }
.dl-skill b { display: block; font-size: .6rem; font-weight: 600; color: var(--md-default-fg-color--light); }
.dl-skill span { font-variant-numeric: tabular-nums; font-size: .78rem; }
.dl-skill small { display: block; font-size: .58rem; color: var(--md-default-fg-color--light); }`;
    const presets = document.createElement("div");
    presets.className = "dl-widget__controls";
    const seg = document.createElement("div");
    seg.className = "dl-widget__seg";
    seg.style.flexWrap = "wrap";
    seg.setAttribute("role", "group");
    seg.setAttribute("aria-label", "Examples");
    for (const [label, values] of PRESETS) {
        const b = document.createElement("button");
        b.type = "button";
        b.textContent = label;
        b.addEventListener("click", () => {
            Object.assign(state, values);
            render();
        });
        seg.appendChild(b);
    }
    presets.appendChild(seg);
    const controls = document.createElement("div");
    controls.className = "dl-widget__controls";
    const inputs = {};
    function slider(label, key, min, max, step) {
        const wrap = document.createElement("label");
        const span = document.createElement("span");
        span.textContent = label;
        const input = document.createElement("input");
        Object.assign(input, { type: "range", min, max, step, value: state[key] });
        const out = document.createElement("output");
        out.style.minWidth = "2.2em";
        input.addEventListener("input", () => {
            state[key] = parseFloat(input.value);
            render();
        });
        wrap.append(span, input, out);
        inputs[key] = [input, out];
        return wrap;
    }
    controls.append(slider("Weekly pattern", "season", 0, 3, 0.1), slider("Model error", "error", 0, 2, 0.1));
    // Play: sweep the strength of the weekly pattern from none to strong.
    let timer = 0;
    const play = document.createElement("button");
    play.type = "button";
    play.className = "dl-widget__btn";
    play.textContent = "Play";
    function stop() {
        clearTimeout(timer);
        timer = 0;
        play.textContent = "Play";
    }
    play.addEventListener("click", () => {
        if (timer) return stop();
        play.textContent = "Pause";
        state.season = 0;
        const tick = () => {
            render();
            if (state.season >= 3) return stop();
            state.season = Math.min(3, Math.round((state.season + 0.1) * 10) / 10);
            timer = setTimeout(tick, 160);
        };
        tick();
    });
    controls.appendChild(play);
    const svg = document.createElementNS(NS, "svg");
    svg.setAttribute("role", "img");
    const legend = document.createElement("div");
    legend.className = "dl-widget__legend";
    const tiles = document.createElement("div");
    tiles.className = "dl-skill";
    const caption = document.createElement("p");
    caption.className = "dl-widget__caption";
    caption.setAttribute("aria-live", "polite");
    el.append(style, presets, controls, svg, legend, tiles, caption);

    function node(tag, attrs, parent) {
        const n = document.createElementNS(NS, tag);
        for (const k in attrs) n.setAttribute(k, attrs[k]);
        if (parent) parent.appendChild(n);
        return n;
    }
    const f2 = (v) => v.toFixed(2);

    let W = 640;
    function render() {
        const c = api.colors();
        const r = compute();
        for (const key in inputs) {
            inputs[key][0].value = state[key];
            inputs[key][1].textContent = state[key].toFixed(1);
        }
        for (const [i, b] of [...seg.children].entries()) {
            const v = PRESETS[i][1];
            b.setAttribute("aria-pressed", String(v.season === state.season && v.error === state.error));
        }

        W = Math.max(280, Math.min(720, Math.round(el.clientWidth - 2) || 640));
        const H = W < 480 ? 200 : 230, padL = 30, padR = 10, padT = 16, padB = 22;
        const first = TRAIN - SHOWN, count = SHOWN + TEST;
        const shown = r.y.slice(first).concat(r.model, r.naive, r.seasonal);
        const lo = Math.min(...shown) - 0.4, hi = Math.max(...shown) + 0.4;
        const x = (t) => padL + ((W - padL - padR) * (t - first + 0.5)) / count;
        const y = (v) => padT + ((H - padT - padB) * (hi - v)) / (hi - lo);
        svg.setAttribute("viewBox", `0 0 ${W} ${H}`);
        svg.setAttribute("aria-label", `A daily series with a weekly pattern, the model's forecast for 14 days and two baselines. Skill ${f2(r.skillNaive)} against the naive forecast and ${f2(r.skillSeasonal)} against the seasonal naive forecast.`);
        svg.innerHTML = "";
        node("rect", { x: x(TRAIN) - (x(1) - x(0)) / 2, y: padT - 6, width: W - padR - x(TRAIN) + (x(1) - x(0)) / 2, height: H - padT - padB + 6, fill: c.fg, "fill-opacity": 0.04 }, svg);
        node("text", { x: x(TRAIN), y: padT + 4, "font-size": 10, fill: c.fgLight }, svg).textContent = "test period";
        node("text", { x: padL, y: padT + 4, "font-size": 10, fill: c.fgLight }, svg).textContent = "end of training";
        for (let k = 0; k <= 3; k++) {
            const v = lo + ((hi - lo) * k) / 3;
            node("line", { x1: padL, x2: W - padR, y1: y(v), y2: y(v), stroke: c.fgLightest }, svg);
            node("text", { x: padL - 5, y: y(v) + 3, "text-anchor": "end", "font-size": 10, fill: c.fgLight }, svg).textContent = v.toFixed(0);
        }
        const path = (values, from) => values.map((v, j) => `${x(from + j).toFixed(1)},${y(v).toFixed(1)}`).join(" ");
        node("polyline", { points: path(r.y.slice(first), first), fill: "none", stroke: c.fg, "stroke-width": 1.6 }, svg);
        node("polyline", { points: path(r.naive, TRAIN), fill: "none", stroke: c.fgLight, "stroke-width": 1.6, "stroke-dasharray": "2 3" }, svg);
        node("polyline", { points: path(r.seasonal, TRAIN), fill: "none", stroke: c.train, "stroke-width": 1.8, "stroke-dasharray": "6 3" }, svg);
        r.model.forEach((v, j) => node("circle", { cx: x(TRAIN + j), cy: y(v), r: 3.6, fill: c.bg, stroke: c.accent, "stroke-width": 2 }, svg));

        legend.innerHTML = "";
        for (const [t, col] of [["truth", c.fg], ["model", c.accent], ["naive (dotted)", c.fgLight], ["seasonal naive (dashed)", c.train]]) {
            const s = document.createElement("span");
            s.style.setProperty("--swatch", col);
            s.textContent = t;
            legend.appendChild(s);
        }

        tiles.innerHTML = "";
        for (const [title, value, note] of [
            ["MAE: model", f2(r.e.model), "test period"],
            ["MAE: naive", f2(r.e.naive), "last value"],
            ["MAE: seasonal naive", f2(r.e.seasonal), "same day last week"],
            ["Skill vs naive", f2(r.skillNaive), "1 − MAE / MAE naive"],
            ["Skill vs seasonal naive", f2(r.skillSeasonal), "1 − MAE / MAE seasonal"],
            ["MASE", f2(r.mase), "naive scaling"],
            ["MASE, seasonal", f2(r.maseSeasonal), "seasonal naive scaling"],
        ]) {
            const cell = document.createElement("div");
            const b = document.createElement("b");
            b.textContent = title;
            const span = document.createElement("span");
            span.textContent = value;
            const small = document.createElement("small");
            small.textContent = note;
            cell.append(b, span, small);
            tiles.appendChild(cell);
        }

        const a = r.skillNaive, s = r.skillSeasonal;
        const pct = (v) => `${Math.abs(100 * v).toFixed(0)}%`;
        const say = (v, name) => (v >= 0 ? `${pct(v)} lower than the ${name} forecast's` : `${pct(v)} higher than the ${name} forecast's`);
        let why;
        if (a > 0.25 && s < 0.1) why = "Against the naive forecast the model looks strong, but copying last week does about as well: most of that skill is the weekly pattern, which the seasonal naive forecast gets for free.";
        else if (a > 0 && s > 0) why = "The model beats both baselines, so it has learned something beyond the last value and beyond last week's pattern.";
        else if (a <= 0 && s <= 0) why = "The model loses to both baselines: the simple methods are the better forecasters here.";
        else if (a <= 0) why = "The model beats copying last week but not the last value; with no weekly pattern, the last value is the stronger baseline.";
        else why = "The model beats the naive forecast but not the seasonal naive one: the stronger baseline is the one to report.";
        caption.textContent = `The model's MAE is ${say(a, "naive")} and ${say(s, "seasonal naive")}. ${why}`;
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
        clearTimeout(timer);
        if (ro) ro.disconnect();
        offTheme();
    };
});
