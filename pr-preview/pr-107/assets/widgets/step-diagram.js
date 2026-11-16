/*
 * Step-through diagrams: a box-and-arrow diagram that builds up one step at
 * a time, with a sentence for each step.
 *
 *     <div class="dl-widget" data-widget="step-diagram" data-diagram="rnn" markdown>
 *     Fallback text for readers without JavaScript.
 *     </div>
 *
 * A diagram is data, not code (see DIAGRAMS below):
 *
 *   nodes  { id, label, col, row }      position on a grid
 *   edges  { from, to, label?, bend? }  an arrow; bend curves it (±1)
 *   steps  { title, text, on: [...] }   ids of the nodes and edges ("a>b")
 *                                       this step adds
 *
 * Labels may use ^ and _ for one raised or lowered group: x^0, W_h, x^{n-1}.
 * Parts added by earlier steps stay visible, parts of later steps are faint,
 * and the current step is highlighted. On narrow screens a diagram with
 * `flip: true` is drawn top to bottom instead of left to right.
 */
dlWidgets.register("step-diagram", (el, api) => {
    const NS = "http://www.w3.org/2000/svg";
    const DIAGRAMS = {
        rnn: {
            flip: true,
            label: "A recurrent network unrolled over four time steps",
            nodes: [
                { id: "h0", label: "h(0)", col: 0, row: 0 },
                ...[1, 2, 3, 4].flatMap((t) => [
                    { id: `h${t}`, label: `h(${t})`, col: t, row: 0 },
                    { id: `x${t}`, label: `x(${t})`, col: t, row: 1 },
                ]),
            ],
            edges: [1, 2, 3, 4].flatMap((t) => [
                { from: `h${t - 1}`, to: `h${t}`, label: "W_h" },
                { from: `x${t}`, to: `h${t}`, label: "W_x" },
            ]),
            steps: [
                { title: "The initial state", text: "Before any input arrives, the unit has a state h(0).", on: ["h0"] },
                ...[1, 2, 3, 4].map((t) => ({
                    title: `Time step ${t}`,
                    text: `The new state mixes the previous state with the new input: h(${t}) = f(W_h h(${t - 1}) + W_x x(${t}) + b).`,
                    on: [`x${t}`, `h${t}`, `h${t - 1}>h${t}`, `x${t}>h${t}`],
                })),
                {
                    title: "One unit, used again and again",
                    text: "Every step applies the same W_h, W_x and b. The diagram is one unit unrolled in time, and h(4) carries information from all four inputs.",
                    on: ["h0", "h1", "h2", "h3", "h4", "x1", "x2", "x3", "x4", "h0>h1", "h1>h2", "h2>h3", "h3>h4", "x1>h1", "x2>h2", "x3>h3", "x4>h4"],
                },
            ],
        },
        attention: {
            label: "Scaled dot-product attention",
            nodes: [
                { id: "X", label: "X", col: 1, row: 0 },
                { id: "Q", label: "Q = X W^Q", col: 0, row: 1 },
                { id: "K", label: "K = X W^K", col: 1, row: 1 },
                { id: "V", label: "V = X W^V", col: 2, row: 1 },
                { id: "A", label: "A = Q K^T", col: 0.5, row: 2 },
                { id: "S", label: "softmax(A / √d_k)", col: 0.5, row: 3 },
                { id: "O", label: "output", col: 1.25, row: 4 },
            ],
            edges: [
                { from: "X", to: "Q" }, { from: "X", to: "K" }, { from: "X", to: "V" },
                { from: "Q", to: "A" }, { from: "K", to: "A" }, { from: "A", to: "S" },
                { from: "S", to: "O" }, { from: "V", to: "O" },
            ],
            steps: [
                { title: "The input", text: "X holds the sequence: one row for each element.", on: ["X"] },
                { title: "Query, key and value", text: "Three linear layers turn the same input into the query Q, the key K and the value V.", on: ["Q", "K", "V", "X>Q", "X>K", "X>V"] },
                { title: "Scores", text: "A = Q Kᵀ compares every query with every key: one number for each pair of positions.", on: ["A", "Q>A", "K>A"] },
                { title: "Weights", text: "The scores are divided by √d_k and passed through a softmax, so each row becomes weights that add up to 1.", on: ["S", "A>S"] },
                { title: "Mixing the values", text: "The weights are multiplied with V: each row of the output is a weighted mix of the rows of V. If the weights were the identity, nothing would be mixed.", on: ["O", "S>O", "V>O"] },
            ],
        },
        diffusion: {
            flip: true,
            label: "The forward and reverse process of a diffusion model with five steps",
            nodes: [0, 1, 2, 3, 4, 5].map((n) => ({ id: `x${n}`, label: `x^${n}`, col: n, row: 0 })),
            edges: [
                ...[1, 2, 3, 4, 5].map((n) => ({ from: `x${n - 1}`, to: `x${n}`, bend: -1, label: n === 3 ? "q" : "" })),
                ...[5, 4, 3, 2, 1].map((n) => ({ from: `x${n}`, to: `x${n - 1}`, bend: -1, label: n === 3 ? "p_θ" : "" })),
            ],
            steps: [
                { title: "The data", text: "x^0 is a sample of real data.", on: ["x0"] },
                { title: "The forward process", text: "Each step adds a little Gaussian noise: q(x^n | x^{n−1}) = N(√(1 − β_n) x^{n−1}, β_n I). Nothing is learned here.", on: ["x1", "x2", "x3", "x4", "x5", "x0>x1", "x1>x2", "x2>x3", "x3>x4", "x4>x5"] },
                { title: "Noise", text: "After the last step the information in the data is smeared out: x^5 is close to pure noise.", on: ["x5"] },
                { title: "The reverse process", text: "The model learns to undo one step at a time: p_θ(x^{n−1} | x^n) is a Gaussian whose mean and variance are predicted from x^n and n.", on: ["x5>x4", "x4>x3", "x3>x2", "x2>x1", "x1>x0"] },
                { title: "Generating", text: "To generate, start from noise x^5 and apply the five reverse steps. What comes out at x^0 is a new sample.", on: ["x5", "x4", "x3", "x2", "x1", "x0", "x5>x4", "x4>x3", "x3>x2", "x2>x1", "x1>x0"] },
            ],
        },
        vae: {
            flip: true,
            label: "One pass through a variational autoencoder",
            nodes: [
                { id: "x", label: "x", col: 0, row: 1 },
                { id: "enc", label: "encoder q_φ(z|x)", col: 1, row: 1 },
                { id: "mu", label: "mean μ", col: 2, row: 0 },
                { id: "sig", label: "variance Σ", col: 2, row: 2 },
                { id: "z", label: "z", col: 3, row: 1 },
                { id: "prior", label: "prior p(z)", col: 3, row: 2.2 },
                { id: "dec", label: "decoder p_θ(x|z)", col: 4, row: 1 },
                { id: "xr", label: "x̂", col: 5, row: 1 },
            ],
            edges: [
                { from: "x", to: "enc" }, { from: "enc", to: "mu" }, { from: "enc", to: "sig" },
                { from: "mu", to: "z" }, { from: "sig", to: "z" }, { from: "z", to: "dec" }, { from: "dec", to: "xr" },
                { from: "prior", to: "z", dashed: true },
            ],
            steps: [
                { title: "The data", text: "x is one data point.", on: ["x"] },
                { title: "Encoding", text: "The encoder does not output a single code. It outputs the parameters of the distribution q_φ(z | x): a mean and a variance.", on: ["enc", "mu", "sig", "x>enc", "enc>mu", "enc>sig"] },
                { title: "Sampling", text: "A latent z is drawn from the Gaussian with that mean and variance.", on: ["z", "mu>z", "sig>z"] },
                { title: "Decoding", text: "The decoder p_θ(x | z) turns the latent z back into data.", on: ["dec", "xr", "z>dec", "dec>xr"] },
                { title: "The loss", text: "Two terms: E_q ln p_θ(x | z) rewards a good reconstruction, and −D_{KL}(q_φ(z | x) ‖ p(z)) keeps the encoder's distribution close to the prior p(z).", on: ["prior", "prior>z", "xr", "x"] },
            ],
        },
    };

    const diagram = DIAGRAMS[el.dataset.diagram];
    if (!diagram) throw new Error(`unknown diagram: ${el.dataset.diagram}`);
    const uid = "dlsd" + Math.random().toString(36).slice(2, 8);
    let step = 0;

    el.innerHTML = "";
    const controls = document.createElement("div");
    controls.className = "dl-widget__controls";
    const button = (text, fn) => {
        const b = document.createElement("button");
        b.type = "button";
        b.className = "dl-widget__btn";
        b.textContent = text;
        b.addEventListener("click", fn);
        return b;
    };
    const prev = button("← Back", () => go(step - 1));
    const next = button("Next →", () => go(step + 1));
    const count = document.createElement("span");
    controls.append(prev, next, count);
    const svg = document.createElementNS(NS, "svg");
    svg.setAttribute("role", "img");
    const caption = document.createElement("p");
    caption.className = "dl-widget__caption";
    caption.setAttribute("aria-live", "polite");
    caption.style.minHeight = "3.6em";
    el.append(controls, svg, caption);

    function go(k) {
        step = Math.max(0, Math.min(diagram.steps.length - 1, k));
        render();
    }
    function node(tag, attrs, parent) {
        const n = document.createElementNS(NS, tag);
        for (const k in attrs) n.setAttribute(k, attrs[k]);
        if (parent) parent.appendChild(n);
        return n;
    }
    // "x^{n-1}" and "W_h" become raised and lowered tspans.
    function label(parent, text, attrs) {
        const t = node("text", attrs, parent);
        const re = /([\^_])(\{[^}]*\}|.)/g;
        let last = 0, m, shift = 0;
        const add = (s, dy, small) => {
            if (!s) return;
            const span = node("tspan", small ? { dy, "font-size": "0.72em" } : { dy }, t);
            span.textContent = s;
        };
        while ((m = re.exec(text))) {
            add(text.slice(last, m.index), -shift, false);
            shift = 0;
            const up = m[1] === "^";
            add(m[2].replace(/^\{|\}$/g, ""), up ? -5 : 4, true);
            shift = up ? -5 : 4;
            last = re.lastIndex;
        }
        add(text.slice(last), -shift, false);
        return t;
    }

    let W = 640;
    function render() {
        const c = api.colors();
        const steps = diagram.steps;
        const now = new Set(steps[step].on);
        const seen = new Set(steps.slice(0, step).flatMap((s) => s.on));
        const stateOf = (id) => (now.has(id) ? "on" : seen.has(id) ? "seen" : "off");

        W = Math.max(280, Math.min(720, Math.round(el.clientWidth - 2) || 640));
        const flip = diagram.flip && W < 520;
        const pos = (n) => (flip ? { col: n.row, row: n.col } : { col: n.col, row: n.row });
        const cols = Math.max(...diagram.nodes.map((n) => pos(n).col)) + 1;
        const rows = Math.max(...diagram.nodes.map((n) => pos(n).row)) + 1;
        const cellW = (W - 8) / cols, cellH = flip ? 54 : 64;
        const longest = Math.max(...diagram.nodes.map((n) => n.label.replace(/[\^_{}]/g, "").length));
        const font = Math.max(9, Math.min(12.5, (cellW * 0.86) / (longest * 0.56)));
        const bw = Math.min(cellW * 0.9, Math.max(46, longest * font * 0.6 + 14)), bh = 28;
        const H = rows * cellH + 12;
        svg.setAttribute("viewBox", `0 0 ${W} ${H}`);
        svg.setAttribute("aria-label", `${diagram.label}. Step ${step + 1} of ${steps.length}: ${steps[step].title}.`);
        svg.innerHTML = "";
        const defs = node("defs", {}, svg);
        for (const [name, col] of [["on", c.accent], ["seen", c.fgLight], ["off", c.fgLighter]]) {
            const mk = node("marker", { id: `${uid}-${name}`, viewBox: "0 0 10 10", refX: 9, refY: 5, markerWidth: 6, markerHeight: 6, orient: "auto-start-reverse" }, defs);
            node("path", { d: "M0,0 L10,5 L0,10 z", fill: col }, mk);
        }
        const center = {};
        for (const n of diagram.nodes) {
            const p = pos(n);
            center[n.id] = { x: 4 + cellW * (p.col + 0.5), y: 6 + cellH * (p.row + 0.5) };
        }
        // Where the line from a box's centre towards a point leaves the box.
        function border(id, to) {
            const a = center[id], dx = to.x - a.x, dy = to.y - a.y;
            if (!dx && !dy) return a;
            const k = Math.min(dx ? (bw / 2 + 2) / Math.abs(dx) : Infinity, dy ? (bh / 2 + 2) / Math.abs(dy) : Infinity);
            return { x: a.x + dx * k, y: a.y + dy * k };
        }
        const stroke = { on: c.accent, seen: c.fgLight, off: c.fgLighter };
        for (const e of diagram.edges) {
            const st = stateOf(`${e.from}>${e.to}`);
            const a = center[e.from], b = center[e.to];
            let ctrl = { x: (a.x + b.x) / 2, y: (a.y + b.y) / 2 };
            if (e.bend) {
                const len = Math.hypot(b.x - a.x, b.y - a.y) || 1;
                ctrl = { x: ctrl.x + ((b.y - a.y) / len) * 26 * e.bend, y: ctrl.y - ((b.x - a.x) / len) * 26 * e.bend };
            }
            const p0 = border(e.from, ctrl), p1 = border(e.to, ctrl);
            node("path", {
                d: `M${p0.x.toFixed(1)},${p0.y.toFixed(1)} Q${ctrl.x.toFixed(1)},${ctrl.y.toFixed(1)} ${p1.x.toFixed(1)},${p1.y.toFixed(1)}`,
                fill: "none", stroke: stroke[st], "stroke-width": st === "on" ? 2 : 1.3, opacity: st === "off" ? 0.35 : 1,
                "stroke-dasharray": e.dashed ? "4 3" : "none", "marker-end": `url(#${uid}-${st})`,
            }, svg);
            if (e.label) {
                const off = e.bend ? 9 * Math.sign(ctrl.y - (a.y + b.y) / 2 || ctrl.x - (a.x + b.x) / 2) : 0;
                label(svg, e.label, {
                    x: e.bend ? ctrl.x + (a.y !== b.y ? off : 0) : ctrl.x + (a.y === b.y ? 0 : 8),
                    y: e.bend ? ctrl.y + (a.y === b.y ? off : 3) : ctrl.y - (a.y === b.y ? 5 : 0),
                    "text-anchor": a.y === b.y || e.bend ? "middle" : "start", "font-size": 10,
                    fill: st === "off" ? c.fgLighter : c.fgLight,
                });
            }
        }
        for (const n of diagram.nodes) {
            const st = stateOf(n.id), p = center[n.id];
            node("rect", {
                x: p.x - bw / 2, y: p.y - bh / 2, width: bw, height: bh, rx: 6,
                fill: st === "on" ? c.accent : c.bg, "fill-opacity": st === "on" ? 0.16 : 1,
                stroke: stroke[st], "stroke-width": st === "on" ? 2 : 1.2, opacity: st === "off" ? 0.4 : 1,
            }, svg);
            label(svg, n.label, { x: p.x, y: p.y + font * 0.36, "text-anchor": "middle", "font-size": font, fill: c.fg, opacity: st === "off" ? 0.4 : 1 });
        }

        prev.disabled = step === 0;
        next.disabled = step === steps.length - 1;
        count.textContent = `Step ${step + 1} of ${steps.length}`;
        caption.innerHTML = "";
        const b = document.createElement("b");
        b.textContent = steps[step].title + ". ";
        caption.appendChild(b);
        // The same ^ and _ markup as in the labels, as <sup> and <sub>.
        const text = steps[step].text, re = /([\^_])(\{[^}]*\}|.)/g;
        let last = 0, m;
        while ((m = re.exec(text))) {
            caption.appendChild(document.createTextNode(text.slice(last, m.index)));
            const part = document.createElement(m[1] === "^" ? "sup" : "sub");
            part.textContent = m[2].replace(/^\{|\}$/g, "");
            caption.appendChild(part);
            last = re.lastIndex;
        }
        caption.appendChild(document.createTextNode(text.slice(last)));
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
        if (ro) ro.disconnect();
        offTheme();
    };
});
