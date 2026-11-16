/*
 * Book map: the pages of the book and the links between them, in two views.
 *
 *   Graph  every page is a dot coloured by chapter, every link a line. The
 *          layout is computed in the browser: linked pages pull together,
 *          all pages push apart, and each chapter has its own corner.
 *   List   every page is a chip, grouped by chapter. This view is also the
 *          keyboard-friendly one.
 *
 * Selecting a page, in either view, highlights the pages it links to and
 * the pages that link to it, and lists them with links.
 *
 * The data is assets/book-map.json, written at build time by
 * scripts/stage_site.py from the links in the built pages.
 *
 *     <div class="dl-widget" data-widget="book-map" markdown>
 *     Fallback text for readers without JavaScript.
 *     </div>
 */
dlWidgets.register("book-map", (el, api) => {
    const NS = "http://www.w3.org/2000/svg";
    // One colour per chapter, readable on light and dark backgrounds.
    const PALETTE = ["#8a8f98", "#4a90d9", "#e8743b", "#2e9e6b", "#b05cc6", "#d9a400", "#d94a6a", "#3aa7b8", "#7a8b2e", "#946b4a"];
    const OUT = "var(--md-accent-fg-color)", IN = "#4a90d9";
    const style = document.createElement("style");
    style.textContent = `
.dl-map__chapter { margin: .5rem 0 .2rem; font-size: .62rem; font-weight: 700; color: var(--md-default-fg-color--light); }
.dl-map__chips { display: flex; flex-wrap: wrap; gap: .25rem; }
.dl-map__chip { max-width: 100%; padding: .12rem .5rem; border: 1px solid var(--md-default-fg-color--lighter); border-radius: 1rem;
  background: transparent; color: var(--md-default-fg-color); font: inherit; font-size: .62rem; line-height: 1.5; cursor: pointer;
  overflow: hidden; text-overflow: ellipsis; white-space: nowrap; }
.dl-map__chip:focus-visible { outline: 2px solid var(--md-accent-fg-color); outline-offset: 1px; }
.dl-map--picked .dl-map__chip { opacity: .4; }
.dl-map--picked .dl-map__chip[data-role] { opacity: 1; }
.dl-map__chip[data-role="self"] { background: var(--md-default-fg-color); color: var(--md-default-bg-color); border-color: var(--md-default-fg-color); }
.dl-map__chip[data-role="out"] { background: var(--md-accent-fg-color); color: #fff; border-color: var(--md-accent-fg-color); }
.dl-map__chip[data-role="in"] { background: #4a90d9; color: #fff; border-color: #4a90d9; }
.dl-map__chip[data-role="both"] { background: linear-gradient(90deg, var(--md-accent-fg-color) 50%, #4a90d9 50%); color: #fff; border-color: transparent; }
.dl-map__detail { margin: .6rem 0 0; padding: .5rem .7rem; border-radius: .3rem; background: var(--md-default-fg-color--lightest); font-size: .68rem; line-height: 1.6; }
.md-typeset .dl-map__detail p { margin: 0 0 .2rem; }
.dl-map__detail a { margin-right: .6rem; }
.dl-map__graph circle[data-i] { cursor: pointer; }
.dl-map__graph text { pointer-events: none; paint-order: stroke; stroke: var(--md-default-bg-color); stroke-width: 3px; stroke-linejoin: round; }`;
    const status = document.createElement("p");
    status.className = "dl-widget__caption";
    status.textContent = "Loading the map…";
    el.innerHTML = "";
    el.append(style, status);

    let cleanup = () => {};
    fetch(api.resolve("assets/book-map.json"))
        .then((r) => (r.ok ? r.json() : Promise.reject(new Error(r.status))))
        .then((data) => {
            cleanup = build(data);
        })
        .catch(() => {
            status.textContent = "The map could not be loaded.";
        });
    return () => cleanup();

    // Positions in a unit square. Deterministic, so the picture is the same on every visit.
    function layout(pages, chapters) {
        const n = pages.length;
        const used = [...new Set(pages.map((p) => p.c))];
        const home = {};
        used.forEach((c, k) => {
            const a = (2 * Math.PI * k) / used.length - Math.PI / 2;
            home[c] = [0.5 + 0.3 * Math.cos(a), 0.5 + 0.3 * Math.sin(a)];
        });
        let seed = 7;
        const rand = () => ((seed = (seed * 16807) % 2147483647) / 2147483647 - 0.5);
        const pos = pages.map((p) => [home[p.c][0] + rand() * 0.2, home[p.c][1] + rand() * 0.2]);
        const edges = [];
        pages.forEach((p, i) => p.o.forEach((t) => edges.push([i, t])));
        for (let it = 0; it < 320; it++) {
            const heat = 0.02 * (1 - it / 320) + 0.002;
            const f = pos.map(() => [0, 0]);
            for (let i = 0; i < n; i++) {
                for (let j = i + 1; j < n; j++) {
                    let dx = pos[i][0] - pos[j][0], dy = pos[i][1] - pos[j][1];
                    const d2 = dx * dx + dy * dy + 1e-5, k = 0.0004 / d2;
                    dx *= k; dy *= k;
                    f[i][0] += dx; f[i][1] += dy; f[j][0] -= dx; f[j][1] -= dy;
                }
            }
            for (const [a, b] of edges) {
                const dx = pos[b][0] - pos[a][0], dy = pos[b][1] - pos[a][1];
                const d = Math.hypot(dx, dy) + 1e-6, k = (d - 0.06) * 0.35;
                f[a][0] += (dx / d) * k; f[a][1] += (dy / d) * k; f[b][0] -= (dx / d) * k; f[b][1] -= (dy / d) * k;
            }
            for (let i = 0; i < n; i++) {
                const h = home[pages[i].c];
                f[i][0] += (h[0] - pos[i][0]) * 0.16;
                f[i][1] += (h[1] - pos[i][1]) * 0.16;
                const len = Math.hypot(f[i][0], f[i][1]) + 1e-9, step = Math.min(len, heat);
                pos[i][0] += (f[i][0] / len) * step;
                pos[i][1] += (f[i][1] / len) * step;
            }
        }
        let x0 = Infinity, x1 = -Infinity, y0 = Infinity, y1 = -Infinity;
        for (const [x, y] of pos) { x0 = Math.min(x0, x); x1 = Math.max(x1, x); y0 = Math.min(y0, y); y1 = Math.max(y1, y); }
        return { pos: pos.map(([x, y]) => [(x - x0) / (x1 - x0 || 1), (y - y0) / (y1 - y0 || 1)]), edges };
    }

    function build(data) {
        const { chapters, pages } = data;
        const incoming = pages.map(() => []);
        pages.forEach((p, i) => p.o.forEach((t) => incoming[t].push(i)));
        const degree = pages.map((p, i) => p.o.length + incoming[i].length);
        const base = api.resolve("");
        const href = (p) => base + (p.u ? p.u + "/" : "");
        const color = (c) => PALETTE[c % PALETTE.length];
        const { pos, edges } = layout(pages, chapters);
        let picked = -1, view = "graph", W = 640;

        // View switch.
        const controls = document.createElement("div");
        controls.className = "dl-widget__controls";
        const seg = document.createElement("div");
        seg.className = "dl-widget__seg";
        seg.setAttribute("role", "group");
        seg.setAttribute("aria-label", "View");
        const viewButtons = ["graph", "list"].map((v) => {
            const b = document.createElement("button");
            b.type = "button";
            b.textContent = v === "graph" ? "Graph" : "List";
            b.addEventListener("click", () => {
                view = v;
                render();
            });
            seg.appendChild(b);
            return [v, b];
        });
        controls.appendChild(seg);

        const legend = document.createElement("div");
        legend.className = "dl-widget__legend";
        const svg = document.createElementNS(NS, "svg");
        svg.setAttribute("role", "img");
        svg.setAttribute("class", "dl-map__graph");
        svg.addEventListener("click", (event) => {
            const i = event.target.dataset ? event.target.dataset.i : undefined;
            pick(i === undefined ? -1 : +i === picked ? -1 : +i);
        });

        const list = document.createElement("div");
        list.className = "dl-map";
        const chips = [];
        chapters.forEach((name, c) => {
            const members = pages.map((p, i) => [p, i]).filter(([p]) => p.c === c);
            if (!members.length) return;
            const h = document.createElement("p");
            h.className = "dl-map__chapter";
            h.textContent = name;
            const row = document.createElement("div");
            row.className = "dl-map__chips";
            for (const [p, i] of members) {
                const b = document.createElement("button");
                b.type = "button";
                b.className = "dl-map__chip";
                b.textContent = p.t;
                b.title = p.t;
                b.addEventListener("click", () => pick(picked === i ? -1 : i));
                chips[i] = b;
                row.appendChild(b);
            }
            list.append(h, row);
        });
        const detail = document.createElement("div");
        detail.className = "dl-map__detail";
        detail.setAttribute("aria-live", "polite");
        status.remove();
        el.append(controls, legend, svg, list, detail);

        function node(tag, attrs, parent) {
            const n = document.createElementNS(NS, tag);
            for (const k in attrs) n.setAttribute(k, attrs[k]);
            if (parent) parent.appendChild(n);
            return n;
        }
        const roleOf = (k) => {
            if (picked < 0) return "";
            if (k === picked) return "self";
            const o = pages[picked].o.includes(k), i = incoming[picked].includes(k);
            return o && i ? "both" : o ? "out" : i ? "in" : "";
        };

        function drawGraph() {
            const c = api.colors();
            W = Math.max(280, Math.min(1000, Math.round(el.clientWidth - 2) || 640));
            const H = Math.round(W < 480 ? W * 1.15 : W * 0.66), pad = 16;
            const x = (i) => pad + pos[i][0] * (W - 2 * pad), y = (i) => pad + pos[i][1] * (H - 2 * pad);
            svg.setAttribute("viewBox", `0 0 ${W} ${H}`);
            svg.setAttribute("aria-label", `A graph of ${pages.length} pages and ${edges.length} links between them. The List view shows the same pages as buttons.`);
            svg.innerHTML = "";
            const any = picked >= 0;
            for (const [a, b] of edges) {
                const mine = a === picked || b === picked;
                node("line", {
                    x1: x(a), y1: y(a), x2: x(b), y2: y(b),
                    stroke: mine ? (a === picked ? OUT : IN) : c.fgLighter,
                    "stroke-width": mine ? 1.8 : 1, "stroke-opacity": any ? (mine ? 0.95 : 0.12) : 0.5,
                }, svg);
            }
            const labels = [];
            pages.forEach((p, i) => {
                const role = roleOf(i), r = 3.5 + Math.sqrt(degree[i]) * 1.1 + (role === "self" ? 2.5 : 0);
                const dot = node("circle", {
                    cx: x(i), cy: y(i), r, fill: color(p.c), "fill-opacity": any && !role ? 0.18 : 1,
                    stroke: role === "self" ? c.fg : role === "out" || role === "both" ? OUT : role === "in" ? IN : c.bg,
                    "stroke-width": role ? 2.2 : 1,
                }, svg);
                node("title", {}, dot).textContent = p.t;
                // A larger, invisible target for fingers.
                node("circle", { cx: x(i), cy: y(i), r: Math.max(r, 11), fill: "transparent", "data-i": i }, svg);
                if (role) labels.push([i, role]);
            });
            for (const [i, role] of labels) {
                const right = x(i) < W * 0.62;
                const t = node("text", {
                    x: x(i) + (right ? 9 : -9), y: y(i) + 3, "text-anchor": right ? "start" : "end",
                    "font-size": role === "self" ? 11 : 9.5, "font-weight": role === "self" ? 700 : 400, fill: c.fg,
                }, svg);
                const title = pages[i].t;
                t.textContent = title.length > 30 ? title.slice(0, 28) + "…" : title;
            }
        }

        function drawLegend() {
            legend.innerHTML = "";
            const items = view === "graph"
                ? [...new Set(pages.map((p) => p.c))].sort((a, b) => a - b).map((c) => [chapters[c], color(c)])
                : [["selected", "var(--md-default-fg-color)"], ["it links to", OUT], ["links to it", IN]];
            for (const [t, col] of items) {
                const s = document.createElement("span");
                s.style.setProperty("--swatch", col);
                s.textContent = t;
                legend.appendChild(s);
            }
        }

        function links(items) {
            const span = document.createElement("span");
            if (!items.length) span.textContent = "none";
            for (const i of items) {
                const a = document.createElement("a");
                a.href = href(pages[i]);
                a.textContent = pages[i].t;
                span.appendChild(a);
            }
            return span;
        }

        function drawDetail() {
            detail.innerHTML = "";
            if (picked < 0) {
                detail.textContent = `${pages.length} pages and ${edges.length} links. Select a page to see what it links to (orange) and what links to it (blue).`;
                return;
            }
            const head = document.createElement("p");
            const open = document.createElement("a");
            open.href = href(pages[picked]);
            open.textContent = pages[picked].t;
            const strong = document.createElement("strong");
            strong.appendChild(open);
            head.append(strong, document.createTextNode(` (${chapters[pages[picked].c]})`));
            const row = (label, items) => {
                const p = document.createElement("p");
                p.append(document.createTextNode(label + " "), links(items));
                return p;
            };
            detail.append(head, row("Links to:", pages[picked].o), row("Linked from:", incoming[picked]));
        }

        function render() {
            for (const [v, b] of viewButtons) b.setAttribute("aria-pressed", String(v === view));
            svg.style.display = view === "graph" ? "" : "none";
            list.style.display = view === "list" ? "" : "none";
            list.classList.toggle("dl-map--picked", picked >= 0);
            chips.forEach((b, k) => {
                if (!b) return;
                const role = roleOf(k);
                if (role) b.dataset.role = role;
                else b.removeAttribute("data-role");
                b.setAttribute("aria-pressed", String(k === picked));
            });
            drawLegend();
            if (view === "graph") drawGraph();
            drawDetail();
        }
        function pick(i) {
            picked = i;
            render();
        }

        render();
        const ro = "ResizeObserver" in window
            ? new ResizeObserver(() => {
                  const w = Math.round(el.clientWidth - 2);
                  if (w && Math.abs(w - W) > 8 && view === "graph") drawGraph();
              })
            : null;
        if (ro) ro.observe(el);
        const offTheme = api.onThemeChange(render);
        return () => {
            if (ro) ro.disconnect();
            offTheme();
        };
    }
});
