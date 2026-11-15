/*
 * Book map: every page of the book as a chip, grouped by chapter. Selecting
 * a page highlights the pages it links to and the pages that link to it,
 * and lists them with links.
 *
 * The data is assets/book-map.json, written at build time by
 * scripts/stage_site.py from the links in the built pages.
 *
 *     <div class="dl-widget" data-widget="book-map" markdown>
 *     Fallback text for readers without JavaScript.
 *     </div>
 */
dlWidgets.register("book-map", (el, api) => {
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
.dl-map__detail a { margin-right: .6rem; }`;
    const status = document.createElement("p");
    status.className = "dl-widget__caption";
    status.textContent = "Loading the map…";
    el.innerHTML = "";
    el.append(style, status);

    fetch(api.resolve("assets/book-map.json"))
        .then((r) => (r.ok ? r.json() : Promise.reject(new Error(r.status))))
        .then(build)
        .catch(() => {
            status.textContent = "The map could not be loaded.";
        });

    function build(data) {
        const { chapters, pages } = data;
        const incoming = pages.map(() => []);
        pages.forEach((p, i) => p.o.forEach((t) => incoming[t].push(i)));
        const base = api.resolve("");
        const href = (p) => base + (p.u ? p.u + "/" : "");
        const root = document.createElement("div");
        root.className = "dl-map";
        const legend = document.createElement("div");
        legend.className = "dl-widget__legend";
        for (const [t, col] of [["selected", "var(--md-default-fg-color)"], ["it links to", "var(--md-accent-fg-color)"], ["links to it", "#4a90d9"]]) {
            const s = document.createElement("span");
            s.style.setProperty("--swatch", col);
            s.textContent = t;
            legend.appendChild(s);
        }
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
                b.setAttribute("aria-pressed", "false");
                b.addEventListener("click", () => pick(picked === i ? -1 : i));
                chips[i] = b;
                row.appendChild(b);
            }
            root.append(h, row);
        });
        const detail = document.createElement("div");
        detail.className = "dl-map__detail";
        detail.setAttribute("aria-live", "polite");
        status.remove();
        el.append(legend, root, detail);

        let picked = -1;
        function links(list) {
            const span = document.createElement("span");
            if (!list.length) span.textContent = "none";
            for (const i of list) {
                const a = document.createElement("a");
                a.href = href(pages[i]);
                a.textContent = pages[i].t;
                span.appendChild(a);
            }
            return span;
        }
        function pick(i) {
            picked = i;
            root.classList.toggle("dl-map--picked", i >= 0);
            chips.forEach((b, k) => {
                if (!b) return;
                b.removeAttribute("data-role");
                b.setAttribute("aria-pressed", String(k === i));
            });
            detail.innerHTML = "";
            if (i < 0) {
                detail.textContent = `${pages.length} pages in ${chapters.length} chapters. Select a page to see what it links to and what links to it.`;
                return;
            }
            const out = pages[i].o, inn = incoming[i];
            for (const k of out) chips[k].dataset.role = "out";
            for (const k of inn) chips[k].dataset.role = chips[k].dataset.role === "out" ? "both" : "in";
            chips[i].dataset.role = "self";
            const head = document.createElement("p");
            const open = document.createElement("a");
            open.href = href(pages[i]);
            open.textContent = pages[i].t;
            const strong = document.createElement("strong");
            strong.appendChild(open);
            head.append(strong, document.createTextNode(` (${chapters[pages[i].c]})`));
            const row = (label, list) => {
                const p = document.createElement("p");
                p.append(document.createTextNode(label + " "), links(list));
                return p;
            };
            detail.append(head, row("Links to:", out), row("Linked from:", inn));
        }
        pick(-1);
    }
});
