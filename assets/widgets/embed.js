/*
 * Click-to-load embed, for heavy pages such as marimo WASM playgrounds.
 *
 *     <div class="dl-widget" data-widget="embed"
 *          data-src="assets/playgrounds/ar/index.html"
 *          data-height="640"
 *          data-label="Start the AR playground"
 *          data-note="Runs Python in your browser; first start downloads ~25 MB." markdown>
 *     How to get the demo without this page (e.g. run it locally).
 *     </div>
 *
 * data-src is relative to the docs root and resolved against the site base.
 *
 * Embedded pages can be generated at build time (scripts/build_playgrounds.sh)
 * and are not part of every build, so the start button only replaces the
 * Markdown fallback once the page is known to exist.
 */
dlWidgets.register("embed", (el, api) => {
    const src = api.resolve(el.dataset.src);
    const height = parseInt(el.dataset.height || "600", 10);
    const label = el.dataset.label || "Load interactive demo";
    const note = el.dataset.note || "";

    function showStart() {
        el.classList.add("dl-embed");
        el.innerHTML = "";

        const start = document.createElement("div");
        start.className = "dl-embed__start";
        start.style.minHeight = Math.min(height, 320) + "px";

        const btn = document.createElement("button");
        btn.type = "button";
        btn.className = "md-button md-button--primary";
        btn.textContent = label;

        const small = document.createElement("div");
        small.textContent = note;

        const open = document.createElement("a");
        open.href = src;
        open.target = "_blank";
        open.rel = "noopener";
        open.textContent = "or open it in a new tab";

        start.append(btn, small, open);
        el.appendChild(start);

        btn.addEventListener("click", () => {
            const frame = document.createElement("iframe");
            frame.src = src;
            frame.title = label;
            frame.loading = "lazy";
            frame.style.height = height + "px";
            frame.setAttribute("allow", "clipboard-write");
            el.replaceChildren(frame);
        });
    }

    // On a missing page or a network error, keep the fallback.
    fetch(src, { method: "HEAD", cache: "no-store" })
        .then((res) => {
            if (res.ok && el.isConnected) showStart();
        })
        .catch(() => {});
});
