/*
 * Page actions for reading with an AI assistant: copy the page as Markdown,
 * or open Claude / ChatGPT with a prompt that points at the page's Markdown.
 *
 * The Markdown comes from the llms.txt build (the llmstxt plugin in
 * mkdocs.yml writes <page>/index.md next to every page's index.html).
 * Nothing is sent anywhere until the reader clicks.
 */
(function () {
    const SITE = "Time Series with Deep Learning Quick Bite";

    function markdownUrl() {
        const u = new URL(location.href);
        u.hash = "";
        u.search = "";
        u.pathname = u.pathname.replace(/[^/]*$/, "") + "index.md";
        return u.toString();
    }

    function prompt(url) {
        return (
            `Read ${url}, a page from the book "${SITE}", ` +
            "and help me understand it. I will ask follow-up questions."
        );
    }

    function flash(el, text) {
        const old = el.textContent;
        el.textContent = text;
        setTimeout(() => (el.textContent = old), 1600);
    }

    async function copyMarkdown(url, button) {
        let missing = false;
        const text = () =>
            fetch(url).then((res) => {
                if (!res.ok) {
                    missing = true;
                    throw new Error(String(res.status));
                }
                return res.text();
            });
        try {
            if (window.ClipboardItem && navigator.clipboard.write) {
                // Passing a promise keeps the click's user activation (Safari).
                const blob = text().then((t) => new Blob([t], { type: "text/plain" }));
                await navigator.clipboard.write([new ClipboardItem({ "text/plain": blob })]);
            } else {
                await navigator.clipboard.writeText(await text());
            }
            flash(button, "Copied");
        } catch (err) {
            flash(button, missing ? "Not available" : "Copy failed");
        }
    }

    function link(label, href, title) {
        const a = document.createElement("a");
        a.textContent = label;
        a.href = href;
        a.title = title;
        a.target = "_blank";
        a.rel = "noopener";
        return a;
    }

    function mount() {
        const h1 = document.querySelector(".md-content__inner h1");
        // Only on real pages: the 404 page has no canonical URL.
        if (!h1 || !document.querySelector('link[rel="canonical"]')) return;
        if (h1.nextElementSibling && h1.nextElementSibling.classList.contains("dl-page-actions")) return;

        const url = markdownUrl();
        const bar = document.createElement("div");
        bar.className = "dl-page-actions";

        const copy = document.createElement("button");
        copy.type = "button";
        copy.textContent = "Copy as Markdown";
        copy.title = "Copy this page as Markdown, e.g. to paste into an AI assistant";
        copy.addEventListener("click", () => copyMarkdown(url, copy));

        bar.append(
            copy,
            link("Ask Claude", "https://claude.ai/new?q=" + encodeURIComponent(prompt(url)), "Open Claude with a prompt about this page"),
            link("Ask ChatGPT", "https://chatgpt.com/?q=" + encodeURIComponent(prompt(url)), "Open ChatGPT with a prompt about this page"),
            link("Markdown", url, "This page as Markdown")
        );
        h1.insertAdjacentElement("afterend", bar);
    }

    if (window.document$ && typeof window.document$.subscribe === "function") {
        window.document$.subscribe(mount);
    } else if (document.readyState === "loading") {
        document.addEventListener("DOMContentLoaded", mount);
    } else {
        mount();
    }
})();
