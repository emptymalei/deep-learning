/*
 * "Ask an AI assistant about this page": one small icon next to the page's
 * edit / view / copy icons, opening a menu with Claude, ChatGPT and Gemini.
 * Each opens in a new tab with a prompt that points at the page's Markdown
 * copy (written by the llmstxt plugin as <page>/index.md); the assistant
 * reads that URL with its own web access. Copying the page as Markdown is
 * Zensical's own `content.action.copy` button.
 *
 * Selecting text on a page brings up the same three assistants in a small
 * bar at the bottom of the screen, with a prompt that quotes the selection.
 *
 * Nothing is sent anywhere until the reader picks an assistant. The styles
 * live here rather than in a stylesheet, so a cached older stylesheet can
 * never leave the menu unstyled.
 */
(function () {
    const SITE = "Time Series with Deep Learning Quick Bite";
    const ICON =
        '<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 24 24" aria-hidden="true">' +
        '<path d="M10 2.5l1.6 4.9 4.9 1.6-4.9 1.6L10 15.5l-1.6-4.9L3.5 9l4.9-1.6zM18 13l.9 2.6 2.6.9-2.6.9L18 20l-.9-2.6-2.6-.9 2.6-.9z"/></svg>';
    const CSS = `
.dl-ai { position: relative; }
[dir=ltr] .dl-ai { float: right; margin-left: .4rem; }
[dir=rtl] .dl-ai { float: left; margin-right: .4rem; }
.md-typeset .dl-ai > .md-content__button.dl-ai__toggle { float: none; margin: .2rem 0 0; border: 0; cursor: pointer; }
.dl-ai__menu { position: absolute; top: calc(100% + .3rem); z-index: 3; min-width: 8.5rem; padding: .25rem 0;
  background: var(--md-default-bg-color); border: 1px solid var(--md-default-fg-color--lightest);
  border-radius: .4rem; box-shadow: var(--md-shadow-z2); font-size: .7rem; line-height: 1.4; }
[dir=ltr] .dl-ai__menu { right: 0; }
[dir=rtl] .dl-ai__menu { left: 0; }
.dl-ai__menu[hidden] { display: none; }
.md-typeset .dl-ai__menu a { display: block; padding: .35rem .8rem; color: var(--md-default-fg-color); text-decoration: none; }
.md-typeset .dl-ai__menu a:hover, .md-typeset .dl-ai__menu a:focus-visible { background: var(--md-default-fg-color--lightest); }
.md-typeset .dl-ai__menu .dl-ai__note { margin: .25rem .8rem .2rem; max-width: 10rem; font-size: .6rem; line-height: 1.35; color: var(--md-default-fg-color--light); }
.dl-ask { position: fixed; left: 50%; bottom: .8rem; z-index: 5; transform: translateX(-50%); display: flex; align-items: center; gap: .1rem;
  max-width: calc(100vw - 1.2rem); padding: .25rem .4rem .25rem .8rem; border-radius: 2rem; white-space: nowrap;
  background: var(--md-default-bg-color); color: var(--md-default-fg-color); border: 1px solid var(--md-default-fg-color--lighter);
  box-shadow: 0 .2rem .8rem rgba(0, 0, 0, .25); font-size: .7rem; }
.dl-ask[hidden] { display: none; }
.dl-ask span { color: var(--md-default-fg-color--light); margin-right: .2rem; }
.dl-ask a { padding: .2rem .5rem; border-radius: 1rem; color: var(--md-accent-fg-color); text-decoration: none; font-weight: 600; }
.dl-ask a:hover, .dl-ask a:focus-visible { background: var(--md-default-fg-color--lightest); }
.dl-ask button { width: 1.7rem; height: 1.7rem; border: 0; border-radius: 50%; background: transparent; color: var(--md-default-fg-color--light); font: inherit; font-size: 1rem; cursor: pointer; }
@media print { .dl-ai, .dl-ask { display: none; } }`;

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

    function addStyles() {
        if (document.getElementById("dl-ai-style")) return;
        const style = document.createElement("style");
        style.id = "dl-ai-style";
        style.textContent = CSS;
        document.head.appendChild(style);
    }

    function mount() {
        const inner = document.querySelector(".md-content__inner");
        const h1 = inner && inner.querySelector("h1");
        // Only on real pages: the 404 page has no canonical URL.
        if (!h1 || !document.querySelector('link[rel="canonical"]')) return;
        if (inner.querySelector(".dl-ai")) return;
        addStyles();

        const url = markdownUrl();
        const wrap = document.createElement("div");
        wrap.className = "dl-ai";

        const toggle = document.createElement("button");
        toggle.type = "button";
        toggle.className = "md-content__button md-icon dl-ai__toggle";
        toggle.title = "Ask an AI assistant about this page";
        toggle.setAttribute("aria-label", toggle.title);
        toggle.setAttribute("aria-haspopup", "true");
        toggle.setAttribute("aria-expanded", "false");
        toggle.innerHTML = ICON;

        const menu = document.createElement("div");
        menu.className = "dl-ai__menu";
        menu.hidden = true;
        const text = prompt(url);
        // The ?q= prefill only works when the reader is signed in, and Gemini
        // has no documented prefill at all, so every link also copies the
        // prompt: if the box comes up empty, the reader pastes it.
        for (const [label, base] of [
            ["Ask Claude", "https://claude.ai/new?q="],
            ["Ask ChatGPT", "https://chatgpt.com/?q="],
            ["Ask Gemini", "https://gemini.google.com/app?q="],
        ]) {
            const a = document.createElement("a");
            a.textContent = label;
            a.href = base + encodeURIComponent(text);
            a.target = "_blank";
            a.rel = "noopener";
            a.addEventListener("click", () => {
                if (navigator.clipboard) navigator.clipboard.writeText(text).catch(() => {});
                close();
            });
            menu.appendChild(a);
        }
        const note = document.createElement("p");
        note.className = "dl-ai__note";
        note.textContent = "The prompt is also copied, so you can paste it if the box is empty.";
        menu.appendChild(note);

        function close() {
            menu.hidden = true;
            toggle.setAttribute("aria-expanded", "false");
        }
        toggle.addEventListener("click", (event) => {
            event.stopPropagation();
            menu.hidden = !menu.hidden;
            toggle.setAttribute("aria-expanded", String(!menu.hidden));
        });
        document.addEventListener("click", (event) => {
            if (!wrap.contains(event.target)) close();
        });
        document.addEventListener("keydown", (event) => {
            if (event.key === "Escape") close();
        });

        wrap.append(toggle, menu);
        // Sit with the theme's icons (edit, view, copy), to their left.
        const buttons = inner.querySelectorAll(":scope > .md-content__button");
        if (buttons.length) {
            buttons[buttons.length - 1].insertAdjacentElement("afterend", wrap);
        } else {
            h1.insertAdjacentElement("beforebegin", wrap);
        }
    }

    // ---- Ask about the selected text ------------------------------------------
    const ASSISTANTS = [
        ["Claude", "https://claude.ai/new?q="],
        ["ChatGPT", "https://chatgpt.com/?q="],
        ["Gemini", "https://gemini.google.com/app?q="],
    ];
    const MAX_QUOTE = 1200;
    let bar = null;
    let dismissed = "";

    function selectedText() {
        const sel = window.getSelection();
        if (!sel || sel.isCollapsed || !sel.rangeCount) return "";
        const article = document.querySelector(".md-content__inner");
        const inside = (n) => n && article && article.contains(n.nodeType === 1 ? n : n.parentNode);
        if (!inside(sel.anchorNode) || !inside(sel.focusNode)) return "";
        // Not while the reader is working inside a widget or a form field.
        const host = (sel.anchorNode.nodeType === 1 ? sel.anchorNode : sel.anchorNode.parentElement);
        if (host && host.closest(".dl-widget, input, textarea")) return "";
        return sel.toString().replace(/\s+/g, " ").trim();
    }

    function selectionPrompt(quote) {
        const cut = quote.length > MAX_QUOTE ? quote.slice(0, MAX_QUOTE) + "…" : quote;
        return (
            `I am reading ${markdownUrl()}, a page from the book "${SITE}". ` +
            `Read the page, then explain this passage from it:\n\n"${cut}"`
        );
    }

    function updateBar() {
        const quote = selectedText();
        if (quote.length < 3 || quote === dismissed) {
            if (bar) bar.hidden = true;
            if (quote.length < 3) dismissed = "";
            return;
        }
        addStyles();
        if (!bar) {
            bar = document.createElement("div");
            bar.className = "dl-ask";
            bar.setAttribute("role", "group");
            bar.setAttribute("aria-label", "Ask an AI assistant about the selected text");
            const label = document.createElement("span");
            label.textContent = "Ask AI:";
            bar.appendChild(label);
            for (const [name] of ASSISTANTS) {
                const a = document.createElement("a");
                a.textContent = name;
                a.target = "_blank";
                a.rel = "noopener";
                a.addEventListener("click", () => {
                    if (navigator.clipboard) navigator.clipboard.writeText(a.dataset.prompt).catch(() => {});
                });
                bar.appendChild(a);
            }
            const close = document.createElement("button");
            close.type = "button";
            close.textContent = "×";
            close.setAttribute("aria-label", "Hide");
            close.addEventListener("click", () => {
                dismissed = selectedText();
                bar.hidden = true;
            });
            // Keep the selection when the bar itself is pressed.
            bar.addEventListener("mousedown", (event) => event.preventDefault());
            bar.appendChild(close);
            document.body.appendChild(bar);
        }
        const text = selectionPrompt(quote);
        [...bar.querySelectorAll("a")].forEach((a, i) => {
            a.href = ASSISTANTS[i][1] + encodeURIComponent(text);
            a.dataset.prompt = text;
        });
        bar.hidden = false;
    }

    let timer = 0;
    document.addEventListener("selectionchange", () => {
        clearTimeout(timer);
        timer = setTimeout(updateBar, 250);
    });

    if (window.document$ && typeof window.document$.subscribe === "function") {
        window.document$.subscribe(mount);
    } else if (document.readyState === "loading") {
        document.addEventListener("DOMContentLoaded", mount);
    } else {
        mount();
    }
})();
