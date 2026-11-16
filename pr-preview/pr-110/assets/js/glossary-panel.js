/*
 * Glossary panel. Glossary terms (includes/glossary.md) become <abbr title>
 * elements, and a title only shows on hover: on a phone there is no hover,
 * and a tap gives at most a tooltip that is gone at once.
 *
 * So a tap or click on a term opens its definition in a panel that stays
 * until the reader closes it: a sheet along the bottom of the screen on
 * phones and tablets, a card on the right on wider screens. Tapping another
 * term swaps the content; the close button, Escape, or the same term again
 * closes it.
 *
 * The theme's own tooltips (content.tooltips) stay for hover: on devices that
 * can hover they work as before, and are only hidden while the panel is open,
 * so a definition never shows twice. On touch devices the theme's tooltip is
 * what flashes on a tap, so there the terms are detached from it.
 *
 * The glossary list in the sidebar (built by scripts/stage_site.py) opens the
 * same panel. Terms inside links keep working as links. The styles live here rather than
 * in a stylesheet, so a cached older stylesheet can never leave the panel
 * unstyled.
 */
(function () {
    const CSS = `
.md-typeset abbr[data-dl-term] { cursor: pointer; border-radius: .15rem; }
.md-typeset abbr[data-dl-term]:focus-visible { outline: 2px solid var(--md-accent-fg-color); outline-offset: 1px; }
.md-typeset abbr[data-dl-term][aria-expanded="true"] { background: var(--md-accent-fg-color--transparent, rgba(255, 110, 66, .15)); }
.dl-glossary { position: fixed; z-index: 4; box-sizing: border-box; overflow: auto;
  background: var(--md-default-bg-color); color: var(--md-default-fg-color);
  border: 1px solid var(--md-default-fg-color--lighter); box-shadow: 0 .2rem 1rem rgba(0, 0, 0, .25);
  font-size: .75rem; line-height: 1.55; visibility: hidden;
  transition: transform .2s ease, visibility 0s linear .2s; }
.dl-glossary--open { visibility: visible; transition: transform .2s ease, visibility 0s; }
.dl-glossary__head { display: flex; align-items: center; justify-content: space-between; gap: .5rem; }
.dl-glossary__kicker { font-size: .6rem; font-weight: 700; letter-spacing: .06em; text-transform: uppercase; color: var(--md-default-fg-color--light); }
.dl-glossary__close { flex: none; width: 2.2rem; height: 2.2rem; margin: -.4rem -.5rem -.4rem 0; border: 0; border-radius: 50%;
  background: transparent; color: var(--md-default-fg-color--light); font: inherit; font-size: 1.3rem; line-height: 1; cursor: pointer; }
.dl-glossary__close:hover, .dl-glossary__close:focus-visible { background: var(--md-default-fg-color--lightest); color: var(--md-default-fg-color); }
.dl-glossary__term { margin: .2rem 0 .25rem; font-size: 1rem; font-weight: 700; }
.dl-glossary__text { margin: 0; }
@media (max-width: 59.9844em) {
  .dl-glossary { left: 0; right: 0; bottom: 0; max-height: 60vh; border-width: 1px 0 0; border-radius: .8rem .8rem 0 0;
    padding: .7rem 1rem calc(1rem + env(safe-area-inset-bottom, 0px)); transform: translateY(100%); }
}
@media (min-width: 60em) {
  .dl-glossary { top: 5.6rem; right: .6rem; width: 13.6rem; max-height: calc(100vh - 7rem); border-radius: .5rem;
    padding: .7rem .9rem 1rem; transform: translateX(calc(100% + 1.2rem)); }
}
.dl-glossary.dl-glossary--open { transform: none; }
.dl-glossary-on .md-tooltip2 { display: none !important; }
@media (prefers-reduced-motion: reduce) { .dl-glossary, .dl-glossary--open { transition: none; } }
@media print { .dl-glossary { display: none; } }`;

    let panel = null;
    let termEl = null;
    let textEl = null;
    let closeBtn = null;
    let active = null;

    function build() {
        if (panel) return;
        const style = document.createElement("style");
        style.id = "dl-glossary-style";
        style.textContent = CSS;
        document.head.appendChild(style);

        panel = document.createElement("aside");
        panel.className = "dl-glossary";
        panel.id = "dl-glossary";
        panel.setAttribute("role", "dialog");
        panel.setAttribute("aria-label", "Glossary");
        panel.setAttribute("aria-hidden", "true");

        const head = document.createElement("div");
        head.className = "dl-glossary__head";
        const kicker = document.createElement("span");
        kicker.className = "dl-glossary__kicker";
        kicker.textContent = "Glossary";
        closeBtn = document.createElement("button");
        closeBtn.type = "button";
        closeBtn.className = "dl-glossary__close";
        closeBtn.setAttribute("aria-label", "Close the glossary panel");
        closeBtn.textContent = "×";
        closeBtn.addEventListener("click", () => {
            close(false);
            closeBtn.blur();
        });
        head.append(kicker, closeBtn);

        termEl = document.createElement("p");
        termEl.className = "dl-glossary__term";
        // Read out the definition when it changes.
        textEl = document.createElement("p");
        textEl.className = "dl-glossary__text";
        textEl.setAttribute("aria-live", "polite");
        panel.append(head, termEl, textEl);
        document.body.appendChild(panel);
    }

    function open(abbr) {
        build();
        if (active) active.setAttribute("aria-expanded", "false");
        active = abbr;
        abbr.setAttribute("aria-expanded", "true");
        termEl.textContent = abbr.textContent;
        textEl.textContent = abbr.dataset.dlTerm;
        panel.scrollTop = 0;
        panel.setAttribute("aria-hidden", "false");
        panel.classList.add("dl-glossary--open");
        document.body.classList.add("dl-glossary-on");
    }

    function close(returnFocus) {
        if (!panel || !active) return;
        const was = active;
        active = null;
        was.setAttribute("aria-expanded", "false");
        panel.setAttribute("aria-hidden", "true");
        panel.classList.remove("dl-glossary--open");
        document.body.classList.remove("dl-glossary-on");
        if (returnFocus && document.contains(was)) was.focus({ preventScroll: true });
    }

    function toggle(abbr) {
        if (abbr === active) close(false);
        else open(abbr);
    }

    // Make the terms of the current page tappable and keyboard-reachable.
    function prepare() {
        close(false);
        const canHover = window.matchMedia("(hover: hover)").matches;
        document.querySelectorAll(".md-typeset abbr[title]").forEach((abbr) => {
            if (abbr.closest("a, button")) return; // the link keeps its job
            if (!canHover) {
                // A copy carries none of the theme's tooltip listeners.
                const copy = abbr.cloneNode(true);
                abbr.replaceWith(copy);
                abbr = copy;
            }
            abbr.dataset.dlTerm = abbr.getAttribute("title");
            abbr.setAttribute("role", "button");
            abbr.setAttribute("tabindex", "0");
            abbr.setAttribute("aria-controls", "dl-glossary");
            abbr.setAttribute("aria-expanded", "false");
            if (!canHover) abbr.removeAttribute("title");
        });
    }

    // Terms in the text (<abbr>) and in the sidebar's glossary list (<button>).
    const termOf = (target) => (target instanceof Element ? target.closest("[data-dl-term]") : null);

    document.addEventListener("click", (event) => {
        const abbr = termOf(event.target);
        if (!abbr) return;
        event.preventDefault();
        toggle(abbr);
    });
    document.addEventListener("keydown", (event) => {
        if (event.key === "Escape") {
            close(true);
            return;
        }
        if (event.key !== "Enter" && event.key !== " ") return;
        const abbr = termOf(event.target);
        // Buttons already turn Enter and Space into a click.
        if (!abbr || event.target !== abbr || abbr.tagName !== "ABBR") return;
        event.preventDefault();
        toggle(abbr);
    });

    if (window.document$ && typeof window.document$.subscribe === "function") {
        window.document$.subscribe(prepare);
    } else if (document.readyState === "loading") {
        document.addEventListener("DOMContentLoaded", prepare);
    } else {
        prepare();
    }
})();
