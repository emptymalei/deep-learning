/*
 * Widget loader for interactive explainers.
 *
 * Usage in Markdown (md_in_html is enabled):
 *
 *     <div class="dl-widget" data-widget="backtest-windows" markdown>
 *     ![Static fallback](../assets/some-figure.png)
 *     </div>
 *
 * Whatever is inside the div is the fallback (no JS, print, llms.txt export).
 * When the widget mounts, it replaces that content.
 *
 * A widget registers itself from its own script file:
 *
 *     dlWidgets.register("backtest-windows", (el, api) => {
 *         ...build DOM inside el...
 *         return () => { ...optional cleanup... };
 *     });
 *
 * The loader works with Material for MkDocs and Zensical: it mounts on the
 * `document$` observable when present (instant navigation) and falls back to
 * DOMContentLoaded otherwise. It uses no MkDocs plugin.
 */
(function () {
    "use strict";

    const registry = new Map();
    const mounted = new WeakSet();
    const themeListeners = new Set();
    const scriptCache = new Map();

    function cssVar(name, fallback) {
        const v = getComputedStyle(document.body || document.documentElement)
            .getPropertyValue(name)
            .trim();
        return v || fallback;
    }

    /** Colors resolved from the active Material/Zensical palette. */
    function colors() {
        return {
            fg: cssVar("--md-default-fg-color", "rgba(0,0,0,.87)"),
            fgLight: cssVar("--md-default-fg-color--light", "rgba(0,0,0,.54)"),
            fgLighter: cssVar("--md-default-fg-color--lighter", "rgba(0,0,0,.32)"),
            fgLightest: cssVar("--md-default-fg-color--lightest", "rgba(0,0,0,.07)"),
            bg: cssVar("--md-default-bg-color", "#fff"),
            codeBg: cssVar("--md-code-bg-color", "#f5f5f5"),
            primary: cssVar("--md-primary-fg-color", "#000"),
            accent: cssVar("--md-accent-fg-color", "#ff6e42"),
            // Categorical colors chosen to stay legible on both light and dark.
            train: "#4a90d9",
            test: cssVar("--md-accent-fg-color", "#ff6e42"),
            gap: cssVar("--md-default-fg-color--lighter", "rgba(0,0,0,.32)"),
        };
    }

    function reducedMotion() {
        return window.matchMedia("(prefers-reduced-motion: reduce)").matches;
    }

    /** Site root URL, read from the theme's #__config block when available. */
    function siteBase() {
        const cfg = document.getElementById("__config");
        if (cfg) {
            try {
                const base = JSON.parse(cfg.textContent).base;
                if (typeof base === "string") {
                    return new URL(base.replace(/\/?$/, "/"), location.href);
                }
            } catch (e) {
                /* fall through */
            }
        }
        return new URL("./", location.href);
    }

    /** Resolve a path written relative to the docs root (e.g. "assets/x.json"). */
    function resolve(path) {
        if (/^(https?:)?\/\//.test(path)) return path;
        return new URL(path.replace(/^\//, ""), siteBase()).href;
    }

    /** Load a classic script once (for CDN libraries such as vega or ort). */
    function loadScript(src) {
        if (!scriptCache.has(src)) {
            scriptCache.set(
                src,
                new Promise((ok, fail) => {
                    const s = document.createElement("script");
                    s.src = src;
                    s.async = true;
                    s.crossOrigin = "anonymous";
                    s.onload = () => ok();
                    s.onerror = () => fail(new Error("Failed to load " + src));
                    document.head.appendChild(s);
                })
            );
        }
        return scriptCache.get(src);
    }

    /** Call cb(colors) whenever the light/dark palette changes. */
    function onThemeChange(cb) {
        themeListeners.add(cb);
        return () => themeListeners.delete(cb);
    }

    const api = { colors, reducedMotion, resolve, loadScript, onThemeChange };

    function mountOne(el) {
        if (mounted.has(el)) return;
        const fn = registry.get(el.dataset.widget);
        if (!fn) return; // mounts later, when its script registers
        mounted.add(el);
        el.classList.add("dl-widget--live");
        try {
            const cleanup = fn(el, api);
            if (typeof cleanup === "function") el._dlCleanup = cleanup;
        } catch (err) {
            console.error("[dl-widget] " + el.dataset.widget, err);
            el.classList.remove("dl-widget--live");
            el.classList.add("dl-widget--error");
        }
    }

    function mountAll() {
        document
            .querySelectorAll(".dl-widget[data-widget]")
            .forEach(mountOne);
    }

    function register(name, fn) {
        registry.set(name, fn);
        if (document.readyState !== "loading") mountAll();
    }

    // Re-mount on page changes (instant navigation) or once on load.
    if (window.document$ && typeof window.document$.subscribe === "function") {
        window.document$.subscribe(() => mountAll());
    } else if (document.readyState === "loading") {
        document.addEventListener("DOMContentLoaded", mountAll);
    } else {
        mountAll();
    }

    // Material/Zensical switch palettes by changing body[data-md-color-scheme].
    const notify = () => {
        const c = colors();
        themeListeners.forEach((cb) => {
            try {
                cb(c);
            } catch (e) {
                console.error(e);
            }
        });
    };
    const startObserver = () =>
        new MutationObserver(notify).observe(document.body, {
            attributes: true,
            attributeFilter: ["data-md-color-scheme", "data-md-color-primary", "data-md-color-accent"],
        });
    if (document.body) startObserver();
    else document.addEventListener("DOMContentLoaded", startObserver);

    window.dlWidgets = Object.assign(window.dlWidgets || {}, { register }, api);
})();
