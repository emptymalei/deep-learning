/*
 * Shared helpers for the SVG explainers: normal-distribution math, a seeded
 * random generator, small DOM builders for sliders, button groups and
 * legends, and the resize/theme lifecycle. Load after loader.js and before
 * the widgets that use it (prob-eval.js, significance.js).
 */
window.dlWidgetKit = (function () {
    const NS = "http://www.w3.org/2000/svg";
    const SQRT2PI = Math.sqrt(2 * Math.PI);
    const SQRTPI = Math.sqrt(Math.PI);

    // ---- Math --------------------------------------------------------------
    const phi = (z) => Math.exp(-0.5 * z * z) / SQRT2PI;
    function erf(x) {
        // Abramowitz & Stegun 7.1.26
        const s = Math.sign(x);
        x = Math.abs(x);
        const t = 1 / (1 + 0.3275911 * x);
        const y = 1 - ((((1.061405429 * t - 1.453152027) * t + 1.421413741) * t - 0.284496736) * t + 0.254829592) * t * Math.exp(-x * x);
        return s * y;
    }
    const Phi = (z) => 0.5 * (1 + erf(z / Math.SQRT2));
    function bisect(f, target, lo, hi) {
        for (let i = 0; i < 48; i++) {
            const mid = 0.5 * (lo + hi);
            if (f(mid) < target) lo = mid;
            else hi = mid;
        }
        return 0.5 * (lo + hi);
    }
    const PhiInv = (p) => bisect(Phi, p, -8, 8);
    /** E|Z| for Z ~ N(mu, var). */
    function absMean(mu, v) {
        const sd = Math.sqrt(v);
        if (sd < 1e-12) return Math.abs(mu);
        const z = mu / sd;
        return mu * (2 * Phi(z) - 1) + 2 * sd * phi(z);
    }
    function rng(seed) {
        let a = seed >>> 0;
        const u = () => {
            a = (a + 0x6d2b79f5) >>> 0;
            let t = a;
            t = Math.imul(t ^ (t >>> 15), t | 1);
            t ^= t + Math.imul(t ^ (t >>> 7), t | 61);
            return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
        };
        const normal = () => Math.sqrt(-2 * Math.log(1 - u())) * Math.cos(2 * Math.PI * u());
        return { u, normal };
    }
    // A forecast is a mixture of normals: [[weight, mean, sd], ...].
    const mixCdf = (mix, x) => mix.reduce((a, [w, m, s]) => a + w * Phi((x - m) / s), 0);
    const mixPdf = (mix, x) => mix.reduce((a, [w, m, s]) => a + (w * phi((x - m) / s)) / s, 0);
    const mixQuantile = (mix, p) => bisect((x) => mixCdf(mix, x), p, -30, 30);
    /** CRPS = E|X - x| - E|X - X'| / 2, lower is better. */
    function mixCrps(mix, x) {
        let a = 0;
        let b = 0;
        for (const [w, m, s] of mix) {
            a += w * absMean(m - x, s * s);
            for (const [w2, m2, s2] of mix) b += w * w2 * absMean(m - m2, s * s + s2 * s2);
        }
        return a - 0.5 * b;
    }

    // ---- DOM helpers ---------------------------------------------------------
    function node(name, attrs, parent, text) {
        const n = document.createElementNS(NS, name);
        for (const k in attrs) n.setAttribute(k, attrs[k]);
        if (text !== undefined) n.textContent = text;
        if (parent) parent.appendChild(n);
        return n;
    }
    function html(tag, cls, parent, text) {
        const n = document.createElement(tag);
        if (cls) n.className = cls;
        if (text !== undefined) n.textContent = text;
        if (parent) parent.appendChild(n);
        return n;
    }
    function slider(parent, label, min, max, step, value, digits, onInput) {
        const wrap = html("label", "", parent);
        html("span", "", wrap, label);
        const input = html("input", "", wrap);
        const out = html("output", "", wrap);
        Object.assign(input, { type: "range", min, max, step, value });
        const show = (v) => (out.textContent = Number(v).toFixed(digits));
        show(value);
        input.addEventListener("input", () => {
            show(input.value);
            onInput(parseFloat(input.value));
        });
        return {
            set(v) {
                input.value = v;
                show(v);
            },
        };
    }
    function seg(parent, label, names, onPick) {
        const box = html("div", "dl-widget__seg", parent);
        box.setAttribute("role", "group");
        box.setAttribute("aria-label", label);
        box.style.flexWrap = "wrap";
        const buttons = names.map((name, i) => {
            const b = html("button", "", box, name);
            b.type = "button";
            b.addEventListener("click", () => onPick(i));
            return b;
        });
        return {
            select(i) {
                buttons.forEach((b, j) => b.setAttribute("aria-pressed", String(i === j)));
            },
        };
    }
    function button(parent, label, onClick) {
        const b = html("button", "dl-widget__btn", parent, label);
        b.type = "button";
        b.addEventListener("click", onClick);
        return b;
    }
    function legend(box, items) {
        box.innerHTML = "";
        for (const [label, col] of items) html("span", "", box, label).style.setProperty("--swatch", col);
    }
    function widthOf(el) {
        return Math.max(300, Math.min(720, Math.round(el.clientWidth - 2) || 640));
    }
    /** Redraw on resize and theme change; returns the cleanup function. */
    function lifecycle(el, api, render, stop) {
        let last = el.clientWidth;
        const ro = "ResizeObserver" in window
            ? new ResizeObserver(() => {
                  if (Math.abs(el.clientWidth - last) > 8) {
                      last = el.clientWidth;
                      render();
                  }
              })
            : null;
        if (ro) ro.observe(el);
        const offTheme = api.onThemeChange(render);
        return () => {
            if (stop) stop();
            if (ro) ro.disconnect();
            offTheme();
        };
    }
    const polyline = (pts) => pts.map((p) => `${p[0].toFixed(1)},${p[1].toFixed(1)}`).join(" ");
    const range = (a, b, n) => Array.from({ length: n + 1 }, (_, i) => a + ((b - a) * i) / n);

    return {
        NS, SQRT2PI, SQRTPI, phi, erf, Phi, bisect, PhiInv, absMean, rng,
        mixCdf, mixPdf, mixQuantile, mixCrps,
        node, html, slider, seg, button, legend, widthOf, lifecycle, polyline, range,
    };
})();
