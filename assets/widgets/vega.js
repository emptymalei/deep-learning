/*
 * Render a Vega-Lite spec exported from Python (Altair), themed to the site.
 *
 *     <div class="dl-widget" data-widget="vega"
 *          data-spec="time-series/assets/timeseries-transformations/box-cox-lambda.vl.json" markdown>
 *     ![Static fallback](assets/timeseries-transformations/....png)
 *     </div>
 *
 * data-spec is relative to the docs root. Specs come from scripts/build_charts.py.
 * The Vega libraries load lazily from jsDelivr, only on pages that use them.
 */
(function () {
    const CDN = "https://cdn.jsdelivr.net/npm/";
    const LIBS = [
        "vega@6.4.0/build/vega.min.js",
        "vega-lite@6.5.0/build/vega-lite.min.js",
        "vega-embed@7.3.0/build/vega-embed.min.js",
    ];

    let libs = null;
    function loadLibs(api) {
        // Sequential: vega-lite and vega-embed expect the earlier globals.
        if (!libs) {
            libs = LIBS.reduce((p, lib) => p.then(() => api.loadScript(CDN + lib)), Promise.resolve());
        }
        return libs;
    }

    function themeConfig(c) {
        const text = { labelColor: c.fgLight, titleColor: c.fgLight };
        return {
            background: null,
            font: getComputedStyle(document.body).fontFamily,
            view: { stroke: null },
            axis: Object.assign(
                { gridColor: c.fgLightest, domainColor: c.fgLighter, tickColor: c.fgLighter },
                text
            ),
            legend: text,
            title: { color: c.fg, fontWeight: 600, anchor: "start", fontSize: 12 },
            line: { color: c.accent },
            rule: { color: c.fgLight },
            range: { category: [c.accent, c.train, c.fgLight] },
        };
    }

    dlWidgets.register("vega", (el, api) => {
        const specUrl = api.resolve(el.dataset.spec);
        const fallback = el.innerHTML;
        let view = null;
        let spec = null;

        async function draw() {
            await loadLibs(api);
            if (!spec) spec = await (await fetch(specUrl)).json();
            const state = view ? view.getState({ data: () => false }) : null;
            if (view) view.finalize();
            const target = document.createElement("div");
            target.style.width = "100%";
            el.replaceChildren(target);
            const result = await window.vegaEmbed(target, spec, {
                actions: false,
                renderer: "svg",
                config: themeConfig(api.colors()),
            });
            view = result.view;
            // Keep slider positions across a light/dark switch.
            if (state) view.setState(state);
        }

        draw().catch((err) => {
            console.error("[dl-widget] vega", err);
            el.innerHTML = fallback;
            el.classList.add("dl-widget--error");
        });

        const off = api.onThemeChange(() => draw().catch(console.error));
        return () => {
            off();
            if (view) view.finalize();
        };
    });
})();
