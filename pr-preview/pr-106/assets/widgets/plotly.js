/*
 * Render a Plotly figure saved from a notebook, themed to the site.
 *
 *     <div class="dl-widget" data-widget="plotly"
 *          data-spec="notebooks/assets/diffusion_model/cell12-0.plotly.json" markdown>
 *     *Interactive Plotly figure: enable JavaScript to view it.*
 *     </div>
 *
 * scripts/stage_site.py writes these blocks for notebook outputs of type
 * application/vnd.plotly.v1+json. plotly.js (~4.8 MB) loads lazily from
 * jsDelivr, only on pages that contain a figure.
 */
(function () {
    const SRC = "https://cdn.jsdelivr.net/npm/plotly.js-dist-min@4.1.2/plotly.min.js";

    function themed(layout, c) {
        const axis = { gridcolor: c.fgLightest, zerolinecolor: c.fgLighter, linecolor: c.fgLighter, color: c.fgLight };
        const out = Object.assign({}, layout, {
            paper_bgcolor: "rgba(0,0,0,0)",
            plot_bgcolor: "rgba(0,0,0,0)",
            font: Object.assign({}, layout.font, { color: c.fg, family: getComputedStyle(document.body).fontFamily }),
            autosize: true,
        });
        delete out.template; // notebook templates hard-code light backgrounds
        delete out.width;
        for (const k of Object.keys(out)) {
            if (/^[xy]axis\d*$|^scene$/.test(k)) out[k] = Object.assign({}, out[k], axis);
        }
        if (!Object.keys(out).some((k) => /^[xy]axis/.test(k))) {
            out.xaxis = axis;
            out.yaxis = axis;
        }
        return out;
    }

    dlWidgets.register("plotly", (el, api) => {
        const url = api.resolve(el.dataset.spec);
        const fallback = el.innerHTML;
        let fig = null;
        let target = null;

        async function draw() {
            await api.loadScript(SRC);
            if (!fig) fig = await (await fetch(url)).json();
            if (!target) {
                target = document.createElement("div");
                el.replaceChildren(target);
            }
            await window.Plotly.react(target, fig.data, themed(fig.layout || {}, api.colors()), {
                responsive: true,
                displaylogo: false,
            });
        }

        draw().catch((err) => {
            console.error("[dl-widget] plotly", err);
            el.innerHTML = fallback;
            el.classList.add("dl-widget--error");
        });
        const off = api.onThemeChange(() => draw().catch(console.error));
        return () => {
            off();
            if (target && window.Plotly) window.Plotly.purge(target);
        };
    });
})();
