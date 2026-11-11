window.MathJax = {
    loader: {load: ['[tex]/boldsymbol']},
    tex: {
        inlineMath: [
            ["\\(", "\\)"]
        ],
        displayMath: [
            ["\\[", "\\]"]
        ],
        processEscapes: true,
        processEnvironments: true,
        tags: "ams",
        packages: {'[+]': ['boldsymbol']}
    },
    options: {
        ignoreHtmlClass: ".*|",
        processHtmlClass: "arithmatex"
    }
};

// On the first page load MathJax typesets by itself once it has loaded; this
// hook re-typesets after instant navigation. Guard against the hook firing
// before MathJax is ready (Zensical emits document$ earlier than Material).
document$.subscribe(() => {
    if (!window.MathJax || typeof MathJax.typesetPromise !== "function") return;
    MathJax.startup.output.clearCache();
    MathJax.typesetClear();
    MathJax.texReset();
    MathJax.typesetPromise();
})
