# Flow

> [!NOTE]
>
> **In one minute**
>
> A flow model transforms a simple distribution into a complex one through a chain of coordinate transformations, tracking the density with the Jacobian. It runs in a generative direction (sample, then transform) or a normalizing direction (simplify the data distribution).

![Generative Flow](<https://dl.leima.is/self-supervised/generative/assets/flow/generative-flow-encoding-decoding.png>)

For a probability density \\(p(x)\\) and a transformation of coordinate \\(x=g(z)\\) or \\(z=f(x)\\), the density can be expressed using the coordinate transformations, i.e.,

\\\[ \\begin{align} p(x) &amp;= \\tilde p (f(x)) \\lvert \\operatorname{det} \\operatorname{D} g(f(x)) \\rvert^{-1} \\\\ &amp;= \\tilde p(f(x)) \\lvert \\operatorname{det}\\operatorname{D} f(x) \\rvert \\end{align} \\\]

where the Jacobian is

\\\[ \\operatorname{D} g(z) \\to \\frac{\\partial }{\\partial z} g. \\\]

The operation \\(g\_{\*}\\circ \\tilde p(z)\\) is the push forward of \\(\\tilde p(z)\\). The operation \\(g\_{\*}\\) will pushforward simple distribution \\(\\tilde p(z)\\) to a more complex distribution \\(p(x)\\).

- The generative direction: sample \\(z\\) from distribution \\(\\tilde p(z)\\), apply transformation \\(g(z)\\);
- The normalizing direction: "simplify" \\(p(x)\\) to some simple distribution \\(\\tilde p(z)\\).

The key to the flow model is the chaining of the transformations

\\\[ \\operatorname{det} \\operatorname{D} f(x) = \\Pi\_{i=1}^N \\operatorname{det} \\operatorname{D} f\_i (x\_i) \\\]

where

\\\[ \\begin{align} x\_i &amp;= g\_i \\circ \\cdots \\circ g\_1 (z)\\\\ &amp;= f\_{i+1} \\circ \\cdots \\circ f\_N (x). \\end{align} \\\]

---

**Related:** [Forecasting with Flow](<https://dl.leima.is/time-series-deep-learning/timeseries.flow/index.md>)

[^/self-supervised/generative/flow/#Liu2020]: Liu X, Zhang F, Hou Z, Wang Z, Mian L, Zhang J, et al. Self-supervised Learning: Generative or Contrastive. arXiv \[cs.LG\]. 2020. Available: [http://arxiv.org/abs/2006.08218](<http://arxiv.org/abs/2006.08218 >)
