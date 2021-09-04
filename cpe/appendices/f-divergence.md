# f-Divergence

The f-divergence is defined as[^f-divergence_wiki]

$$
\operatorname{D}_f = \int f(\frac{p}{q}) q\mathrm d\mu,
$$

where $p$ and $q$ are two densities and $\mu$ is a reference distribution.

For $f(x) = x \log x$ with $x=p/q$, f-divergence is reduced to the KL divergence

$$
\begin{align}
&\int f(\frac{p}{q}) q\mathrm d\mu \\
=& \int \frac{p}{q} \log \left( \frac{p}{q} \right) \mathrm d\mu \\
=&  \int p \log \left( \frac{p}{q} \right) \mathrm d\mu.
\end{align}
$$

For more special cases of f-divergence, please refer to wikipedia[^f-divergence_wiki].




[^f-divergence_wiki]: Contributors to Wikimedia projects. F-divergence. In: Wikipedia [Internet]. 17 Jul 2021 [cited 4 Sep 2021]. Available: https://en.wikipedia.org/wiki/F-divergence

