# f-Divergence

The f-divergence is defined as

$$
\operatorname{D}_f = \int f(\frac{p}{q}) q\mathrm d\mu,
$$

where $p$ and $q$ are two densities and $\mu$ is a reference distribution.

For $f(x) = x \log x$ with $x=p/q$, f-divergence is reduced to the KL divergence

$$
\begin{align}
\int f(\frac{p}{q}) q\mathrm d\mu =& \int \frac{p}{q} \log \left( \frac{p}{q} \right)\mathrm d\mu \\
\end{align}
$$