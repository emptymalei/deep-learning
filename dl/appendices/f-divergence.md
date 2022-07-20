# f-Divergence

The f-divergence is defined as[^f-divergence_wiki]

$$
\operatorname{D}_f = \int f\left(\frac{p}{q}\right) q\mathrm d\mu,
$$

where $p$ and $q$ are two densities and $\mu$ is a reference distribution.

!!! warning "Requirements on the generating function"

    The generating function $f$ is required to

    - be convex, and
    - $f(1) =0$.



For $f(x) = x \log x$ with $x=p/q$, f-divergence is reduced to the KL divergence

$$
\begin{align}
&\int f\left(\frac{p}{q}\right) q\mathrm d\mu \\
=& \int \frac{p}{q} \log \left( \frac{p}{q} \right) \mathrm d\mu \\
=&  \int p \log \left( \frac{p}{q} \right) \mathrm d\mu.
\end{align}
$$

For more special cases of f-divergence, please refer to wikipedia[^f-divergence_wiki]. Nowozin 2016 also provides a concise review of f-divergence[^Nowozin2016].



[^f-divergence_wiki]: Contributors to Wikimedia projects. F-divergence. In: Wikipedia [Internet]. 17 Jul 2021 [cited 4 Sep 2021]. Available: https://en.wikipedia.org/wiki/F-divergence


[^Nowozin2016]: Nowozin S, Cseke B, Tomioka R. f-GAN: Training Generative Neural Samplers using Variational Divergence Minimization. arXiv [stat.ML]. 2016. Available: http://arxiv.org/abs/1606.00709
