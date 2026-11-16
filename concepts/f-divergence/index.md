# f-Divergence

> [!NOTE]
>
> **In one minute**
>
> The f-divergence compares two densities through a generating function f that is convex and satisfies f(1) = 0. Choosing f(x) = x log x gives the KL divergence.

The f-divergence is defined as[^/concepts/f-divergence/#f-divergence_wiki]

\\\[ \\operatorname{D}\_f = \\int f\\left(\\frac{p}{q}\\right) q\\mathrm d\\mu, \\\]

where \\(p\\) and \\(q\\) are two densities and \\(\\mu\\) is a reference distribution.

> [!WARNING]
>
> **Requirements on the generating function**
>
> The generating function \\(f\\) is required to
>
> - be convex, and
> - \\(f(1) =0\\).

For \\(f(x) = x \\log x\\) with \\(x=p/q\\), f-divergence is reduced to the KL divergence

\\\[ \\begin{align} &amp;\\int f\\left(\\frac{p}{q}\\right) q\\mathrm d\\mu \\\\ =&amp; \\int \\frac{p}{q} \\log \\left( \\frac{p}{q} \\right) \\mathrm d\\mu \\\\ =&amp; \\int p \\log \\left( \\frac{p}{q} \\right) \\mathrm d\\mu. \\end{align} \\\]

For more special cases of f-divergence, please refer to wikipedia[^/concepts/f-divergence/#f-divergence_wiki]. Nowozin 2016 also provides a concise review of f-divergence[^/concepts/f-divergence/#Nowozin2016].

---

**Related:** [KL Divergence](<https://dl.leima.is/concepts/kl-divergence/index.md>) · [f-GAN](<https://dl.leima.is/self-supervised/adversarial/f-gan/index.md>)

[^/concepts/f-divergence/#f-divergence_wiki]: Contributors to Wikimedia projects. F-divergence. In: Wikipedia \[Internet\]. 17 Jul 2021 \[cited 4 Sep 2021\]. Available: [https://en.wikipedia.org/wiki/F-divergence](<https://en.wikipedia.org/wiki/F-divergence >)

[^/concepts/f-divergence/#Nowozin2016]: Nowozin S, Cseke B, Tomioka R. f-GAN: Training Generative Neural Samplers using Variational Divergence Minimization. arXiv \[stat.ML\]. 2016. Available: [http://arxiv.org/abs/1606.00709](<http://arxiv.org/abs/1606.00709 >)
