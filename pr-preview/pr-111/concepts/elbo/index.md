# ELBO

> [!NOTE]
>
> **In one minute**
>
> The evidence lower bound (ELBO) is a lower bound on the log probability of the data in a latent variable model. The page derives it twice: with Jensen's inequality, and from the KL divergence between an approximate posterior and the true posterior, where that KL divergence is the gap between the bound and the log probability.
>
> **Read first:** [Entropy](<https://dl.leima.is/concepts/entropy/index.md>) · [KL Divergence](<https://dl.leima.is/concepts/kl-divergence/index.md>)

Given a probability distribution density \\(p(x)\\) and a latent variable \\(z\\), the marginalization of the joint probability is

\\\[ \\int \\mathrm dz p(x, z) = p(x). \\\]

## Using Jensen's Inequality

In many models, we are interested in the log probability density \\(\\log p(X)\\) which can be decomposed using an auxiliary density of the latent variable \\(q(Z)\\),

\\\[ \\begin{align} \\log p(x) =&amp; \\log \\int \\mathrm d z p(x, z) \\\\ =&amp; \\log \\int \\mathrm d z p(x, z) \\frac{q(z)}{q(z)} \\\\ =&amp; \\log \\int \\mathrm d z q(z) \\frac{p(x, z)}{q(z)} \\\\ =&amp; \\log \\mathbb E\_q \\left\[ \\frac{p(x, z)}{q(z)} \\right\]. \\end{align} \\\]

> [!NOTE]
>
> **Jensen's Inequality**
>
> Jensen's inequality shows that[^/concepts/elbo/#jensens-inequality-wiki]
>
> \\\[ \\log \\mathbb E\_q \\left\[ \\frac{p(x, z)}{q(Z)} \\right\] \\geq \\mathbb E\_q \\left\[ \\log\\left(\\frac{p(x, z)}{q(Z)}\\right) \\right\], \\\]
>
> as \\(\\log\\) is a concave function.

Applying Jensen's inequality,

\\\[ \\begin{align} \\log p(x) =&amp; \\log \\mathbb E\_q \\left\[ \\frac{p(x, z)}{q(z)} \\right\] \\\\ \\geq&amp; \\mathbb E\_q \\left\[ \\log\\left(\\frac{p(x, z)}{q(z)}\\right) \\right\] \\\\ =&amp; \\mathbb E\_q \\left\[ \\log p(x, z)- \\log q(z) \\right\] \\\\ =&amp; \\mathbb E\_q \\left\[ \\log p(x, z) \\right\] - \\mathbb E\_q \\left\[ \\log q(z) \\right\] . \\end{align} \\\]

Using the definition of [entropy](<https://dl.leima.is/concepts/entropy/index.md#shannon-entropy>) and [cross entropy](<https://dl.leima.is/concepts/entropy/index.md#cross-entropy>), we know that

\\\[ H(q(z)) = - \\mathbb E\_q \\left\[ \\log q(z) \\right\] \\\]

is the entropy of \\(q(z)\\), and

\\\[ H(q(z);p(x,z)) = -\\mathbb E\_q \\left\[ \\log p(x, z) \\right\] \\\]

is the cross entropy. We define

\\\[ L = \\mathbb E\_q \\left\[ \\log p(x, z) \\right\] - \\mathbb E\_q \\left\[ \\log q(z) \\right\] = - H(q(z);p(x,z)) + H(q(z)), \\\]

which is called the evidence lower bound (**ELBO**). It is a lower bound because

\\\[ \\log p(x) \\geq L. \\\]

## Using KL Divergence

In a latent variable model, we need the posterior \\(p(z|x)\\). When this is intractable, we find an approximation \\(q(z|\\theta)\\) where \\(\\theta\\) is the parametrization, e.g., neural network parameters. To make sure we have a good approximation of the posterior, we require the [KL divergence](<https://dl.leima.is/concepts/kl-divergence/index.md>) of \\(q(z|\\theta)\\) and \\(p(z|x)\\) to be small. The KL divergence in this situation is[^/concepts/elbo/#Yang2017]

\\\[ \\begin{align} &amp;\\operatorname{ D}\_\\text{KL}(q(z|\\theta)\\parallel p(z|x)) \\\\ =&amp; -\\mathbb E\_q \\log\\frac{p(z|x)}{q(z|\\theta)} \\\\ =&amp; -\\mathbb E\_q \\log\\frac{p(x, z)/p(x)}{q(z|\\theta)} \\\\ =&amp; -\\mathbb E\_q \\log\\frac{p(x, z)}{q(z|\\theta)} - \\mathbb E\_q \\log\\frac{1}{p(x)} \\\\ =&amp; - L + \\log p(x). \\end{align} \\\]

Since \\(\\operatorname{D}\_{\\text{KL}}(q(z|\\theta)\\parallel p(z|x))\\geq 0\\), we have

\\\[ \\log p(x) \\geq L, \\\]

which also indicates that \\(L\\) is the lower bound of \\(\\log p(x)\\).

> [!NOTE]
>
> **Jensen gap**
>
> The difference between \\(\\log p(x)\\) and \\(L\\) is the Jensen gap, i.e.,
>
> \\\[ L - \\log p(x) = - \\operatorname{D}\_\\text{KL}(q(z|\\theta)\\parallel p(z|x)). \\\]

---

**Related:** [Variational AutoEncoder](<https://dl.leima.is/self-supervised/generative/vae/index.md>) · [Denoising Diffusion Probabilistic Models](<https://dl.leima.is/energy-based-models/ebm.diffusion/index.md>)

[^/concepts/elbo/#jensens-inequality-wiki]: Contributors to Wikimedia projects. Jensen’s inequality. In: Wikipedia \[Internet\]. 27 Aug 2021 \[cited 5 Sep 2021\]. Available: [https://en.wikipedia.org/wiki/Jensen%27s\_inequality](<https://en.wikipedia.org/wiki/Jensen%27s_inequality >)

[^/concepts/elbo/#Yang2017]: Yang X. Understanding the Variational Lower Bound. 14 Apr 2017 \[cited 5 Sep 2021\]. Available: [https://xyang35.github.io/2017/04/14/variational-lower-bound/](<https://xyang35.github.io/2017/04/14/variational-lower-bound/ >)
