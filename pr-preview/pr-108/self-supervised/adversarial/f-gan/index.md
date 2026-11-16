# f-GAN

> [!NOTE]
>
> **In one minute**
>
> f-GAN generalizes the GAN: instead of the Jensen-Shannon divergence, it estimates an f-divergence between the data distribution and the generated distribution. Training searches for the saddle point of a variational objective, and a particular choice of functions recovers the loss of the original GAN.
>
> **Read first:** [GAN](<https://dl.leima.is/self-supervised/adversarial/gan/index.md>) · [f-Divergence](<https://dl.leima.is/concepts/f-divergence/index.md>)

The essence of [GAN](<https://dl.leima.is/self-supervised/adversarial/gan/index.md#divergence>) is comparing the generated distribution \\(p\_G\\) and the data distribution \\(p\_\\text{data}\\). The vanilla GAN considers the Jensen-Shannon divergence \\(\\operatorname{D}\_\\text{JS}(p\_\\text{data}\\Vert p\_{G})\\). The discriminator \\({\\color{green}D}\\) serves the purpose of forcing this divergence to be small.

> [!NOTE]
>
> **Why do we need the discriminator?**
>
> If the JS divergence is an objective, why do we need the discriminator? Even in f-GAN we need a functional to approximate the f-divergence. This functional we choose works like the discriminator of GAN.

There exists a more generic form of JS divergence, which is called [f-divergence](<https://dl.leima.is/concepts/f-divergence/index.md>)[^/self-supervised/adversarial/f-gan/#f-divergence_wiki]. f-GAN obtains the model by estimating the f-divergence between the data distribution and the generated distribution[^/self-supervised/adversarial/f-gan/#Nowozin2016].

## Variational Divergence Minimization

The Variational Divergence Minimization (VDM) extends the variational estimation of f-divergence[^/self-supervised/adversarial/f-gan/#Nowozin2016]. VDM searches for the saddle point of an objective \\(F({\\color{red}\\theta}, {\\color{blue}\\omega})\\), i.e., min w.r.t. \\(\\theta\\) and max w.r.t \\({\\color{blue}\\omega}\\), where \\({\\color{red}\\theta}\\) is the parameter set of the generator \\({\\color{red}Q\_\\theta}\\), and \\({\\color{blue}\\omega}\\) is the parameter set of the variational approximation to estimate f-divergence, \\({\\color{blue}T\_\\omega}\\).

The objective \\(F({\\color{red}\\theta}, {\\color{blue}\\omega})\\) is related to the choice of \\(f\\) in f-divergence and the variational functional \\({\\color{blue}T}\\),

\\\[ \\begin{align} &amp; F(\\theta, \\omega)\\\\ =&amp; \\mathbb E\_{x\\sim p\_\\text{data}} \\left\[ {\\color{blue}T\_\\omega}(x) \\right\] - \\mathbb E\_{x\\sim {\\color{red}Q\_\\theta} } \\left\[ f^\*({\\color{blue}T\_\\omega}(x)) \\right\] \\\\ =&amp; \\mathbb E\_{x\\sim p\_\\text{data}} \\left\[ g\_f(V\_{\\color{blue}\\omega}(x)) \\right\] - \\mathbb E\_{x\\sim {\\color{red}Q\_\\theta} } \\left\[ f^\*(g\_f(V\_{\\color{blue}\\omega}(x))) \\right\]. \\end{align} \\\]

In the above objective,

- \\(f^\*\\) is the [Legendre–Fenchel transformation](<https://en.wikipedia.org/wiki/Convex_conjugate>) of \\(f\\), i.e., \\(f^\*(t) = \\operatorname{sup}\_{u\\in \\mathrm{dom}\_f}\\left\\{ ut - f(u) \\right\\}\\).

> [!TIP]
>
> **\\(T\\)**
>
> The function \\(T\\) is used to estimate the lower bound of f-divergence[^/self-supervised/adversarial/f-gan/#Nowozin2016].

> [!TIP]
>
> **Choice of \\(g\_f\\) and \\(V\\)**
>
> Nowozin et al provided a table for \\(g\_f\\) and \\(V\\)[^/self-supervised/adversarial/f-gan/#Nowozin2016].
>
> ![](<https://dl.leima.is/self-supervised/adversarial/assets/f-gan/f-gan-gf-v.png>)

We estimate

- \\(\\mathbb E\_{x\\sim p\_\\text{data}}\\) by sampling from the mini-batch, and
- \\(\\mathbb E\_{x\\sim {\\color{red}Q\_\\theta} }\\) by sampling from the generator.

> [!NOTE]
>
> **Reduce to GAN**
>
> The VDM loss can be reduced to the [loss of GAN](<https://dl.leima.is/self-supervised/adversarial/gan/index.md#the-minimax-game-loss>) by setting[^/self-supervised/adversarial/f-gan/#Nowozin2016]
>
> \\\[ \\begin{align} \\log {\\color{green}D\_\\omega} =&amp; g\_f(V\_{\\color{blue}\\omega}(x)) \\\\ \\log \\left( 1 - {\\color{green}D\_\\omega} \\right) =&amp; -f^\*\\left( g\_f(V\_{\\color{blue}\\omega}(x)) \\right). \\end{align} \\\]
>
> It is straightforward to validate that the following result is a solution to the above set of equations,
>
> \\\[ g\_f(V) = \\log \\frac{1}{1 + e^{-V}}. \\\]

## Code

- [minlee077/f-GAN-pytorch](<https://github.com/minlee077/f-GAN-pytorch>)
- [shayneobrien/generative-models](<https://github.com/shayneobrien/generative-models/blob/master/src/f_gan.py>)

---

**Related:** [InfoGAN](<https://dl.leima.is/self-supervised/adversarial/infogan/index.md>)

[^/self-supervised/adversarial/f-gan/#f-divergence_wiki]: Contributors to Wikimedia projects. F-divergence. In: Wikipedia \[Internet\]. 17 Jul 2021 \[cited 6 Sep 2021\]. Available: [https://en.wikipedia.org/wiki/F-divergence\#Instances\_of\_f-divergences](<https://en.wikipedia.org/wiki/F-divergence#Instances_of_f-divergences >)

[^/self-supervised/adversarial/f-gan/#Nowozin2016]: Nowozin S, Cseke B, Tomioka R. f-GAN: Training Generative Neural Samplers using Variational Divergence Minimization. arXiv \[stat.ML\]. 2016. Available: [http://arxiv.org/abs/1606.00709](<http://arxiv.org/abs/1606.00709 >)

[^/self-supervised/adversarial/f-gan/#convex_conjugate_wiki]: Contributors to Wikimedia projects. Convex conjugate. In: Wikipedia \[Internet\]. 20 Feb 2021 \[cited 7 Sep 2021\]. Available: [https://en.wikipedia.org/wiki/Convex\_conjugate](<https://en.wikipedia.org/wiki/Convex_conjugate >)
