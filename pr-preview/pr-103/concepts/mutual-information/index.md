# Mutual Information

> [!NOTE]
>
> **In one minute**
>
> Mutual information of two random variables is the expected log ratio of their joint distribution to the product of their marginals. That makes it the KL divergence between the joint distribution and the product of the marginals.
>
> **Read first:** [KL Divergence](<https://dl.leima.is/concepts/kl-divergence/index.md>)

Mutual information is

\\\[ I(X;Y) = \\mathbb E\_{p\_{XY}} \\ln \\frac{P\_{XY}}{P\_X P\_Y}. \\\]

Mutual information is closely related to [KL divergence](<https://dl.leima.is/concepts/kl-divergence/index.md>),

\\\[ I(X;Y) = D\_{\\mathrm{KL}} \\left( P\_{XY}(x,y) \\parallel P\_X(x) P\_{Y}(y) \\right). \\\]

---

**Related:** [InfoGAN](<https://dl.leima.is/self-supervised/adversarial/infogan/index.md>) · [Deep Infomax](<https://dl.leima.is/self-supervised/contrastive/deep-infomax/index.md>)
