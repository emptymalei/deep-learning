# Contrastive Predictive Coding

> [!NOTE]
>
> **In one minute**
>
> Contrastive predictive coding (CPC) encodes each segment of a sequence, models the encodings with an autoregressive model, and trains with the InfoNCE loss, which estimates a lower bound of mutual information. The encoders can then be used for downstream tasks such as classification.
>
> **Read first:** [Mutual Information](<https://dl.leima.is/concepts/mutual-information/index.md>) · [Autoregressive Model](<https://dl.leima.is/self-supervised/generative/autoregressive/index.md>)

Contrastive Predictive Coding, CPC, is an autoregressive model combined with InfoNCE loss[^/self-supervised/contrastive/contrastive-predictive-coding/#Oord2018].

> [!NOTE]
>
> **Predictive Coding**
>
> As a related topic, predictive coding is a different scheme than backpropagation. Predictive coding updates the weights using local updating rules only[^/self-supervised/contrastive/contrastive-predictive-coding/#bib-Millidge2020-mr].

There are two key ideas in CPC:

- Autoregressive models in latent space, and
- InfoNCE loss that combines mutual information and NCE.

For the series of segments, \\(\\{x\_t\\}\\), we apply an encoder on each segment, and calculate the latent space, \\(\\{{\\color{blue}\\hat x\_t}\\}\\). The latent space \\(\\{{\\color{blue}\\hat x\_t}\\}\\) is then modeled using an autoregressive model to calculate the coding, \\(\\{{\\color{red}c\_t}\\}\\).

![CPC overview](<https://dl.leima.is/self-supervised/contrastive/assets/contrastive-predictive-coding/cpc-overview.png>)

van den Oord et al

The loss is built on NCE to estimate the lower bound of mutual information,

\\\[ \\mathcal L = -\\mathbb E\_X \\left\[ \\log \\frac{f\_k(x\_{t+k}, c\_t)}{\\sum\_{j} f\_k(x\_{j}, c\_t) } \\right\], \\\]

where \\(f\_k(x\_{t+i}, c\_t)\\) is estimated using a log-bilinear model, \\(f\_k(x\_{t+i}, c\_t) = \\exp\\left( z\_{t+i} W\_i c\_t \\right)\\). This is also a cross entropy loss.

Minimizing \\(\\mathcal L\\) leads to a \\(f\_k\\) that estimates the ratio[^/self-supervised/contrastive/contrastive-predictive-coding/#Oord2018]

\\\[ \\frac{p(x\_{t+k}\\mid c\_t)}{p(x\_{t+k})} = \\frac{p(x\_{t+k}, c\_t)}{p(x\_{t+k})p(c\_t)}. \\\]

We can perform downstream tasks such as classifications using the encoders.

> [!WARNING]
>
> **Maximizing this lower bound?**
>
> This so-called lower bound for mutual information in this case is not always going to work[^/self-supervised/contrastive/contrastive-predictive-coding/#Newell2020]. In some cases, the representations learned using this lower bound doesn't help or even worsen the performance of downstream tasks.

## Code

[rschwarz15/CPCV2-PyTorch](<https://github.com/rschwarz15/CPCV2-PyTorch>)

---

**Related:** [Deep Infomax](<https://dl.leima.is/self-supervised/contrastive/deep-infomax/index.md>) · [Alignment and Uniformity](<https://dl.leima.is/concepts/alignment-and-uniformity/index.md>)

[^/self-supervised/contrastive/contrastive-predictive-coding/#Newell2020]: Newell A, Deng J. How Useful is Self-Supervised Pretraining for Visual Tasks? arXiv \[cs.CV\]. 2020. Available: [http://arxiv.org/abs/2003.14323](<http://arxiv.org/abs/2003.14323 >)

[^/self-supervised/contrastive/contrastive-predictive-coding/#Oord2018]: van den Oord A, Li Y, Vinyals O. Representation learning with Contrastive Predictive Coding. arXiv \[cs.LG\]. 2018. Available: [http://arxiv.org/abs/1807.03748](<http://arxiv.org/abs/1807.03748 >)

[^/self-supervised/contrastive/contrastive-predictive-coding/#bib-Millidge2020-mr]: Millidge B., Tschantz A., Buckley C.L.. Predictive Coding Approximates Backprop along Arbitrary Computation Graphs. 2020. [http://arxiv.org/abs/2006.04182](<http://arxiv.org/abs/2006.04182>)
