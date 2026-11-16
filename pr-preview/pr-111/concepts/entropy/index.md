# Entropy

> [!NOTE]
>
> **In one minute**
>
> Shannon entropy is the expected information content of a distribution. Cross entropy measures one distribution against another and splits into the entropy plus the KL divergence; it is widely used as the loss in classification.

## Shannon Entropy

Shannon entropy \\(S\\) is the expectation of information content \\(I(X)=-\\log \\left(p\\right)\\)[^/concepts/entropy/#shannon_entropy_wiki],

\\\[\\begin{equation} H(p) = \\mathbb E\_{p}\\left\[ -\\log \\left(p\\right) \\right\]. \\end{equation}\\\]

## Cross Entropy

Cross entropy is[^/concepts/entropy/#cross_entropy_wiki]

\\\[ H(p, q) = \\mathbb E\_{p} \\left\[ -\\log q \\right\]. \\\]

Cross entropy \\(H(p, q)\\) can also be decomposed,

\\\[ H(p, q) = H(p) + \\operatorname{D}\_{\\mathrm{KL}} \\left( p \\parallel q \\right), \\\]

where \\(H(p)\\) is the [entropy of \\(P\\)](<#shannon-entropy>) and \\(\\operatorname{D}\_{\\mathrm{KL}}\\) is the [KL Divergence](<https://dl.leima.is/concepts/kl-divergence/index.md>).

Cross entropy is widely used in classification problems, e.g., logistic regression.

## Try It: Entropy, Cross Entropy and KL Divergence

Two distributions \\(p\\) and \\(q\\) over six classes. Move their peaks and sharpness, and watch the decomposition above: the cross entropy bar is the entropy of \\(p\\) plus the KL divergence.

An interactive chart of two distributions over six classes, with sliders for the peak and sharpness of each. It shows the entropy of \\(p\\), the cross entropy \\(H(p, q)\\) and the KL divergence as bars, in nats. It needs JavaScript. In short: \\(H(p, q) = H(p) + \\operatorname{D}\_{\\mathrm{KL}}(p \\parallel q)\\), so the cross entropy is smallest when \\(q\\) equals \\(p\\).

---

**Related:** [KL Divergence](<https://dl.leima.is/concepts/kl-divergence/index.md>) · [Information Gain](<https://dl.leima.is/concepts/information-gain/index.md>)

[^/concepts/entropy/#shannon_entropy_wiki]: Contributors to Wikimedia projects. Entropy (information theory). In: Wikipedia \[Internet\]. 29 Aug 2021 \[cited 4 Sep 2021\]. Available: [https://en.wikipedia.org/wiki/Entropy\_(information\_theory)](<https://en.wikipedia.org/wiki/Entropy_(information_theory) >)

[^/concepts/entropy/#cross_entropy_wiki]: Contributors to Wikimedia projects. Cross entropy. In: Wikipedia \[Internet\]. 4 Jul 2021 \[cited 4 Sep 2021\]. Available: [https://en.wikipedia.org/wiki/Cross\_entropy](<https://en.wikipedia.org/wiki/Cross_entropy >)
