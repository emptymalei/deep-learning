# Entropy


## Shannon Entropy

Shannon entropy $S$ is the expectation of information content $I(X)=-\log \left(p\right)$[^shannon_entropy_wiki],

\begin{equation}
H(p) = \mathbb E_{p}\left[ -\log \left(p\right) \right].
\end{equation}




[^shannon_entropy_wiki]: Contributors to Wikimedia projects. Entropy (information theory). In: Wikipedia [Internet]. 29 Aug 2021 [cited 4 Sep 2021]. Available: https://en.wikipedia.org/wiki/Entropy_(information_theory)





## Cross Entropy

Cross entropy is[^cross_entropy_wiki]

$$
H(p, q) = \mathbb E_{p} \left[ -\log q \right].
$$

Cross entropy $H(p, q)$ can also be decomposed,

$$
H(p, q) = H(p) + \operatorname{D}_{\mathrm{KL}} \left( p \parallel q \right),
$$

where $H(p)$ is the [entropy of $P$](#shannon-entropy) and $\operatorname{D}_{\mathrm{KL}}$ is the [KL Divergence](kl-divergence.md).


Cross entropy is widely used in classification problems, e.g., logistic regression.


[^cross_entropy_wiki]: Contributors to Wikimedia projects. Cross entropy. In: Wikipedia [Internet]. 4 Jul 2021 [cited 4 Sep 2021]. Available: https://en.wikipedia.org/wiki/Cross_entropy




