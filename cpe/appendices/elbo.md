# ELBO

Given a probability distribution density $p(x)$ and a latent variable $z$, the marginalization of the joint probability is

$$
\int \mathrm dz p(x, z) = p(x).
$$


## Using Jensen's Inequality

In many models, we are interested in the log probability density $\log p(X)$ which can be decomposed using an auxiliary density of the latent variable $q(Z)$,

$$
\begin{align}
\log p(x) =& \log \int \mathrm d z p(x, z) \\
=& \log \int \mathrm d z p(x, z) \frac{q(z)}{q(z)} \\
=& \log \int \mathrm d z q(x) \frac{p(x, z)}{q(z)} \\
=& \log \mathbb E_q \left[ \frac{p(x, z)}{q(z)} \right].
\end{align}
$$

Jensen's inequality shows that

$$
\log \mathbb E_q \left[ \frac{p(X, Z)}{q(Z)} \right] \geq  \mathbb E_q \left[ \log\left(\frac{p(X, Z)}{q(Z)}\right) \right],
$$

as $\log$ is a concave function.

Applying this inequality, we get

$$
\begin{align}
\log p(X) =& \log \mathbb E_q \left[ \frac{p(X, Z)}{q(Z)} \right] \\
\geq&  \mathbb E_q \left[ \log\left(\frac{p(X, Z)}{q(Z)}\right) \right] \\
=& \mathbb E_q \left[ \log p(X, Z)- \log q(Z) \right] \\
=& \mathbb E_q \left[ \log p(X, Z) \right] - \mathbb E_q \left[ \log q(Z) \right] .
\end{align}
$$

Using the definition of entropy and cross entropy, we know that

$$
H(q(Z)) = - \mathbb E_q \left[ \log q(Z) \right]
$$

is the entropy of $q(Z)$ and

$$
H(q(Z);p(X,Z)) = -\mathbb E_q \left[ \log p(X, Z) \right]
$$

is the cross entropy. For convinience, we denote

$$
L = \mathbb E_q \left[ \log p(X, Z) \right] - \mathbb E_q \left[ \log q(Z) \right] = - H(q(Z);p(X,Z)) + H(q(Z)),
$$

which is called the evidence lower bound (ELBO) as

$$
\log p(X) \geq L.
$$

## KL Divergence

In a latent variable model, we might need to calculate the posterior $p(Z|X)$. When this is intractable, we find an approximation $q(Z|\theta)$ where $\theta$ is the parametrization such as neural network parameters. To make sure we have a good approximation of the posterior, we find the KL divergence of $q(Z|\theta)$ and $p(Z|X)$.

The KL divergence is

$$
\begin{align}
D_\text{KL}(q(Z|\theta)\parallel p(Z|X)) =& -\mathbb E_q \log\frac{p(X|Z)}{q(Z|\theta)} \\
=& -\mathbb E_q \log\frac{p(X, Z)/p(X)}{q(Z|\theta)} \\
=& -\mathbb E_q \log\frac{p(X, Z)}{q(Z|\theta)} - \mathbb E_q \log\frac{1}{p(X)} \\
=& - L + \log p(X).
\end{align}
$$

Since $D(q(Z|\theta)\parallel p(Z|X))\geq 0$, we have

$$
\log p(X) \geq L,
$$

which also indicates that $L$ is the lower bound of $\log p(X)$.

In fact,

$$
L - \log p(X) = - D_\text{KL}(q(Z|\theta)\parallel p(Z|X))
$$

is the Jensen gap.