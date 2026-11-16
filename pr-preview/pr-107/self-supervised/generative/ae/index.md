# Autoencoders

> [!NOTE]
>
> **In one minute**
>
> An autoencoder encodes its input into a compact latent space and decodes it back. The simplest version is two small neural networks, an encoder and a decoder, trained with a loss such as cross entropy for binary data.

Autoencoders (AE) are machines that encode inputs into a compact latent space.

![AE](<https://dl.leima.is/self-supervised/generative/assets/ae/autoencoder-basic-1.jpg>)

> [!TIP]
>
> **Notation: dot (\\(\\cdot\\))**
>
> We use a single vertically centered dot, i.e., \\(\\cdot\\), to indicate that the function or machine can take in arguments.

A simple autoencoder can be achieved using two neural nets, e.g.,

\\\[ \\begin{align} {\\color{green}h} &amp;= {\\color{blue}g}{\\color{blue}(}{\\color{blue}b} + {\\color{blue}w} x{\\color{blue})} \\\\ \\hat x &amp;= {\\color{red}\\sigma}{\\color{red}(c} + {\\color{red}v} {\\color{green}h}{\\color{red})}, \\end{align} \\\]

where in this simple example,

- \\({\\color{blue}g(b + w \\cdot )}\\) is the encoder, and
- \\({\\color{red}\\sigma(c + v \\cdot )}\\) is the decoder.

For binary labels, we can use a simple cross entropy as the loss.

## Code

See Lippe[^/self-supervised/generative/ae/#Lippe].

---

**Related:** [Variational AutoEncoder](<https://dl.leima.is/self-supervised/generative/vae/index.md>)

[^/self-supervised/generative/ae/#Lippe]: Lippe P. Tutorial 9: Deep Autoencoders — UvA DL Notebooks v1.1 documentation. In: UvA Deep Learning Tutorials \[Internet\]. \[cited 20 Sep 2021\]. Available: [https://uvadlc-notebooks.readthedocs.io/en/latest/tutorial\_notebooks/tutorial9/AE\_CIFAR10.html](<https://uvadlc-notebooks.readthedocs.io/en/latest/tutorial_notebooks/tutorial9/AE_CIFAR10.html >)
