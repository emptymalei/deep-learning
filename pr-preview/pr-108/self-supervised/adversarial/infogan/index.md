# InfoGAN

> [!NOTE]
>
> **In one minute**
>
> InfoGAN makes the latent space of a GAN more interpretable. Next to the random noise it feeds the generator a latent code, and it adds the mutual information between that code and the generated data to the GAN objective, so the generator is penalized when it loses the information in the code.
>
> **Read first:** [GAN](<https://dl.leima.is/self-supervised/adversarial/gan/index.md>) · [Mutual Information](<https://dl.leima.is/concepts/mutual-information/index.md>)

In GAN, the latent space input is usually random noise, e.g., Gaussian noise. The objective of [GAN](<https://dl.leima.is/self-supervised/adversarial/gan/index.md>) is a very generic one. It doesn't say anything about how exactly the latent space will be used. This is not desirable in many problems. We would like to have more interpretability in the latent space. InfoGAN introduced constraints to the objective to enforce interpretability of the latent space[^/self-supervised/adversarial/infogan/#Chen2016].

## Constraint

The constraint InfoGAN proposed is [mutual information](<https://dl.leima.is/concepts/mutual-information/index.md>),

\\\[ \\underset{{\\color{red}G}}{\\operatorname{min}} \\underset{{\\color{green}D}}{\\operatorname{max}} V\_I ({\\color{green}D}, {\\color{red}G}) = V({\\color{green}D}, {\\color{red}G}) - \\lambda I(c; {\\color{red}G}(z,c)), \\\]

where

- \\(c\\) is the latent code,
- \\(z\\) is the random noise input,
- \\(V({\\color{green}D}, {\\color{red}G})\\) is the objective of GAN,
- \\(I(c; {\\color{red}G}(z,c))\\) is the mutual information between the input latent code and generated data.

Using the lambda multiplier, we punish the model if the generator loses information in latent code \\(c\\).

## Training

![InfoGAN](<https://dl.leima.is/self-supervised/adversarial/assets/infogan/infogan-structure-1.jpeg>)

The training steps are almost the same as [GAN](<https://dl.leima.is/self-supervised/adversarial/gan/index.md>) but with one extra loss to be calculated in each mini-batch.

1. Train \\(\\color{red}G\\) using loss: \\(\\operatorname{MSE}(v', v)\\);
2. Train \\(\\color{green}D\\) using loss: \\(\\operatorname{MSE}(v', v)\\);
3. Apply Constraint:

   1. Sample data from mini-batch;
   2. Calculate loss \\(\\lambda\_{l} H(l';l)+\\lambda\_c \\operatorname{MSE}(c,c')\\)

## Code

[eriklindernoren/PyTorch-GAN](<https://github.com/eriklindernoren/PyTorch-GAN/blob/master/implementations/infogan/infogan.py>)

---

**Related:** [f-GAN](<https://dl.leima.is/self-supervised/adversarial/f-gan/index.md>)

[^/self-supervised/adversarial/infogan/#Chen2016]: Chen X, Duan Y, Houthooft R, Schulman J, Sutskever I, Abbeel P. InfoGAN: Interpretable Representation Learning by Information Maximizing Generative Adversarial Nets. arXiv \[cs.LG\]. 2016. Available: [http://arxiv.org/abs/1606.03657](<http://arxiv.org/abs/1606.03657 >)

[^/self-supervised/adversarial/infogan/#Agakov2004]: Agakov DBF. The im algorithm: a variational approach to information maximization. Adv Neural Inf Process Syst. 2004. Available: [https://books.google.com/books?hl=en&amp;lr=&amp;id=0F-9C7K8fQ8C&amp;oi=fnd&amp;pg=PA201&amp;dq=Algorithm+variational+approach+Information+Maximization+Barber+Agakov&amp;ots=TJGrkVS610&amp;sig=yTKM2ZdcZQBTY4e5Vqk42ayUDxo](<https://books.google.com/books?hl=en&amp;lr=&amp;id=0F-9C7K8fQ8C&amp;oi=fnd&amp;pg=PA201&amp;dq=Algorithm+variational+approach+Information+Maximization+Barber+Agakov&amp;ots=TJGrkVS610&amp;sig=yTKM2ZdcZQBTY4e5Vqk42ayUDxo >)
