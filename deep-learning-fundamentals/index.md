# Deep Learning Fundamentals

> [!NOTE]
>
> **In one minute**
>
> An overview of the deep learning part: why a plain feedforward network is rarely enough, how self-supervised learning helps models generalize, and the notation used in the following chapters.

Deep learning, as the rising method for time series forecasting, requires the knowledge of some fundamental principles.

In this part, we explain and demonstrate some popular deep learning models. Note that we do not intend to cover all models but only discuss a few popular principles.

The simplest deep learning model, is a fully connected [Feedforward Neural Network (FFNN)](<https://en.wikipedia.org/wiki/Feedforward_neural_network>). A FFNN might work for in-distribution predictions, it is likely to overfit and perform poorly for out-of-distribution predictions. In reality, most of the deep learning models are much more complicated than a FFNN, and a large population of deep learning models are utilizing the self-supervised learning concept, providing better generalizations[^/deep-learning-fundamentals/#bib-Liu2020-yh].

In the following chapters, we provide some popular deep learning architectures and cool ideas.

> [!NOTE]
>
> **Notations**
>
> In this document, we use the following notations.
>
> - Sets, domains, abstract variables, \\(X\\), \\(Y\\);
> - Probability distribution \\(P\\), \\(Q\\);
> - Probability density \\(p\\), \\(q\\);
> - Slicing arrays from index \\(i\\) to index \\(j\\) using \\({}\_{i:j}\\).

---

**Related:** [Learning from Data](<https://dl.leima.is/deep-learning-fundamentals/learning/index.md>) · [Neural Networks](<https://dl.leima.is/deep-learning-fundamentals/neural-net/index.md>)

[^/deep-learning-fundamentals/#bib-Liu2020-yh]: Liu X., Zhang F., Hou Z. et al.. Self-supervised Learning: Generative or Contrastive. 2020. [http://arxiv.org/abs/2006.08218](<http://arxiv.org/abs/2006.08218>)
