# Deep Learning Fundamentals

Deep learning, as the rising method for time series forecasting, requires the knowledge of some fundamental principles.

In this part, we explain and demonstrate some popular deep learning models. Note that we do not intend to cover all models but only discuss a few popular principles.

The simplest deep learning model, is a fully connected [Feedforward Neural Network (FFNN)](https://en.wikipedia.org/wiki/Feedforward_neural_network). A FFNN might work for in-distribution predictions, it is likely to overfit and perform poorly for out-of-distribution predictions. In reality, most of the deep learning models are much more complicated than a FFNN, and a large population of deep learning models are utilizing the self-supervised learning concept, providing better generalizations[@Liu2020-yh].

In the following chapters, we provide some popular deep learning architectures and cool ideas. We follow [(Liu et al. 2020)](https://arxiv.org/abs/2006.08218) to categorize some of the models.

!!! info "Notations"

    In this document, we use the following notations.

    - Sets, domains, abstract variables, $X$, $Y$;
    - Probability distribution $P$, $Q$;
    - Probability density $p$, $q$.
