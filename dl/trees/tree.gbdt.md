# Gradient Boosted Trees

Boosted trees is another ensemble method of trees. Similar to [random forest](tree.random-forest.md), boosted trees makes prediction by combining the predictions from each tree. However, instead of performing average, boosted trees are additive models where the prediction $f(\mathbf X)$ is the additions of each predictions[@Hastie2013-tt],

$$
f(\mathbf X) = \sum_t^T f_t(\mathbf X),
$$

where $f_t(\mathbf X)$ is the prediction for tree $i$ and $T$ is the total number of trees. Given such a setup, the training becomes very different from random forests. As of 2023, there are two popular implementations of boosted trees, [LightGBM](https://github.com/microsoft/LightGBM) and [XGBoost](https://github.com/dmlc/xgboost). Training a boosted trees model finds a sequence of trees

$$
\{ f_1, f_2, \cdots, f_t, \cdots, f_T \}.
$$

For a specified loss function $\mathscr L(\mathbf y, \hat{\mathbf y})$, the sequence of trees helps reducing the loss step by step. At step $i$, the loss is

$$
\mathscr L(y, f_1(\mathbf X) + f_2(\mathbf X) + \cdots + f_i(\mathbf X) ).
$$

To optimize the model, we have to add a tree that reduces the loss the most and approximations are applied for numerical computations[@Chen2016-mi].

!!! note ""
    The [XGBoost documentation](https://xgboost.readthedocs.io/en/stable/tutorials/model.html) and the original paper on XGBoost explains the idea nicely with examples.

    Chen T, Guestrin C. XGBoost: A Scalable Tree Boosting System. arXiv [cs.LG]. 2016. Available: http://arxiv.org/abs/1603.02754


    There are more than one realization of gradient boosted trees[@Ke2017-jv][@Shi2018-bk].
