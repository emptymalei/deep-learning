# Gradient Boosted Trees

> [!NOTE]
>
> **In one minute**
>
> Boosted trees are also an ensemble, but an additive one: the prediction is the sum of the trees' predictions, and training adds one tree at a time, each chosen to reduce the remaining loss. LightGBM and XGBoost are the popular implementations.
>
> **Read first:** [Random Forest](<https://dl.leima.is/trees/tree.random-forest/index.md>)

Boosted trees is another ensemble method of trees. Similar to [random forest](<https://dl.leima.is/trees/tree.random-forest/index.md>), boosted trees makes prediction by combining the predictions from each tree. However, instead of performing average, boosted trees are additive models where the prediction \\(f(\\mathbf X)\\) is the additions of each predictions[^/trees/tree.gbdt/#bib-Hastie2013-tt],

\\\[ f(\\mathbf X) = \\sum\_t^T f\_t(\\mathbf X), \\\]

where \\(f\_t(\\mathbf X)\\) is the prediction for tree \\(i\\) and \\(T\\) is the total number of trees. Given such a setup, the training becomes very different from random forests. As of 2023, there are two popular implementations of boosted trees, [LightGBM](<https://github.com/microsoft/LightGBM>) and [XGBoost](<https://github.com/dmlc/xgboost>). Training a boosted trees model finds a sequence of trees

\\\[ \\{ f\_1, f\_2, \\cdots, f\_t, \\cdots, f\_T \\}. \\\]

For a specified loss function \\(\\mathscr L(\\mathbf y, \\hat{\\mathbf y})\\), the sequence of trees helps reducing the loss step by step. At step \\(i\\), the loss is

\\\[ \\mathscr L(y, f\_1(\\mathbf X) + f\_2(\\mathbf X) + \\cdots + f\_i(\\mathbf X) ). \\\]

To optimize the model, we have to add a tree that reduces the loss the most and approximations are applied for numerical computations[^/trees/tree.gbdt/#bib-Chen2016-mi].

> [!NOTE]
>
> The [XGBoost documentation](<https://xgboost.readthedocs.io/en/stable/tutorials/model.html>) and the original paper on XGBoost explains the idea nicely with examples.
>
> Chen T, Guestrin C. XGBoost: A Scalable Tree Boosting System. arXiv \[cs.LG\]. 2016. Available: [http://arxiv.org/abs/1603.02754](<http://arxiv.org/abs/1603.02754>)
>
> There are more than one realization of gradient boosted trees[^/trees/tree.gbdt/#bib-Ke2017-jv][^/trees/tree.gbdt/#bib-Shi2018-bk].

---

**Related:** [Forecasting with Trees Using Darts](<https://dl.leima.is/trees/tree.darts/index.md>)

[^/trees/tree.gbdt/#bib-Hastie2013-tt]: Hastie T., Tibshirani R., Friedman J.. The Elements of Statistical Learning: Data Mining, Inference, and Prediction. *Springer Science &amp; Business Media*. 2013. [https://play.google.com/store/books/details?id=yPfZBwAAQBAJ](<https://play.google.com/store/books/details?id=yPfZBwAAQBAJ>)

[^/trees/tree.gbdt/#bib-Chen2016-mi]: Chen T., Guestrin C.. XGBoost: A Scalable Tree Boosting System. 2016. [http://arxiv.org/abs/1603.02754](<http://arxiv.org/abs/1603.02754>)

[^/trees/tree.gbdt/#bib-Ke2017-jv]: Ke G., Meng Q., Finley T. et al.. LightGBM: A Highly Efficient Gradient Boosting Decision Tree. *Advances in Neural Information Processing Systems*. 2017. [https://proceedings.neurips.cc/paper/2017/file/6449f44a102fde848669bdd9eb6b76fa-Paper.pdf](<https://proceedings.neurips.cc/paper/2017/file/6449f44a102fde848669bdd9eb6b76fa-Paper.pdf>)

[^/trees/tree.gbdt/#bib-Shi2018-bk]: Shi Y., Li J., Li Z.. Gradient Boosting With Piece-Wise Linear Regression Trees. 2018. [http://arxiv.org/abs/1802.05640](<http://arxiv.org/abs/1802.05640>)
