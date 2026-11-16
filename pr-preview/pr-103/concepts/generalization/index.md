# Generalization

> [!NOTE]
>
> **In one minute**
>
> The generalization error is the difference between a model's loss on the whole population and its loss on the data it was fitted to. The population loss is unknown, so in practice it is approximated with a held-out test set.

To measure the generalization, we define a generalization error[^/concepts/generalization/#bib-Roelofs2019-dm],

\\\[ \\mathcal G = \\mathcal L\_{P}(\\hat f) - \\mathcal L\_E(\\hat f), \\\]

where \\(\\mathcal L\_{P}\\) is the population loss, \\(\\mathcal L\_E\\) is the empirical loss, and \\(\\hat f\\) is our model by minimizing the empirical loss.

However, we do not know the actual joint probability \\(p(x, y)\\) of our dataset \\(\\\\{x\_i, y\_i\\\\}\\). Thus the population loss is not known. In machine learning, we usually use cross validation where we split our dataset into train and test dataset. We approximate the population loss using the test dataset.

---

**Related:** [Time Series Forecasting Evaluation](<https://dl.leima.is/time-series/timeseries-evaluation.forecasting/index.md>)

[^/concepts/generalization/#bib-Roelofs2019-dm]: Roelofs R.. Measuring Generalization and Overfitting in Machine Learning. 2019. [https://escholarship.org/uc/item/6j01x9mz](<https://escholarship.org/uc/item/6j01x9mz>)
