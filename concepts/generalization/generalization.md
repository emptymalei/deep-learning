# Generalization

To measure the generalization, we define a generalization error[@Roelofs2019-dm],

$$
\mathcal G = \mathcal L_{P}(\hat f) - \mathcal L_E(\hat f),
$$

where $\mathcal L_{P}$ is the population loss, $\mathcal L_E$ is the empirical loss, and $\hat f$ is our model by minimizing the empirical loss.

However, we do not know the actual joint probability $p(x, y)$ of our dataset $\\{x_i, y_i\\}$. Thus the population loss is not known. In machine learning, we usually use cross validation where we split our dataset into train and test dataset. We approximate the population loss using the test dataset.
