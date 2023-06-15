# Hierarchical Time Series Reconciliation

Reconciliation is a post-processing method to adjust the forecasts to be coherent. Given **base forecasts** $\hat{\mathbf y}(t)$ (forecasts for all levels but each level forecasted independently), we use $\mathbf P$ to map them to the bottom-level forecasts

$$
\begin{equation}
\tilde{\mathbf b}(t) = \mathbf P \hat{\mathbf y}(t).
\end{equation}
$$

!!! note "$P$ and $S$"

    In the [previous section](../timeseries-hierarchical.data.md), we discussed the summing matrix $\color{blue}S$. The summing matrix maps the bottom-level forecasts $\color{red}{\mathbf b}(t)$ to all forecasts on all levels $\color{green}\mathbf y(t)$. The example we provided was

    $${\color{green}\begin{pmatrix}
    s(t) \\
    s_\mathrm{CA}(t) \\
    s_\mathrm{TX}(t) \\
    s_\mathrm{WI}(t)
    \end{pmatrix}} = {\color{blue}\begin{pmatrix}
    1 & 1 & 1 \\
    1 & 0 & 0 \\
    0 & 1 & 0 \\
    0 & 0 & 1
    \end{pmatrix}} {\color{red}\begin{pmatrix}
    s_\mathrm{CA}(t) \\
    s_\mathrm{TX}(t) \\
    s_\mathrm{WI}(t)
    \end{pmatrix}}.
    $$

    If we forecast different levels independently, the forecasts we get

    $$
    \hat{\mathbf y}(t) = \begin{pmatrix}
    \hat s(t) \\
    \hat s_\mathrm{CA}(t) \\
    \hat s_\mathrm{TX}(t) \\
    \hat s_\mathrm{WI}(t)
    \end{pmatrix},
    $$

    are not necessarily coherent. However, if we can choose a proper $\mathbf P$, we can convert the base forecasts into some bottom-level forecasts

    $$
    \begin{pmatrix}
    \tilde s_\mathrm{CA}(t) \\
    \tilde s_\mathrm{TX}(t) \\
    \tilde s_\mathrm{WI}(t)
    \end{pmatrix} = \mathbf P \begin{pmatrix}
    \hat s(t) \\
    \hat s_\mathrm{CA}(t) \\
    \hat s_\mathrm{TX}(t) \\
    \hat s_\mathrm{WI}(t)
    \end{pmatrix}.
    $$

    From the usage, $\mathbf S$ and $\mathbf P$ are like conjugates. We have the following relation

    $$
    \begin{pmatrix}
    \tilde s_\mathrm{CA}(t) \\
    \tilde s_\mathrm{TX}(t) \\
    \tilde s_\mathrm{WI}(t)
    \end{pmatrix} = \mathbf P \mathbf S {\color{red}\begin{pmatrix}
    s_\mathrm{CA}(t) \\
    s_\mathrm{TX}(t) \\
    s_\mathrm{WI}(t)
    \end{pmatrix}}.
    $$

    It is clear that $\mathbf P \mathbf S$ is identity if we set

    $$
    \begin{pmatrix}
    \tilde s_\mathrm{CA}(t) \\
    \tilde s_\mathrm{TX}(t) \\
    \tilde s_\mathrm{WI}(t)
    \end{pmatrix} = \begin{pmatrix}
    s_\mathrm{CA}(t) \\
    s_\mathrm{TX}(t) \\
    s_\mathrm{WI}(t)
    \end{pmatrix}.
    $$



To generate the coherent forecasts $\tilde{\mathbf y}(t)$, we use [the summing matrix](timeseries-hierarchical.data.md#summing-matrix) to map the bottom level forecasts to base forecasts of all levels[^Hyndman2021][@Rangapuram2021-xi]

$$
\begin{equation}
\tilde{\mathbf y}(t) = \mathbf S\tilde{\mathbf b}(t) = \mathbf S \mathbf P \hat{\mathbf y}(t).
\label{eq:reconciliation-compact-form}
\end{equation}
$$

!!! example "Walmart Sales in Stores"

    We reuse the example of the [Walmart sales data](timeseries-hierarchical.data.md). The base forecasts for all levels are

    $$
    \hat{\mathbf y}(t) = \begin{pmatrix}
    \hat s(t) \\
    \hat s_\mathrm{CA}(t) \\
    \hat s_\mathrm{TX}(t) \\
    \hat s_\mathrm{WI}(t)
    \end{pmatrix}.
    $$

    The simplest mapping to the bottom-level forecasts is

    $$
    \tilde{\mathbf b}(t) = \begin{pmatrix}
    0 & 1 & 0 & 0 \\
    0 & 0 & 1 & 0 \\
    0 & 0 & 0 & 1
    \end{pmatrix}\begin{pmatrix}
    \hat s(t) \\
    \hat s_\mathrm{CA}(t) \\
    \hat s_\mathrm{TX}(t) \\
    \hat s_\mathrm{WI}(t)
    \end{pmatrix}.
    $$

    where

    $$
    \tilde{\mathbf b}(t) = \begin{pmatrix}
    \tilde s_\mathrm{CA}(t) \\
    \tilde s_\mathrm{TX}(t) \\
    \tilde s_\mathrm{WI}(t)
    \end{pmatrix}
    $$

    are the bottom-level forecasts to be transformed into coherent forecasts.

    In this simple method, our mapping matrix $\mathbf P$ can be

    $$
    \mathbf P = \begin{pmatrix}
    0 & 1 & 0 & 0 \\
    0 & 0 & 1 & 0 \\
    0 & 0 & 0 & 1
    \end{pmatrix}.
    $$

    Using this $\mathbf P$, we get

    $$
    \tilde{\mathbf b}(t) = \hat{\mathbf b}(t) = \begin{pmatrix}
    \hat s_\mathrm{CA}(t) \\
    \hat s_\mathrm{TX}(t) \\
    \hat s_\mathrm{WI}(t)
    \end{pmatrix}.
    $$

    The last step is to apply the summing matrix

    $$
    \mathbf S = \begin{pmatrix}
    1 & 1 & 1 \\
    1 & 0 & 0 \\
    0 & 1 & 0 \\
    0 & 0 & 1
    \end{pmatrix},
    $$

    so that

    $$
    \tilde{\mathbf y}(t) = \mathbf S \tilde{\mathbf b}(t) = \begin{pmatrix}
     \hat s_\mathrm{CA}(t) + \hat s_\mathrm{TX}(t) + \hat s_\mathrm{WI}(t) \\
    \hat s_\mathrm{CA}(t) \\
    \hat s_\mathrm{TX}(t) \\
    \hat s_\mathrm{WI}(t)
    \end{pmatrix}.
    $$

    In summary, our coherent forecasts for each level are

    $$
    \begin{align}
    \tilde s (t) &= \hat s_\mathrm{CA}(t) + \hat s_\mathrm{TX}(t) + \hat s_\mathrm{WI}(t) \\
    \tilde s_\mathrm{CA}(t) &= \hat s_\mathrm{CA}(t) \\
    \tilde s_\mathrm{TX}(t) &= \hat s_\mathrm{TX}(t) \\
    \tilde s_\mathrm{WI}(t) &= \hat s_\mathrm{WI}(t).
    \end{align}
    $$

    The $\mathbf P$ we used in this example represents the **bottom-up method**.

    Results like $\tilde s_\mathrm{CA}(t) = \hat s_\mathrm{CA}(t)$ look comfortable but they are not necessary. In other reconciliation methods, these relations might be broken, i.e., $\tilde s_\mathrm{CA}(t) = \hat s_\mathrm{CA}(t)$ may not be true.


!!! note "Component Form"

    We rewrite

    $$
    \tilde{\mathbf y}(t) = \mathbf S \mathbf P \hat{\mathbf y}(t)
    $$

    using the component form

    $$
    \tilde y_i = S_{ij} G_{jk} \hat y_k.
    $$


There is more than one $\mathbf P$ that can map the forecasts to the bottom-level forecasts. Three of the so-called single-level approaches[^Hyndman2021] are bottom-up, top-down, and middle-out[@Rangapuram2021-xi].

Apart from these intuitive methods, Wickramasuriya et al. (2017) proposed the MinT method to find the optimal $\mathbf P$ matrix that gives us the minimal trace of the covariance of the **reconciled forecast error**[@Wickramasuriya2019-cv],

$$
\tilde{\boldsymbol \epsilon} = \mathbf y(t) - \tilde{\mathbf y}(t),
$$

with $\mathbf y$ being the ground truth and $\tilde{\mathbf y}$ being the coherent forecasts. Wickramasuriya et al. (2017) showed that the optimal $\mathbf P$ is

$$
\hat{\mathbf P} = (\mathbf S^T \mathbf W(t)^{-1} \mathbf S)^{-1} (\mathbf S^{T}\mathbf W(t)^{-1}),
$$

where $W_{h} = \mathbb E\left[ \tilde{\boldsymbol \epsilon} \tilde{\boldsymbol \epsilon}^T \right] = \mathbb E \left[ (\mathbf y(t) - \tilde{\mathbf y}(t))(\mathbf y(t) - \tilde{\mathbf y}(t))^T \right]$ is the covariance matrix of the reconciled forecast error.

MinT is easy to calculate but it assumes that the forecasts are unbiased. To relieve this constraint, Van Erven & Cugliari (2013) proposed a game-theoretic method called GTOP[@Van_Erven2015-ht]. In deep learning, Rangapuram et al. (2021) developed an end-to-end model for coherent probabilistic hierarchical forecasts[@Rangapuram2021-xi]. For these advanced topics, we redirect the readers to the original papers.

[^Hyndman2021]: Hyndman, R.J., & Athanasopoulos, G. (2021) Forecasting: principles and practice, 3rd edition, OTexts: Melbourne, Australia. OTexts.com/fpp3. Accessed on 2022-11-27.
