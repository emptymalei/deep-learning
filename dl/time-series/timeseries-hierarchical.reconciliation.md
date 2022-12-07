# Hierarchical Time Series Reconciliation

Reconciliation is a post-processing method to adjust the forecasts to be coherent. Given **base forecasts** $\hat{\mathbf y}(t)$ for all levels which were forecasted independently, we use $\mathbf P$ to map them to the bottom level forecasts

\begin{equation}
\hat{\mathbf b}(t) = \mathbf P \hat{\mathbf y}(t).
\end{equation}

To generate the coherent forecasts $\tilde{\mathbf y}(t)$, we use [the summing matrix](timeseries-hierarchical.data.md#summing-matrix) to map the bottom level forecasts to base forecasts of all levels[^Hyndman2021][@Rangapuram2021-xi]

$$
\begin{equation}
\tilde{\mathbf y}(t) = \mathbf S\hat{\mathbf b}(t) = \mathbf S \mathbf P \hat{\mathbf y}(t).
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

    The bottom level forecasts are

    $$
    \hat{\mathbf b}(t) = \begin{pmatrix}
    \hat s_\mathrm{CA}(t) \\
    \hat s_\mathrm{TX}(t) \\
    \hat s_\mathrm{WI}(t)
    \end{pmatrix}.
    $$

    The simplest mapping to the bottom level forecasts is

    $$
    \hat{\mathbf b}(t) = \begin{pmatrix}
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

    In this simple method, our mapping matrix $\mathbf P$ is

    $$
    \mathbf P = \begin{pmatrix}
    0 & 1 & 0 & 0 \\
    0 & 0 & 1 & 0 \\
    0 & 0 & 0 & 1
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
    \tilde{\mathbf y}(t) = \mathbf S \hat{\mathbf b}(t) = \begin{pmatrix}
     \hat s_\mathrm{CA}(t) + \hat s_\mathrm{TX}(t) + \hat s_\mathrm{WI}(t) \\
    \hat s_\mathrm{CA}(t) \\
    \hat s_\mathrm{TX}(t) \\
    \hat s_\mathrm{WI}(t)
    \end{pmatrix}.
    $$

    In summary, our coherent forecasts for each levels are

    $$
    \begin{align}
    \tilde s (t) &= \hat s_\mathrm{CA}(t) + \hat s_\mathrm{TX}(t) + \hat s_\mathrm{WI}(t) \\
    \tilde s_\mathrm{CA}(t) &= \hat s_\mathrm{CA}(t) \\
    \tilde s_\mathrm{TX}(t) &= \hat s_\mathrm{TX}(t) \\
    \tilde s_\mathrm{WI}(t) &= \hat s_\mathrm{WI}(t).
    \end{align}
    $$

    The $\mathbf P$ we used in this example represents the **bottom-up method**.

    Results like $\tilde s_\mathrm{CA}(t) = \hat s_\mathrm{CA}(t)$ looks comfortable but they are not necessary. In other reconciliation methods, these relations might be broken, i.e., $\tilde s_\mathrm{CA}(t) = \hat s_\mathrm{CA}(t)$ may not be true.


!!! note "Component Form"

    We rewrite

    $$
    \tilde{\mathbf y}(t) = \mathbf S \mathbf P \hat{\mathbf y}(t)
    $$

    using the component form

    $$
    \tilde y_i = S_{ij} G_{jk} \hat y_k.
    $$


There are more than one $\mathbf P$ that can map the forecasts to the bottom level forecasts. Three of the so-called single level approaches[^Hyndman2021] are bottom-up, top-down, and middle-out[@Rangapuram2021-xi].

Apart from these intuitive methods, Wickramasuriya et al. (2017) proposed the MinT method to find the optimal $\mathbf G$ matrix that gives us the minimal variance the **reconciled forecast errors**[@Wickramasuriya2019-cv],

$$
\tilde{\boldsymbol \epsilon} = \mathbf y(t) - \tilde{\mathbf y}(t),
$$

with $\mathbf y$ being the ground truth and $\tilde{\mathbf y}$ being the coherent forecasts. Wickramasuriya et al. (2017) showed the optimal $\mathbf P$ is

$$
\hat{\mathbf P} = (\mathbf S^T \mathbf W(t)^{-1} \mathbf S)^{-1} (\mathbf S^{T}\mathbf W(t)^{-1}),
$$

where $W_{h} = \mathbb E\left[ \tilde{\boldsymbol \epsilon} \tilde{\boldsymbol \epsilon}^T \right] = \mathbb E \left[ (\mathbf y(t) - \tilde{\mathbf y}(t))(\mathbf y(t) - \tilde{\mathbf y}(t))^T \right]$.

MinT is easy to calculate but it assumes the forecasts are unbiased. To solve this problem Van Erven & Cugliari (2013) proposed a geme theoretic method called GTOP[@Van_Erven2015-ht]. Rangapuram et al. (2021) developed an end-to-end model for coherent probabilistic hierarchical forecasts[@Rangapuram2021-xi]. For these advanced topics, we redirect the readers to the original papers.

[^Hyndman2021]: Hyndman, R.J., & Athanasopoulos, G. (2021) Forecasting: principles and practice, 3rd edition, OTexts: Melbourne, Australia. OTexts.com/fpp3. Accessed on 2022-11-27.
