# Hierarchical Time Series Reconciliation

Reconciliation is post-processing method to adjust the forecasts to be coherent. Given forecasts $\hat{\mathbf y}(t)$ for all levels, we use $\mathbf G$ to map them to the base forecasts

$$
\hat{\mathbf b}(t) = \mathbf G \hat{\mathbf y}(t).
$$

To generate the coherent forecasts $\tilde{\mathbf y}(t)$, we use [the summing matrix](timeseries-hierarchical.data.md#summing-matrix) to map the base forecasts to all levels[^Hyndman2021]

$$
\tilde{\mathbf y}(t) = \mathbf S\hat{\mathbf b}(t) = \mathbf S \mathbf G \hat{\mathbf y}(t).
$$

!!! example "Walmart Sales in Stores"

    We reuse the example of the [Walmart sales data](timeseries-hierarchical.data.md). The original forecasts for all levels are

    $$
    \hat{\mathbf y}(t) = \begin{pmatrix}
    \hat s(t) \\
    \hat s_\mathrm{CA}(t) \\
    \hat s_\mathrm{TX}(t) \\
    \hat s_\mathrm{WI}(t)
    \end{pmatrix}.
    $$

    The simplest mapping to the base forecasts is

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


[^Hyndman2021]: Hyndman, R.J., & Athanasopoulos, G. (2021) Forecasting: principles and practice, 3rd edition, OTexts: Melbourne, Australia. OTexts.com/fpp3. Accessed on 2022-11-27.
