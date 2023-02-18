# Time Series Analysis

Time series analysis is not our focus here. However, it is beneficial to grasp some basic ideas of time series.

## Stationarity

Time series data is stationary if the distribution of the observables do not change[^wiki-stationary-process][^nist-stationarity][^Das2019].

A strict stationary series guarantees the same distribution for a segment $\{x_{i+1}, \cdots, x_{x+k}\}$ and a time-shifted segment $x_{i+1+\Delta}, \cdots, x_{x+k+\Delta}\}$ for integer $\Delta$[^wiki-stationary-process].

A less strict form (WSS) concerns only the mean and autocorrelation[^wiki-stationary-process][^Shalizi2012], i.e.,

$$
\begin{align}
\mathbb E[x_{i+1}] &= \mathbb E[x_{i+\Delta}] \\
\mathbb{Cov}[x_{i+1}, x_{i+k}] &= \mathbb{Cov}[x_{i+1+\Delta}, x_{x+k+\Delta}]
\end{align}
$$

In deep learning, a lot of models require the training data to be I.I.D.[^Schoelkopf2021][^Dawid2022]. The I.I.D. requirement in time series is stationarity.

A stationary time series is clean and pure. However, real-world data is not necessarily stationary, e.g., macroeconomic series data are non-stationary[^Das2019].

## Serial Dependence

Autocorrelation measures the serial dependency of a time series[^wiki-autocorrelation]. By definition, the autocorrelation is the autocovariance normalized by the variance,

$$
\rho = \frac{\mathbb{Cov}[x_t, x_{t+\delta}]}{\mathbb{Var}[x_t]}.
$$

One naive expectation is that the autocorrelation diminishes if $\delta \to \infty$[^Shalizi2012].


[^wiki-stationary-process]: Contributors to Wikimedia projects. Stationary process. In: Wikipedia [Internet]. 18 Sep 2022 [cited 13 Nov 2022]. Available: https://en.wikipedia.org/wiki/Stationary_process
[^nist-stationarity]: 6.4.4.2. Stationarity. In: Engineering Statistics Handbook [Internet]. NIST; [cited 13 Nov 2022]. Available: https://www.itl.nist.gov/div898/handbook/pmc/section4/pmc442.htm#:~:text=Stationarity%20can%20be%20defined%20in,no%20periodic%20fluctuations%20(seasonality).
[^Shalizi2012]: Shalizi C. 36-402, Undergraduate Advanced Data Analysis (2012). In: Undergraduate Advanced Data Analysis [Internet]. 2012 [cited 13 Nov 2022]. Available: https://www.stat.cmu.edu/~cshalizi/uADA/12/
[^Schoelkopf2021]: Schölkopf B, Locatello F, Bauer S, Ke NR, Kalchbrenner N, Goyal A, et al. Toward Causal Representation Learning. Proc IEEE. 2021;109: 612–634. doi:10.1109/JPROC.2021.3058954
[^wiki-autocorrelation]: Contributors to Wikimedia projects. Autocorrelation. In: Wikipedia [Internet]. 10 Nov 2022 [cited 13 Nov 2022]. Available: https://en.wikipedia.org/wiki/Autocorrelation
[^Das2019]: Das P. Econometrics in Theory and Practice. Springer Nature Singapore; [doi:10.1007/978-981-32-9019-8](https://link.springer.com/book/10.1007/978-981-32-9019-8)
[^Dawid2022]: Dawid P, Tewari A. On learnability under general stochastic processes. Harvard Data Science Review. 2022;4. [doi:10.1162/99608f92.dec7d780](https://hdsr.mitpress.mit.edu/pub/qixx99zn/release/1)
