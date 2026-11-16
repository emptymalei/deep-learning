# Time Series Analysis

> [!NOTE]
>
> **In one minute**
>
> Two ideas from time series analysis that the rest of the book leans on: stationarity (the distribution of the series doesn't change over time, with a weaker form that only looks at the mean and autocorrelation) and serial dependence, measured by autocorrelation. It also notes how terminology differs between fields, such as panel data versus multivariate time series.

Time series analysis is not our focus here. However, it is beneficial to grasp some basic ideas of time series.

## Stationarity

Time series data is stationary if the distribution of the observables do not change[^/time-series/timeseries-data.analysis/#wiki-stationary-process][^/time-series/timeseries-data.analysis/#nist-stationarity][^/time-series/timeseries-data.analysis/#Das2019].

A strict stationary series guarantees the same distribution for a segment \\(\\{x\_{i+1}, \\cdots, x\_{x+k}\\}\\) and a time-shifted segment \\(x\_{i+1+\\Delta}, \\cdots, x\_{x+k+\\Delta}\\}\\) for integer \\(\\Delta\\)[^/time-series/timeseries-data.analysis/#wiki-stationary-process].

A less strict form (WSS) concerns only the mean and autocorrelation[^/time-series/timeseries-data.analysis/#wiki-stationary-process][^/time-series/timeseries-data.analysis/#Shalizi2012], i.e.,

\\\[ \\begin{align} \\mathbb E\[x\_{i+1}\] &amp;= \\mathbb E\[x\_{i+\\Delta}\] \\\\ \\mathbb{Cov}\[x\_{i+1}, x\_{i+k}\] &amp;= \\mathbb{Cov}\[x\_{i+1+\\Delta}, x\_{x+k+\\Delta}\] \\end{align} \\\]

In deep learning, a lot of models require the training data to be I.I.D.[^/time-series/timeseries-data.analysis/#Schoelkopf2021][^/time-series/timeseries-data.analysis/#Dawid2022]. The I.I.D. requirement in time series is stationarity.

A stationary time series is clean and pure. However, real-world data is not necessarily stationary, e.g., macroeconomic series data are non-stationary[^/time-series/timeseries-data.analysis/#Das2019].

## Serial Dependence

Autocorrelation measures the serial dependency of a time series[^/time-series/timeseries-data.analysis/#wiki-autocorrelation]. By definition, the autocorrelation is the autocovariance normalized by the variance,

\\\[ \\rho = \\frac{\\mathbb{Cov}\[x\_t, x\_{t+\\delta}\]}{\\mathbb{Var}\[x\_t\]}. \\\]

One naive expectation is that the autocorrelation diminishes if \\(\\delta \\to \\infty\\)[^/time-series/timeseries-data.analysis/#Shalizi2012].

## Terminology

Terminologies for time series data may be different in different fields[^/time-series/timeseries-data.analysis/#Hyndman2020]. For example, we may encounter the term "panel data" in econometrics, which is the same as "multivariate time series" in "data science".

> [!NOTE]
>
> **Panel Data**
>
> Panel data is multivariate time series data,
>
> \\\[ \\mathbf y\_t \\to y\_{it}. \\\]
>
> | time | variable \\(y\_1\\) | variable \\(y\_2\\) | variable \\(y\_3\\) |
> | --- | --- | --- | --- |
> | \\(t\_1\\) | \\(y\_{11}\\) | \\(y\_{21}\\) | \\(y\_{31}\\) |
> | \\(t\_2\\) | \\(y\_{12}\\) | \\(y\_{22}\\) | \\(y\_{32}\\) |
> | \\(t\_3\\) | \\(y\_{13}\\) | \\(y\_{23}\\) | \\(y\_{33}\\) |
> | \\(t\_4\\) | \\(y\_{14}\\) | \\(y\_{24}\\) | \\(y\_{34}\\) |
> | \\(t\_5\\) | \\(y\_{15}\\) | \\(y\_{25}\\) | \\(y\_{35}\\) |

1. [doi:10.1162/99608f92.dec7d780](<https://hdsr.mitpress.mit.edu/pub/qixx99zn/release/1>)

---

**Related:** [Box-Cox Transformation](<https://dl.leima.is/time-series/timeseries-data.box-cox/index.md>) · [AR](<https://dl.leima.is/time-series/timeseries-basics.ar/index.md>)

[^/time-series/timeseries-data.analysis/#wiki-stationary-process]: Contributors to Wikimedia projects. Stationary process. In: Wikipedia \[Internet\]. 18 Sep 2022 \[cited 13 Nov 2022\]. Available: [https://en.wikipedia.org/wiki/Stationary\_process](<https://en.wikipedia.org/wiki/Stationary_process >)

[^/time-series/timeseries-data.analysis/#nist-stationarity]: 6.4.4.2. Stationarity. In: Engineering Statistics Handbook \[Internet\]. NIST; \[cited 13 Nov 2022\]. Available: [https://www.itl.nist.gov/div898/handbook/pmc/section4/pmc442.htm\#:\~:text=Stationarity%20can%20be%20defined%20in,no%20periodic%20fluctuations%20(seasonality).](<https://www.itl.nist.gov/div898/handbook/pmc/section4/pmc442.htm#:~:text=Stationarity can be defined in,no periodic fluctuations (seasonality). >)

[^/time-series/timeseries-data.analysis/#Shalizi2012]: Shalizi C. 36-402, Undergraduate Advanced Data Analysis (2012). In: Undergraduate Advanced Data Analysis \[Internet\]. 2012 \[cited 13 Nov 2022\]. Available: [https://www.stat.cmu.edu/\~cshalizi/uADA/12/](<https://www.stat.cmu.edu/~cshalizi/uADA/12/ >)

[^/time-series/timeseries-data.analysis/#Schoelkopf2021]: Schölkopf B, Locatello F, Bauer S, Ke NR, Kalchbrenner N, Goyal A, et al. Toward Causal Representation Learning. Proc IEEE. 2021;109: 612–634. doi:10.1109/JPROC.2021.3058954

[^/time-series/timeseries-data.analysis/#wiki-autocorrelation]: Contributors to Wikimedia projects. Autocorrelation. In: Wikipedia \[Internet\]. 10 Nov 2022 \[cited 13 Nov 2022\]. Available: [https://en.wikipedia.org/wiki/Autocorrelation](<https://en.wikipedia.org/wiki/Autocorrelation >)

[^/time-series/timeseries-data.analysis/#Das2019]: Das P. Econometrics in Theory and Practice. Springer Nature Singapore; [doi:10.1007/978-981-32-9019-8](<https://link.springer.com/book/10.1007/978-981-32-9019-8>)

[^/time-series/timeseries-data.analysis/#Dawid2022]: Dawid P, Tewari A. On learnability under general stochastic processes. Harvard Data Science Review. 2022;

[^/time-series/timeseries-data.analysis/#Hyndman2020]: Hyndman R. Rob J Hyndman - Terminology matters. In: Rob J Hyndman \[Internet\]. 26 Jun 2020 \[cited 9 Nov 2023\]. Available: [https://robjhyndman.com/hyndsight/terminology-matters/\#same-concept-different-terminology](<https://robjhyndman.com/hyndsight/terminology-matters/index.md#same-concept-different-terminology >)
