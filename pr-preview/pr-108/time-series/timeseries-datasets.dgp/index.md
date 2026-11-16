# Generating Processes for Time Series

> [!NOTE]
>
> **In one minute**
>
> Time series come from many kinds of data generating processes: simple deterministic ones such as exponential growth and circular motion, pure noise, linear processes such as AR, and processes where several series interact. Simulating them builds intuition for modeling real data.

The data generating processes (DGP) for time series are diverse. For example, in physics, we have all sort of dynamical systems that generates time series data and many dynamics models are formulated based on the time series data. In industries, time series data are often coming from stochastic processes.

We present some data generating processes to help us build up intuition when modeling real-world data.

## Simple Examples of DGP

> [!NOTE]
>
> **Exponential Growth**
>
> Exponential growth is a frequently observed natural and economical phenomenon.
>
> \\\[ y = e^{c \\cdot t} \\\]
>
> ![circular motion series](<https://dl.leima.is/time-series/assets/timeseries-generating-process/exponential-growth-time-series.png>) ![circular motion series](<https://dl.leima.is/time-series/assets/timeseries-generating-process/exponential-growth-time-series-autocorrelation.png>)

> [!NOTE]
>
> **Circular Motion**
>
> The circular motion shows some cyclic patterns.
>
> \\\[ y = \\sin(w \\cdot t) \\\]
>
> ![circular motion series](<https://dl.leima.is/time-series/assets/timeseries-generating-process/circular-motion-time-series.png>)
>
> ![circular motion series autocorrelation](<https://dl.leima.is/time-series/assets/timeseries-generating-process/circular-motion-time-series-autocorrelation.png>)

> [!NOTE]
>
> **Random Gaussian**
>
> Time series can also be noisy Gaussian samples.
>
> ![random gaussian series](<https://dl.leima.is/time-series/assets/timeseries-generating-process/random-guassian-time-series.png>) ![random gaussian series](<https://dl.leima.is/time-series/assets/timeseries-generating-process/random-guassian-time-series-autocorrelation.png>)

## General Linear Processes

A popular model for modeling as well as generating time series is the [autoregressive (AR)](<https://dl.leima.is/time-series/timeseries-basics.ar/index.md>) model. An AR is formulated as

\\\[ x\_t = \\phi\_0 + \\phi\_1 x\_{t-1} + \\epsilon\_t. \\\]

> [!NOTE]
>
> **AR(p) and the Lag Operator**
>
> A general autoregressive model of p-th order is
>
> \\\[ x\_t = \\sum\_l \\phi \_i x\_{t-i} + \\epsilon\_t, \\\]
>
> where \\(l\\) is the lag.
>
> Define a lag operator \\(\\hat L\\) with \\(\\hat L x\_t = x\_{t-1}\\). The definition can also be rewritten using the lag operator
>
> \\\[ x\_t = \\sum\_l \\phi \_i {\\hat L}^i x\_{t} + \\epsilon\_t. \\\]

We write down each time step in the following table.

| \\(t\\) | \\(x\_t\\) |
| :---: | :---: |
| 0 | \\(y\_0\\) |
| 1 | \\(\\phi\_0 + \\phi\_1 y\_0 + \\epsilon\_1\\) |
| 2 | \\(\\phi\_0 + \\phi\_1 (\\phi\_0 + \\phi\_1 y\_0 + \\epsilon\_1) + \\epsilon\_2 = \\phi\_0 (1 + \\phi\_1) + \\phi\_1^2 y\_0 + \\phi\_1\\epsilon\_1 + \\epsilon\_2\\) |
| 3 | \\(\\phi\_0 + \\phi\_1 (\\phi\_0 + \\phi\_1\\phi\_0 + \\phi\_1^2 y\_0 + \\phi\_1\\epsilon\_1 + \\epsilon\_2) + \\epsilon\_3 = \\phi\_0(1 + \\phi\_1 + \\phi\_1^2) + \\phi\_1^3 y\_0 + \\phi\_1^2\\epsilon\_1 + \\phi\_1\\epsilon\_2 + \\epsilon\_3\\) |
| ... | ... |
| \\(t\\) | \\(\\phi\_0 \\sum\_{i=0}^{t-1} \\phi\_1^i + \\phi\_1^t y\_0 + \\sum\_{i=1}^{t-1} \\phi\_1^{t-i} \\epsilon\_{i}\\) |

We have found a new formula for AR(1), i.e.

\\\[ x\_t = \\phi\_0 \\sum\_{i=0}^{t-1} \\phi\_1^i + \\phi\_1^t y\_0 + \\sum\_{i=1}^{t-1} \\phi\_1^{t-i} \\epsilon\_{i}, \\\]

which is very similar to a general linear process[^/time-series/timeseries-datasets.dgp/#Das2019]

\\\[ x\_t - \\mu = \\sum\_{i=0}^\\tau \\alpha\_i \\epsilon\_{t-i}. \\\]

The general linear process is the Taylor expansion of an arbitrary DGP \\(x\_t = \\operatorname{DGP}(\\epsilon\_t, ...)\\)[^/time-series/timeseries-datasets.dgp/#Das2019].

## Interactions between Series

The interactions between the series can be modeled as explicit interactions, e.g., many spiking neurons, or through hidden variables, e.g., hidden state model[^/time-series/timeseries-datasets.dgp/#hmm]. Among these models, [Vector Autoregressive model, aka VAR,](<https://dl.leima.is/time-series/timeseries-basics.var/index.md>) is a simple but popular model.

---

**Related:** [AR](<https://dl.leima.is/time-series/timeseries-basics.ar/index.md>) · [VAR](<https://dl.leima.is/time-series/timeseries-basics.var/index.md>) · [Time Series Data Generating Process: Langevin Equation](<https://dl.leima.is/time-series/timeseries-datasets.dgp.langevin/index.md>)

[^/time-series/timeseries-datasets.dgp/#Das2019]: Das P. Econometrics in Theory and Practice. Springer Nature Singapore; [doi:10.1007/978-981-32-9019-8](<https://link.springer.com/book/10.1007/978-981-32-9019-8>)

[^/time-series/timeseries-datasets.dgp/#hmm]: Contributors to Wikimedia projects. Hidden Markov model. In: Wikipedia \[Internet\]. 22 Oct 2022 \[cited 22 Nov 2022\]. Available: [https://en.wikipedia.org/wiki/Hidden\_Markov\_model](<https://en.wikipedia.org/wiki/Hidden_Markov_model >)
