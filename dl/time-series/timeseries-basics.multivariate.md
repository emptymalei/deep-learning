# Multivariate Time Series Forecasting


## AR, MA, Integrated, and Vector

```mermaid
flowchart TD

AR --"interdependencies"--> VAR
MA --"add autoregressive"--> ARMA
AR --"add moving average"--> ARMA

ARMA --"difference between values"--> ARIMA
ARMA --"interdependencies"--> VARMA
VAR --"moving average"--> VARMA

ARIMA --"interdependencies"--> VARIMA
VAR --"difference and moving average"--> VARIMA
VARMA --"difference"--> VARIMA
```


[^wu2020]: Wu Z, Pan S, Long G, Jiang J, Chang X, Zhang C. Connecting the Dots: Multivariate Time Series Forecasting with Graph Neural Networks. arXiv [cs.LG]. 2020. Available: http://arxiv.org/abs/2005.11650
