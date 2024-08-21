# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py:light
#     text_representation:
#       extension: .py
#       format_name: light
#       format_version: '1.5'
#       jupytext_version: 1.15.2
#   kernelspec:
#     display_name: deep-learning
#     language: python
#     name: deep-learning
# ---

# # Box-Cox Transformation

# +
from typing import Any, Dict

import matplotlib.pyplot as plt
import pandas as pd
from darts import TimeSeries
from darts.dataprocessing.transformers import BoxCox
from darts.datasets import AirPassengersDataset
from darts.utils import statistics as dus

# +
ap_series = AirPassengersDataset().load()

_, ax = plt.subplots(figsize=(10, 6.18))
ap_series.plot(label=f"Air Passenger Original Data", ax=ax)


# +
_, ax = plt.subplots(figsize=(10, 6.18))

boxcox_opt = BoxCox()
ap_boxcox_opt_transformed = boxcox_opt.fit_transform(ap_series)
ap_boxcox_opt_transformed.plot(
    label=f"$\lambda={boxcox_opt._fitted_params[0].item():0.3f}$", ax=ax
)
# -

_, ax = plt.subplots(figsize=(10, 6.18))
lmbda = 0.01
boxcox = BoxCox(lmbda=lmbda)
boxcox_transformed = boxcox.fit_transform(ap_series)
boxcox_transformed.plot(label=f"Box-Cox Transformed Data (lambdax={lmbda})", ax=ax)

# +
_, ax = plt.subplots(figsize=(10, 6.18))

for lmbda in [0.01, 0.1, 0.2]:
    boxcox_lmbda = BoxCox(lmbda=lmbda)
    boxcox_lmbda_transformed = boxcox_lmbda.fit_transform(ap_series)
    boxcox_lmbda_transformed.plot(
        label=f"Box-Cox Transformed Data (lambda={lmbda})", ax=ax
    )


# -

# ## Plot and Check Variance


# +
def var_series(series: TimeSeries, window: int = 36) -> TimeSeries:
    series_rolling_var = TimeSeries.from_dataframe(
        series.pd_dataframe().rolling(window=window).var().dropna()
    )

    return series_rolling_var


def check_stationary(series: TimeSeries) -> Dict[str, Any]:
    return {
        "is_stationary": dus.stationarity_tests(series),
        "kpss": dus.stationarity_test_kpss(series),
        "adf": dus.stationarity_test_adf(series),
    }


# +
boxcox_grid = []
lmbdas = [0.01, 0.1]
rolling_window = 12

for idx, lmbda in enumerate(lmbdas):
    boxcox_lmbda = BoxCox(lmbda=lmbda)
    boxcox_lmbda_transformed = boxcox_lmbda.fit_transform(ap_series)
    var_lmbda_series = var_series(boxcox_lmbda_transformed, window=rolling_window)
    _, ax = plt.subplots(figsize=(10, 6.18))
    var_lmbda_series.plot(
        label=f"Variance (Rolling Window={rolling_window}) (Box-Cox lambda={lmbda})",
        ax=ax,
    )
    plt.show()
# -

_, ax = plt.subplots(figsize=(10, 6.18))
var_ap_series = var_series(ap_series, window=rolling_window)
var_ap_series.plot(label=f"Variance (Rolling Window={rolling_window})", ax=ax)
