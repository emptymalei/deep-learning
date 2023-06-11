# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py:light
#     text_representation:
#       extension: .py
#       format_name: light
#       format_version: '1.5'
#       jupytext_version: 1.14.5
#   kernelspec:
#     display_name: deep-learning
#     language: python
#     name: deep-learning
# ---

# # Forecasting with Trees Using Darts
#
# In this notebook, we explore some basic ideas of how to forecast using trees with the help of the package called [Darts](https://github.com/unit8co/darts).

# +
import pandas as pd
import numpy as np

import matplotlib.pyplot as plt
import seaborn as sns

from darts import TimeSeries
from darts.datasets import AirPassengersDataset
import darts.utils as du
from darts import metrics

from darts.models import LightGBMModel, RandomForest, NaiveDrift
from darts.dataprocessing.transformers import BoxCox

from sklearn.linear_model import LinearRegression

from typing import List, Dict, Callable

# -

# ## Following the Darts Official Tutorial
#
# Darts provides a [tutorial here](https://unit8co.github.io/darts/quickstart/00-quickstart.html) to help the users get started. Here we replicate some of them to provide a minimal working example for tree-based models.

darts_air_passenger_series = AirPassengersDataset().load()
darts_air_passenger_series.plot()

darts_air_passenger_series

# From the outputs, we see that the time series dataset contains montly data for 144 months.

train_series_length = 120
test_series_length = len(darts_air_passenger_series) - train_series_length
train_series_length, test_series_length

# +
(
    darts_air_passenger_train,
    darts_air_passenger_test,
) = darts_air_passenger_series.split_before(train_series_length)

darts_air_passenger_train.plot(label="Training Data")
darts_air_passenger_test.plot(label="Test Data")
# -

# ### First Random Forest Model

ap_horizon = len(darts_air_passenger_test)
ap_rf_params = dict(lags=52, output_chunk_length=ap_horizon)

rf_ap = RandomForest(**ap_rf_params)

rf_ap.fit(darts_air_passenger_train)

# To observe how the model performs on the training data, we predict a time range that has already seen by the model during training.
#

darts_air_passenger_train.drop_after(
    darts_air_passenger_train.time_index[-ap_horizon]
).plot(label="Prediction Input")
darts_air_passenger_train.drop_before(
    darts_air_passenger_train.time_index[-ap_horizon]
).plot(label="True Values")
rf_ap.predict(
    n=ap_horizon,
    series=darts_air_passenger_train.drop_after(
        darts_air_passenger_train.time_index[-ap_horizon]
    ),
).plot(label="Predictions (In-sample)", linestyle="--")

# The predictions looks amazing. However, we all know that tree-based models are not good at out of sample extrapolations. In our case, the trend of the time series may cause some problems. To test this idea, we plot out the predictions for the test date range.

darts_air_passenger_train.plot(label="Train")
darts_air_passenger_test.plot(label="Test")
pred_rf_ap = rf_ap.predict(n=ap_horizon)
pred_rf_ap.plot(label="Prediction", linestyle="--")

# ### Detrending Helps

# We train the same model but with the detrended dataset, and reconstruct the predictions using the trend. This method demonstrate that detrended data is easier for random forest.

# +
fig, ax = plt.subplots(figsize=(10, 6.18))

sns.histplot(
    darts_air_passenger_train.pd_dataframe(),
    x="#Passengers",
    kde=True,
    binwidth=50,
    binrange=(0, 700),
    label="Training Distribution",
    stat="probability",
    ax=ax,
)
sns.histplot(
    darts_air_passenger_test.pd_dataframe(),
    x="#Passengers",
    kde=True,
    binwidth=50,
    binrange=(0, 700),
    label="Test Distribution",
    stat="probability",
    color="r",
    ax=ax,
)

ax.set_xlabel("# Passengers")

plt.legend()

# +
(
    darts_air_passenger_trend,
    darts_air_passenger_seasonal,
) = du.statistics.extract_trend_and_seasonality(
    darts_air_passenger_series,
    #     model=du.utils.ModelMode.ADDITIVE,
    #     method="STL"
)

darts_air_passenger_series.plot()
darts_air_passenger_trend.plot()
(darts_air_passenger_trend * darts_air_passenger_seasonal).plot()
# -

(
    darts_air_passenger_seasonal_train,
    darts_air_passenger_seasonal_test,
) = darts_air_passenger_seasonal.split_before(120)
darts_air_passenger_seasonal_train.plot(label="Seasonal Component Train")
darts_air_passenger_seasonal_test.plot(label="Seasonal Component Test")

darts_air_passenger_seasonal_test.pd_dataframe()

# +
fig, ax = plt.subplots(figsize=(10, 6.18))

sns.histplot(
    darts_air_passenger_seasonal_train.pd_dataframe(),
    x="0",
    kde=True,
    binwidth=0.1,
    binrange=(0.7, 1.3),
    label="Training Distribution",
    stat="probability",
    #     fill=False,
    ax=ax,
)
sns.histplot(
    darts_air_passenger_seasonal_test.pd_dataframe(),
    x="0",
    kde=True,
    binwidth=0.1,
    binrange=(0.7, 1.3),
    label="Test Distribution",
    stat="probability",
    color="r",
    #     fill=False,
    ax=ax,
)

ax.set_xlabel("# Passengers")

plt.legend()
# -

rf_ap_seasonal = RandomForest(**ap_rf_params)

rf_ap_seasonal.fit(darts_air_passenger_seasonal_train)

darts_air_passenger_train.plot(label="Train")
darts_air_passenger_test.plot(label="Test")
pred_rf_ap_seasonal = rf_ap_seasonal.predict(
    n=ap_horizon
) * darts_air_passenger_trend.drop_before(119)
pred_rf_ap_seasonal.plot(label="Trend * Predicted Seasonal Component", linestyle="--")

# This indiates that the performance of trees on out of sample predictions if we only predict on the cycle part of the series. In a real world case, however, we have to predict the trend accurately for this to work. To better reconstruct the trend, there are also tricks like [Box-Cox transformations](../time-series/timeseries-data.box-cox.md).

# ## Train, Test, and Metrics

# It is not easy to determine a best model simply looking at the charts. We need some metrics.

air_passenger_boxcox = BoxCox()
darts_air_passenger_train_boxcox = air_passenger_boxcox.fit_transform(
    darts_air_passenger_train
)
darts_air_passenger_test_boxcox = air_passenger_boxcox.transform(
    darts_air_passenger_test
)
darts_air_passenger_train_boxcox.plot(label="Train (Box-Cox Transformed)")
darts_air_passenger_test_boxcox.plot(label="Test (Box-Cox Transformed)")


# +
def linear_trend_model(series: TimeSeries) -> LinearRegression:
    """Fit a linear trend of the series. This can be used to find the linear
    model using training data.

    :param series: training timeseries
    """
    positional_index_start = 0
    series_trend, _ = du.statistics.extract_trend_and_seasonality(series)
    model = LinearRegression()
    length = len(series_trend)
    model.fit(
        np.arange(positional_index_start, positional_index_start + length).reshape(
            length, 1
        ),
        series_trend.values(),
    )

    return model


def find_linear_trend(
    series: TimeSeries, model, positional_index_start: int = 0
) -> TimeSeries:
    """Using the fitted linear model to find or extrapolate the linear trend.

    :param series: train or test timeseries
    :param model: LinearRegression model that has `predict` method
    :param positional_index_start: the position of the first value in the original timeseries.
    """
    length = len(series)
    linear_preds = model.predict(
        np.arange(positional_index_start, positional_index_start + length).reshape(
            length, 1
        )
    ).squeeze()

    dataframe = pd.DataFrame(
        {"date": series.time_index, "# Passengers": linear_preds}
    ).set_index("date")

    return TimeSeries.from_dataframe(dataframe)


# -

ap_trend_lm = linear_trend_model(darts_air_passenger_train_boxcox)
ap_trend_lm

# +
ap_trend_linear_train = find_linear_trend(
    model=ap_trend_lm, series=darts_air_passenger_train_boxcox
)
ap_trend_linear_test = find_linear_trend(
    model=ap_trend_lm,
    series=darts_air_passenger_test_boxcox,
    positional_index_start=train_history_length,
)

darts_air_passenger_train_boxcox.plot(label="Train")
ap_trend_linear_train.plot(label="Linear Trend (Train)")
darts_air_passenger_test_boxcox.plot(label="Test")
ap_trend_linear_test.plot(label="Linear Trend (Test)")
# -

darts_air_passenger_train_transformed = (
    darts_air_passenger_train_boxcox - ap_trend_linear_train
)
darts_air_passenger_train_transformed.plot()

rf_bc_lt = RandomForest(**ap_rf_params)
rf_bc_lt.fit(darts_air_passenger_train_transformed)

darts_air_passenger_train.plot()
darts_air_passenger_test.plot()
pred_rf_bc_lt = boxcox.inverse_transform(
    rf_bc_lt.predict(n=ap_horizon) + ap_trend_linear_test
)
pred_rf_bc_lt.plot(label="Box-Cox + Linear Detrend Predictions", linestyle="--")

# ### Metrics

darts_air_passenger_test.plot(label="Test")
pred_rf_ap.plot(label="Simple RF", linestyle="--")
pred_rf_ap_seasonal.plot(label="RF on Global Detrended Data (Cheating)", linestyle="--")
pred_rf_bc_lt.plot(label="Box-Cox + Linear Detrend", linestyle="--")

benchmark_metrics = [
    metrics.mae,
    metrics.mape,
    metrics.mse,
    metrics.rmse,
    metrics.smape,
]


def benchmark_predictions(
    series_true: TimeSeries,
    series_prediction: TimeSeries,
    metrics: List[Callable],
    experiment_id: str,
) -> Dict:
    results = []
    for m in benchmark_metrics:
        results.append(
            {
                "metric": f"{m.__name__}",
                "value": m(series_true, series_prediction),
                "experiment": experiment_id,
            }
        )

    return results


# +
benchmark_results = []

for i, pred in zip(
    ["simple_rf", "detrended_cheating", "boxcox_linear_trend"],
    [pred_rf_ap, pred_rf_ap_seasonal, pred_rf_bc_lt],
):
    benchmark_results += benchmark_predictions(
        series_true=darts_air_passenger_test,
        series_prediction=pred,
        metrics=benchmark_metrics,
        experiment_id=i,
    )

df_benchmark_metrics = pd.DataFrame(benchmark_results)
df_benchmark_metrics

# +
metric_chart_grid = sns.FacetGrid(
    df_benchmark_metrics,
    col="metric",
    hue="metric",
    col_wrap=2,
    height=4,
    aspect=1 / 0.618,
    sharey=False,
)

metric_chart_grid.map(
    sns.barplot, "experiment", "value", order=df_benchmark_metrics.experiment.unique()
)
# for axes in metric_chart_grid.axes.flat:
#     _ = axes.set_xticklabels(axes.get_xticklabels(), rotation=90)
# metric_chart_grid.fig.tight_layout(w_pad=1)
# -
