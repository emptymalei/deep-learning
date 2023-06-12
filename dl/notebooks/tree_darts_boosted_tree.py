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

# + [markdown] id="nBR1ou1dwYGl"
# # Forecasting with Boosted Trees Using Darts
#
# In this notebook, we explore some basic ideas of how to forecast using trees with the help of the package called [Darts](https://github.com/unit8co/darts).

# + id="IyjC2kJDwYGn"
import pandas as pd
import numpy as np

import matplotlib.pyplot as plt
import seaborn as sns

from darts import TimeSeries
from darts.datasets import AirPassengersDataset
import darts.utils as du
from darts import metrics

from darts.models import LightGBMModel, NaiveDrift
from darts.dataprocessing.transformers import BoxCox

from sklearn.linear_model import LinearRegression

from typing import List, Dict, Callable

# + [markdown] id="3cI3-YSIwYGo"
# ## Following the Darts Official Tutorial
#
# Darts provides a [tutorial here](https://unit8co.github.io/darts/quickstart/00-quickstart.html) to help the users get started. Here we replicate some of them to provide a minimal working example for tree-based models.

# + colab={"base_uri": "https://localhost:8080/", "height": 462} id="e2v4VEmUwYGo" outputId="ff9f9ad0-0f05-4267-9078-c2f90ba17c97"
darts_air_passenger_series = AirPassengersDataset().load()
darts_air_passenger_series.plot()

# + colab={"base_uri": "https://localhost:8080/", "height": 924} id="wmMLvM1KwYGp" outputId="a0702970-5438-4be8-eba0-855bb7976842"
darts_air_passenger_series

# + [markdown] id="RsS5saFxwYGp"
# From the outputs, we see that the time series dataset contains montly data for 144 months.

# + colab={"base_uri": "https://localhost:8080/"} id="O60cbKA_wYGp" outputId="16e74bc4-580d-4826-f259-c0be4928df00"
train_series_length = 120
test_series_length = len(darts_air_passenger_series) - train_series_length
train_series_length, test_series_length

# + colab={"base_uri": "https://localhost:8080/", "height": 462} id="2ItAkXqFwYGp" outputId="19c3b9d0-e875-4985-cd0d-68a013a4574f"
(
    darts_air_passenger_train,
    darts_air_passenger_test,
) = darts_air_passenger_series.split_before(train_series_length)

darts_air_passenger_train.plot(label="Training Data")
darts_air_passenger_test.plot(label="Test Data")

# + [markdown] id="4lKC6b3SwYGq"
# ### First GBDT Model

# + id="rXKC8hkQwYGq"
ap_horizon = len(darts_air_passenger_test)
ap_gbdt_params = dict(lags=52, output_chunk_length=ap_horizon)

# + id="ULfLRR7owYGq"
gbdt_ap = LightGBMModel(**ap_gbdt_params)

# + colab={"base_uri": "https://localhost:8080/"} id="D517yhtKwYGq" outputId="753af900-cd8c-4a4f-d159-8dcd30a06269"
gbdt_ap.fit(darts_air_passenger_train)

# + [markdown] id="lfZePmaXMNfH"
# Insample predictions: We plot out the predictions for the last 24 days in the training data.

# + colab={"base_uri": "https://localhost:8080/", "height": 462} id="q56fp4wpwYGq" outputId="fb21e5bb-4095-4ecf-fcd4-61e0b0895580"
darts_air_passenger_train.drop_after(
    darts_air_passenger_train.time_index[-ap_horizon]
).plot(label="Prediction Input")
darts_air_passenger_train.drop_before(
    darts_air_passenger_train.time_index[-ap_horizon]
).plot(label="True Values")
gbdt_ap.predict(
    n=ap_horizon,
    series=darts_air_passenger_train.drop_after(
        darts_air_passenger_train.time_index[-ap_horizon]
    ),
).plot(label="Predictions (In-sample)", linestyle="--")

# + [markdown] id="pW8snuHhMJKY"
# To observe the actual performance, we plot out the predictions of the test dates.

# + colab={"base_uri": "https://localhost:8080/", "height": 462} id="yhYnrO0xwYGr" outputId="b2a2d0bb-39ee-4a90-f3b4-73914ab9a828"
darts_air_passenger_train.plot(label="Train")
darts_air_passenger_test.plot(label="Test")
pred_gbdt_ap = gbdt_ap.predict(n=ap_horizon)
pred_gbdt_ap.plot(label="Prediction", linestyle="--")

# + [markdown] id="bgQRMT2ZwYGr"
# ### Detrending Helps

# + [markdown] id="VWVn6lFUwYGr"
# We train the same model but with the detrended dataset, and reconstruct the predictions using the trend.

# + colab={"base_uri": "https://localhost:8080/", "height": 462} id="wd9-x7I5wYGr" outputId="9ab7eca2-f35c-4e63-ca61-5f890c3d3ab8"
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

# + colab={"base_uri": "https://localhost:8080/", "height": 462} id="5gpXnTsSwYGr" outputId="c6b5756d-4dc6-47e6-e39a-b562455f5f28"
(
    darts_air_passenger_seasonal_train,
    darts_air_passenger_seasonal_test,
) = darts_air_passenger_seasonal.split_before(120)
darts_air_passenger_seasonal_train.plot(label="Seasonal Component Train")
darts_air_passenger_seasonal_test.plot(label="Seasonal Component Test")

# + colab={"base_uri": "https://localhost:8080/", "height": 578} id="e5yMH7wCwYGs" outputId="00f0578b-cd3d-4c9c-b15b-4f3040318317"
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

# + id="6n0X8Rw1wYGs"
gbdt_ap_seasonal = LightGBMModel(**ap_gbdt_params)

# + colab={"base_uri": "https://localhost:8080/"} id="pFHhoZAKwYGs" outputId="50459119-bafe-4696-d608-21616c8350a1"
gbdt_ap_seasonal.fit(darts_air_passenger_seasonal_train)

# + colab={"base_uri": "https://localhost:8080/", "height": 462} id="x-Wluyz6wYGs" outputId="d275a304-5128-434c-d706-409de13cb046"
darts_air_passenger_train.plot(label="Train")
darts_air_passenger_test.plot(label="Test")
pred_gbdt_ap_seasonal = gbdt_ap_seasonal.predict(
    n=ap_horizon
) * darts_air_passenger_trend.drop_before(119)
pred_gbdt_ap_seasonal.plot(label="Trend * Predicted Seasonal Component", linestyle="--")

# + [markdown] id="m2pPeXiSwYGs"
# This indiates that the performance of trees on out of sample predictions if we only predict on the cycle part of the series. In a real world case, however, we have to predict the trend accurately for this to work. To better reconstruct the trend, there are also tricks like [Box-Cox transformations](../time-series/timeseries-data.box-cox.md).

# + [markdown] id="IpXytbBWwYGs"
# ## Train, Test, and Metrics

# + [markdown] id="7NhjgZnzwYGs"
# It is not easy to determine a best model simply looking at the charts. We need some metrics.

# + colab={"base_uri": "https://localhost:8080/", "height": 462} id="OFHbbcMQwYGs" outputId="f6e1963c-7d4c-44f4-def2-8ef7b3c8e7f9"
air_passenger_boxcox = BoxCox()
darts_air_passenger_train_boxcox = air_passenger_boxcox.fit_transform(
    darts_air_passenger_train
)
darts_air_passenger_test_boxcox = air_passenger_boxcox.transform(
    darts_air_passenger_test
)
darts_air_passenger_train_boxcox.plot(label="Train (Box-Cox Transformed)")
darts_air_passenger_test_boxcox.plot(label="Test (Box-Cox Transformed)")


# + id="RwqpEVGCwYGs"
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


# + colab={"base_uri": "https://localhost:8080/", "height": 75} id="SXo0V-CwwYGt" outputId="7b73798d-7211-40b8-a1b7-7530b4573efa"
ap_trend_lm = linear_trend_model(darts_air_passenger_train_boxcox)
ap_trend_lm

# + colab={"base_uri": "https://localhost:8080/", "height": 462} id="hmgP-fHQwYGt" outputId="c18b4b38-6c7b-4e39-a3d0-ff86ba5d63a2"
ap_trend_linear_train = find_linear_trend(
    model=ap_trend_lm, series=darts_air_passenger_train_boxcox
)
ap_trend_linear_test = find_linear_trend(
    model=ap_trend_lm,
    series=darts_air_passenger_test_boxcox,
    positional_index_start=train_series_length,
)

darts_air_passenger_train_boxcox.plot(label="Train")
ap_trend_linear_train.plot(label="Linear Trend (Train)")
darts_air_passenger_test_boxcox.plot(label="Test")
ap_trend_linear_test.plot(label="Linear Trend (Test)")

# + colab={"base_uri": "https://localhost:8080/", "height": 462} id="i-M8BwoFwYGt" outputId="49e076c9-2ed2-43a7-d2d2-dd8c277c8e5f"
darts_air_passenger_train_transformed = (
    darts_air_passenger_train_boxcox - ap_trend_linear_train
)
darts_air_passenger_train_transformed.plot()

# + colab={"base_uri": "https://localhost:8080/"} id="3GIVYJidwYGt" outputId="7d06536b-307a-4056-f57c-512c97a0ed46"
gbdt_bc_lt = LightGBMModel(**ap_gbdt_params)
gbdt_bc_lt.fit(darts_air_passenger_train_transformed)

# + colab={"base_uri": "https://localhost:8080/", "height": 462} id="pMWLdQk7wYGt" outputId="46906de1-366a-4e4e-eac9-a9f5ae4cbfb6"
darts_air_passenger_train.plot()
darts_air_passenger_test.plot()
pred_gbdt_bc_lt = air_passenger_boxcox.inverse_transform(
    gbdt_bc_lt.predict(n=ap_horizon) + ap_trend_linear_test
)
pred_gbdt_bc_lt.plot(label="Box-Cox + Linear Detrend Predictions", linestyle="--")

# + [markdown] id="76W-UUf4wYGt"
# ### Metrics

# + colab={"base_uri": "https://localhost:8080/", "height": 473} id="zUUbjb9NwYGt" outputId="9365d640-c66c-45de-ce81-f3e1f4132fea"
darts_air_passenger_test.plot(label="Test")
pred_gbdt_ap.plot(label="Simple GBDT", linestyle="--")
pred_gbdt_ap_seasonal.plot(
    label="GBDT on Global Detrended Data (Cheating)", linestyle="--"
)
pred_gbdt_bc_lt.plot(label="GBDT on Box-Cox + Linear Detrend Data", linestyle="--")

# + id="cjgcUttEwYGt"
benchmark_metrics = [
    metrics.mae,
    metrics.mape,
    metrics.mse,
    metrics.rmse,
    metrics.smape,
]


# + id="DMOnvM8gwYGx"
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


# + colab={"base_uri": "https://localhost:8080/", "height": 520} id="_Y6jm6Y9wYGy" outputId="f684e709-7e48-423b-d8de-013e272bdb1e"
benchmark_results = []

for i, pred in zip(
    ["simple_gbdt", "detrended_cheating", "boxcox_linear_trend"],
    [pred_gbdt_ap, pred_gbdt_ap_seasonal, pred_gbdt_bc_lt],
):
    benchmark_results += benchmark_predictions(
        series_true=darts_air_passenger_test,
        series_prediction=pred,
        metrics=benchmark_metrics,
        experiment_id=i,
    )

df_benchmark_metrics = pd.DataFrame(benchmark_results)
df_benchmark_metrics

# + colab={"base_uri": "https://localhost:8080/", "height": 1000} id="n4XHYPvRwYGy" outputId="cfc06499-8b89-4bf8-af9d-4fabb3ef3a74"
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

# + id="LFds5KtVwYGy"
