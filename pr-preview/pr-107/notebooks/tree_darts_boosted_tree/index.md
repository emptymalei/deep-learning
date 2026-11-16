# Forecasting with Boosted Trees Using Darts

In this notebook, we explore some basic ideas of how to forecast using trees with the help of the package called [Darts](<https://github.com/unit8co/darts>).

On macOS, LightGBM crashes when it starts OpenMP after PyTorch has (darts imports PyTorch), so import it first. import lightgbm \# noqa: F401 \# isort: skip

```
from typing import Callable, Dict, List

import darts.utils as du
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from darts import TimeSeries, metrics
from darts.dataprocessing.transformers import BoxCox
from darts.datasets import AirPassengersDataset
from darts.models import LightGBMModel, NaiveDrift
from sklearn.linear_model import LinearRegression
```

## Following the Darts Official Tutorial

Darts provides a [tutorial here](<https://unit8co.github.io/darts/quickstart/00-quickstart.html>) to help the users get started. Here we replicate some of them to provide a minimal working example for tree-based models.

```
darts_air_passenger_series = AirPassengersDataset().load()
darts_air_passenger_series.plot()
```

```
darts_air_passenger_series
```

From the outputs, we see that the time series dataset contains montly data for 144 months.

```
train_series_length = 120
test_series_length = len(darts_air_passenger_series) - train_series_length
train_series_length, test_series_length
```

```
(
    darts_air_passenger_train,
    darts_air_passenger_test,
) = darts_air_passenger_series.split_before(train_series_length)

darts_air_passenger_train.plot(label="Training Data")
darts_air_passenger_test.plot(label="Test Data")
```

### First Random Forest Model

```
ap_horizon = len(darts_air_passenger_test)
ap_gbdt_params = dict(lags=52, output_chunk_length=ap_horizon)
```

```
gbdt_ap = LightGBMModel(**ap_gbdt_params)
```

```
gbdt_ap.fit(darts_air_passenger_train)
```

Insample predictions: We plot out the predictions for the last 24 days in the training data.

```
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
```

To observe the actual performance, we plot out the predictions of the test dates.

```
darts_air_passenger_train.plot(label="Train")
darts_air_passenger_test.plot(label="Test")
pred_gbdt_ap = gbdt_ap.predict(n=ap_horizon)
pred_gbdt_ap.plot(label="Prediction", linestyle="--")
```

### Detrending Helps

We train the same model but with the detrended dataset, and reconstruct the predictions using the trend.

```
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
```

```
(
    darts_air_passenger_seasonal_train,
    darts_air_passenger_seasonal_test,
) = darts_air_passenger_seasonal.split_before(120)
darts_air_passenger_seasonal_train.plot(label="Seasonal Component Train")
darts_air_passenger_seasonal_test.plot(label="Seasonal Component Test")
```

```
fig, ax = plt.subplots(figsize=(10, 6.18))

sns.histplot(
    darts_air_passenger_seasonal_train.to_dataframe(),
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
    darts_air_passenger_seasonal_test.to_dataframe(),
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
```

```
gbdt_ap_seasonal = LightGBMModel(**ap_gbdt_params)
```

```
gbdt_ap_seasonal.fit(darts_air_passenger_seasonal_train)
```

```
darts_air_passenger_train.plot(label="Train")
darts_air_passenger_test.plot(label="Test")
pred_rf_ap_seasonal = gbdt_ap_seasonal.predict(
    n=ap_horizon
) * darts_air_passenger_trend.drop_before(119)
pred_rf_ap_seasonal.plot(label="Trend * Predicted Seasonal Component", linestyle="--")
```

This indiates that the performance of trees on out of sample predictions if we only predict on the cycle part of the series. In a real world case, however, we have to predict the trend accurately for this to work. To better reconstruct the trend, there are also tricks like [Box-Cox transformations](<https://dl.leima.is/time-series/timeseries-data.box-cox/index.md>).

## Train, Test, and Metrics

It is not easy to determine a best model simply looking at the charts. We need some metrics.

```
air_passenger_boxcox = BoxCox()
darts_air_passenger_train_boxcox = air_passenger_boxcox.fit_transform(
    darts_air_passenger_train
)
darts_air_passenger_test_boxcox = air_passenger_boxcox.transform(
    darts_air_passenger_test
)
darts_air_passenger_train_boxcox.plot(label="Train (Box-Cox Transformed)")
darts_air_passenger_test_boxcox.plot(label="Test (Box-Cox Transformed)")
```

```
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
```

```
ap_trend_lm = linear_trend_model(darts_air_passenger_train_boxcox)
ap_trend_lm
```

```
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
```

```
darts_air_passenger_train_transformed = (
    darts_air_passenger_train_boxcox - ap_trend_linear_train
)
darts_air_passenger_train_transformed.plot()
```

```
gbdt_bc_lt = LightGBMModel(**ap_gbdt_params)
gbdt_bc_lt.fit(darts_air_passenger_train_transformed)
```

```
darts_air_passenger_train.plot()
darts_air_passenger_test.plot()
pred_gbdt_bc_lt = air_passenger_boxcox.inverse_transform(
    gbdt_bc_lt.predict(n=ap_horizon) + ap_trend_linear_test
)
pred_gbdt_bc_lt.plot(label="Box-Cox + Linear Detrend Predictions", linestyle="--")
```

## Linear Tree Horizon

Detrending is not the only possibility. LightGBM implements a linear tree version of the base learners.

```
ap_gbdt_linear_tree_params = dict(
    lags=52, output_chunk_length=ap_horizon, linear_tree=True
)
```

```
gbdt_linear_tree_ap = LightGBMModel(**ap_gbdt_linear_tree_params)
```

```
gbdt_linear_tree_ap.fit(darts_air_passenger_train)
```

```
darts_air_passenger_train.plot(label="Train")
darts_air_passenger_test.plot(label="Test")
pred_gbdt_linear_tree_ap = gbdt_linear_tree_ap.predict(n=ap_horizon)
pred_gbdt_linear_tree_ap.plot(label="Linear Tree Prediction", linestyle="--")
```

### Metrics

```
darts_air_passenger_test.plot(label="Test")
pred_gbdt_ap.plot(label="Simple GBDT", linestyle="--")
pred_rf_ap_seasonal.plot(
    label="GBDT on Global Detrended Data (Cheating)", linestyle="--"
)
pred_gbdt_bc_lt.plot(label="GBDT on Box-Cox + Linear Detrend Data", linestyle="--")
pred_gbdt_linear_tree_ap.plot(label="Linear Tree|", linestyle="--", color="r")
```

```
benchmark_metrics = [
    metrics.mae,
    metrics.mape,
    metrics.mse,
    metrics.rmse,
    metrics.smape,
    metrics.r2_score,
]
```

```
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
```

```
benchmark_results = []

for i, pred in zip(
    ["simple_gbdt", "detrended_cheating", "boxcox_linear_trend", "linear_tree"],
    [pred_gbdt_ap, pred_rf_ap_seasonal, pred_gbdt_bc_lt, pred_gbdt_linear_tree_ap],
):
    benchmark_results += benchmark_predictions(
        series_true=darts_air_passenger_test,
        series_prediction=pred,
        metrics=benchmark_metrics,
        experiment_id=i,
    )

df_benchmark_metrics = pd.DataFrame(benchmark_results)
df_benchmark_metrics
```

```
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
```
