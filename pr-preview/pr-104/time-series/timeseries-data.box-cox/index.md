# Box-Cox Transformation

> [!NOTE]
>
> **In one minute**
>
> Real series are often non-stationary, with a variance that changes with the level. The Box-Cox transformation, a family of transforms indexed by λ (λ = 0 is the log), reduces this; the air passenger data shows it in action. It doesn't always produce perfectly stationary data.
>
> **Read first:** [Time Series Analysis](<https://dl.leima.is/time-series/timeseries-data.analysis/index.md>)

Many time series models require [stationary data](<https://dl.leima.is/time-series/timeseries-data.analysis/index.md>). However, real-world time series data may be non-stationary and [heteroscedastic](<https://en.wikipedia.org/wiki/Homoscedasticity_and_heteroscedasticity>)[^/time-series/timeseries-data.box-cox/#scedasticity]. Box-cox transformation is useful when reducing the non-stationarity and heteroscedasticity.

> [!NOTE]
>
> Rob J Hyndman and George Athanasopoulos's famous textbook FPP2 provides some [nice examples of box-cox transformations](<https://otexts.com/fpp2/transformations.html#mathematical-transformations>).

To see Box-Cox transformation in action, we show an example using the [air passenger dataset](<https://www.kaggle.com/datasets/rakannimer/air-passengers>).

The air passenger dataset is a monthly dataset. We can observe the trend and varying variance simply by eyes.

![Air Passenger Data](<https://dl.leima.is/time-series/assets/timeseries-transformations/timeseries_data.box-cox.airpassenger_original.png>)

Applying Box-Cox transformations with different lambdas leads to different results shown below. Drag the \\(\\lambda\\) slider to transform the series yourself: \\(\\lambda = 1\\) only shifts the data, \\(\\lambda = 0\\) is the log transform. The lower panel shows the rolling 12-month standard deviation of the transformed series divided by its average, so a flat line at 1 means the variance no longer grows with the level.

![Box-Cox transformed Air Passenger data](<https://dl.leima.is/time-series/assets/timeseries-transformations/timeseries_data.box-cox.airpassenger_transformed.png>)

To check the variance, we plot out the variance rolling on a 12-month window.

![Air Passenger Var](<https://dl.leima.is/time-series/assets/timeseries-transformations/timeseries_data.box-cox.airpassenger_original_var.png>)

Box-Cox transformation with \\(\\lambda =0.1\\) reduces the variability in variance.

![Box-Cox Air Passenger Var](<https://dl.leima.is/time-series/assets/timeseries-transformations/timeseries_data.box-cox.airpassenger_bc_lambda_0p1_var.png>)

> [!WARNING]
>
> **Box-Cox May not Always Reach Perfect Stationary Data**
>
> Box-Cox transformation is a simple transformation that helps us reduce the non-stationarity and heteroscedasticy. However, we may not always be able to convert the dataset to stationary and homoscedastic data. This can be observed by performing checks using tools such as [`stationarity_tests` in Darts](<https://unit8co.github.io/darts/generated_api/darts.utils.statistics.html?highlight=trend#darts.utils.statistics.remove_trend>).

[^/time-series/timeseries-data.box-cox/#scedasticity]: Homoscedasticity and heteroscedasticity. (2023, June 2). In Wikipedia. [https://en.wikipedia.org/wiki/Homoscedasticity\_and\_heteroscedasticity](<https://en.wikipedia.org/wiki/Homoscedasticity_and_heteroscedasticity >)
