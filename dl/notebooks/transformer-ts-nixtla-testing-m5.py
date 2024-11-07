# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: light
#       format_version: '1.5'
#       jupytext_version: 1.15.2
#   kernelspec:
#     display_name: .venv
#     language: python
#     name: python3
# ---

# # Transformer Forecaster with neuralforecast

# +
import os

os.environ["NIXTLA_ID_AS_COL"] = "1"

import datetime
from typing import Optional

# +
import pandas as pd

# +
from loguru import logger

# -

# ## Load Data


data_source = "https://raw.githubusercontent.com/datumorphism/dataset-m5-simplified/b486cd6a3e183b80016a91ba0fd9b19493cdc587/dataset/m5_store_sales.csv"

df = (
    pd.read_csv(data_source, parse_dates=["date"])
    .rename(columns={"date": "ds", "CA": "y"})[["ds", "y"]]
    .assign(unique_id=1)
)
# -

df.describe()

# +
import matplotlib.pyplot as plt
import seaborn as sns
from statsmodels.graphics.tsaplots import plot_acf

sns.set_theme(context="paper", style="ticks", palette="colorblind")


_, ax = plt.subplots(figsize=(10, 13), nrows=3)

sns.lineplot(df, x="ds", y="y", color="k", ax=ax[0])

ax[0].set_title("Walmart Sales in CA")
ax[0].set_ylabel("Sales")

sns.lineplot(
    (df.loc[(df.ds >= "2015-01-01") & (df.ds < "2016-01-01")]),
    x="ds",
    y="y",
    color="k",
    ax=ax[1],
)

ax[1].set_title("Walmart Sales in CA in 2015")
ax[1].set_ylabel("Sales")

plot_acf(df.y, ax=ax[2], color="k")
# -

# ## Prepare Data

df.ds.max(), df.ds.min()

# +
train_test_split_date = "2016-01-01"

df_train = df[df.ds < train_test_split_date]
df_test = df[(df.ds >= train_test_split_date)]

horizon = 3
horizon
# -

# ## Baselines

from statsforecast import StatsForecast
from statsforecast.arima import arima_string
from statsforecast.models import AutoARIMA

sf = StatsForecast(
    models=[
        AutoARIMA(
            # season_length = 365.25
            seasonal=False
        )
    ],
    freq="d",
)

sf.fit(df_train)


def forecast_test(
    forecaster_pred: callable,
    df_train: pd.DataFrame,
    df_test: pd.DataFrame,
) -> pd.DataFrame:
    """Generate forecasts for tests

    :param forecaster_pred: forecasting function/method
    :param df_train: train dataframe
    :param df_test: test dataframe
    """
    dfs_pred = []
    for i in range(len(df_test)):
        logger.debug(f"Prediction Step: {i}")
        df_pred_input_i = pd.concat([df_train, df_test[:i]]).reset_index(drop=True)
        df_pred_output_i = forecaster_pred(df_pred_input_i)
        df_pred_output_i["step"] = i
        dfs_pred.append(df_pred_output_i)
    df_y_hat = pd.concat(dfs_pred)

    return df_y_hat


def visualize_predictions(
    df_train: pd.DataFrame,
    df_test: pd.DataFrame,
    df_pred: pd.DataFrame,
    model_name: str,
    n_ahead: Optional[int] = None,
    title: Optional[str] = None,
    ax: Optional[plt.axes] = None,
) -> None:
    """
    Visualizes the forecasts

    :param df_train: train dataframe
    :param df_test: test dataframe
    :param df_pred: prediction dataframe
    :param model_name: which model to visualize
    :param n_ahead: which future step to visualze
    :param title: title of the chart
    :param ax: matplotlib axes
    """
    if ax is None:
        _, ax = plt.subplots(figsize=(10, 6.18))

    sns.lineplot(df_train, x="ds", y="y", ax=ax)

    sns.lineplot(df_test, x="ds", y="y", color="k", ax=ax)

    if n_ahead is None:
        sns.lineplot(
            df_pred, x="ds", y=model_name, hue="step", linestyle="dashed", ax=ax
        )
    else:
        dfs_pred_n_ahead = []
        for i in df_pred.step.unique():
            dfs_pred_n_ahead.append(
                df_pred.loc[df_pred.step == i].iloc[n_ahead - 1 : n_ahead]
            )
        df_pred_n_ahead = pd.concat(dfs_pred_n_ahead)
        sns.lineplot(df_pred_n_ahead, x="ds", y=model_name, linestyle="dashed", ax=ax)

    ax.set_title("Sales Forecast" if title is None else title)
    ax.set_ylabel("Sales")
    ax.set_xlabel("Date")
    # ax.legend(prop={'size': 15})
    # ax.grid()


# +
df_y_hat_arima = forecast_test(
    lambda x: sf.forecast(df=x, h=horizon), df_train=df_train, df_test=df_test
)

df_y_hat_arima
# -

visualize_predictions(
    df_train=df_train, df_test=df_test, df_pred=df_y_hat_arima, model_name="AutoARIMA"
)

visualize_predictions(
    df_train=df_train.loc[df_train.ds > "2015-01-01"],
    df_test=df_test,
    df_pred=df_y_hat_arima,
    model_name="AutoARIMA",
)

visualize_predictions(
    df_train=df_train.loc[df_train.ds > "2015-01-01"],
    df_test=df_test,
    df_pred=df_y_hat_arima,
    model_name="AutoARIMA",
    n_ahead=1,
)

visualize_predictions(
    df_train=df_train.loc[df_train.ds > "2015-01-01"],
    df_test=df_test,
    df_pred=df_y_hat_arima,
    model_name="AutoARIMA",
    n_ahead=2,
)

visualize_predictions(
    df_train=df_train.loc[df_train.ds > "2015-01-01"],
    df_test=df_test,
    df_pred=df_y_hat_arima,
    model_name="AutoARIMA",
    n_ahead=3,
    title="Forecasting 3 Steps Ahead (ARIMA)",
)

arima_string(sf.fitted_[0, 0].model_)

# ## Transformers

from neuralforecast import NeuralForecast
from neuralforecast.models import VanillaTransformer, iTransformer

models = [
    VanillaTransformer(
        hidden_size=128,
        n_head=4,
        learning_rate=0.0001,
        scaler_type="robust",
        max_steps=500,
        batch_size=32,
        windows_batch_size=512,
        random_seed=16,
        input_size=30,
        step_size=3,
        h=horizon,
        # **{'hidden_size': 128,
        #     'n_head': 4,
        #     'learning_rate': 0.00010614524276500768,
        #     'scaler_type': 'robust',
        #     'max_steps': 500,
        #     'batch_size': 32,
        #     'windows_batch_size': 512,
        #     'random_seed': 16,
        #     'input_size': 30,
        #     'step_size': 3,
        #     "h": horizon,
        # }
    ),
    iTransformer(
        input_size=30,
        h=horizon,
        max_steps=50,
        n_series=1,
    ),
]
nf = NeuralForecast(models=models, freq="d")

nf.fit(df=df_train)

df_y_hat_transformer = forecast_test(nf.predict, df_train=df_train, df_test=df_test)

df_y_hat_transformer

df_test

df_y_hat_transformer.ds.dt.freq

visualize_predictions(
    df_train=df_train,
    df_test=df_test,
    df_pred=df_y_hat_transformer,
    model_name="VanillaTransformer",
)

visualize_predictions(
    df_train=df_train.loc[df_train.ds >= "2015-01-01"],
    df_test=df_test,
    df_pred=df_y_hat_transformer,
    model_name="VanillaTransformer",
    n_ahead=1,
)

visualize_predictions(
    df_train=df_train.loc[df_train.ds >= "2015-01-01"],
    df_test=df_test,
    df_pred=df_y_hat_transformer,
    model_name="VanillaTransformer",
    n_ahead=2,
)

visualize_predictions(
    df_train=df_train.loc[df_train.ds >= "2015-01-01"],
    df_test=df_test,
    df_pred=df_y_hat_transformer,
    model_name="VanillaTransformer",
    n_ahead=3,
    title="Forecasting 3 Steps Ahead (VanillaTransformer)",
)

visualize_predictions(
    df_train=df_train.loc[df_train.ds >= "2015-01-01"],
    df_test=df_test,
    df_pred=df_y_hat_transformer,
    model_name="iTransformer",
    n_ahead=3,
    title="Forecasting 3 Steps Ahead (iTransformer)",
)

# ## Evaluations

from datasetsforecast.evaluation import accuracy
from datasetsforecast.losses import mae, mape, mse, rmse, smape

df_y_hat_transformer.loc[df_y_hat_transformer.step == 0]

df_test

df_y_hat_transformer.loc[df_y_hat_transformer.step == 0]

# +
transformer_evals = []

for s in df_y_hat_transformer.step.unique():
    df_transformer_eval_s = accuracy(
        Y_hat_df=df_y_hat_transformer.loc[df_y_hat_transformer.step == s],
        Y_test_df=df_test,
        Y_df=df_train,
        metrics=[mse, mae, rmse],
        # agg_by=['unique_id']
    )
    df_transformer_eval_s["step"] = s
    transformer_evals.append(df_transformer_eval_s)

df_transformer_eval = pd.concat(transformer_evals)

df_transformer_eval.head()

# +
baseline_evals = []

for s in df_y_hat_arima.step.unique():
    df_baseline_eval_s = accuracy(
        Y_hat_df=df_y_hat_arima.loc[df_y_hat_arima.step == s],
        Y_test_df=df_test,
        Y_df=df_train,
        metrics=[mse, mae, rmse],
        # agg_by=['unique_id']
    )
    df_baseline_eval_s["step"] = s
    baseline_evals.append(df_baseline_eval_s)


df_baseline_eval = pd.concat(baseline_evals)

df_baseline_eval.head()


# +
df_eval_metrics = pd.merge(
    df_transformer_eval, df_baseline_eval, how="left", on=["metric", "step"]
).melt(
    id_vars=["metric", "step"],
    value_vars=[
        "VanillaTransformer",
        "iTransformer",
        "AutoARIMA",
    ],
    var_name="model",
)

df_eval_metrics

# +
_, ax = plt.subplots(figsize=(13, 5), ncols=3)

for i, m in enumerate(df_eval_metrics.metric.unique()):
    sns.violinplot(
        df_eval_metrics.loc[df_eval_metrics.metric == m],
        x="model",
        y="value",
        hue="model",
        fill=False,
        ax=ax[i],
        label=m,
    )
    ax[i].set_title(f"Metric: {m}")

# +
_, ax = plt.subplots(figsize=(10, 13), nrows=3)

visualize_predictions(
    df_train=df_train.loc[df_train.ds > "2015-01-01"],
    df_test=df_test,
    df_pred=df_y_hat_arima,
    model_name="AutoARIMA",
    n_ahead=3,
    title="Forecasting 3 Steps Ahead (ARIMA)",
    ax=ax[0],
)

visualize_predictions(
    df_train=df_train.loc[df_train.ds >= "2015-01-01"],
    df_test=df_test,
    df_pred=df_y_hat_transformer,
    model_name="VanillaTransformer",
    n_ahead=3,
    title="Forecasting 3 Steps Ahead (VanillaTransformer)",
    ax=ax[1],
)

visualize_predictions(
    df_train=df_train.loc[df_train.ds >= "2015-01-01"],
    df_test=df_test,
    df_pred=df_y_hat_transformer,
    model_name="iTransformer",
    n_ahead=3,
    title="Forecasting 3 Steps Ahead (iTransformer)",
    ax=ax[2],
)
# -
