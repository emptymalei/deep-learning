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
# -


# ## Load Data

# +
import pandas as pd

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

df_train = df[df.ds <= train_test_split_date]
df_test = df[df.ds > train_test_split_date]

horizon = 3
horizon
# -

# ## Baselines

from statsforecast import StatsForecast
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


def forecast_test(forecaster_pred, df_train, df_test):
    dfs_pred = []
    for i in range(len(df_test)):
        df_pred_input_i = pd.concat([df_train, df_test[:i]])
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
):
    _, ax = plt.subplots()

    sns.lineplot(df_train, x="ds", y="y", ax=ax)

    sns.lineplot(df_test, x="ds", y="y", color="k", ax=ax)

    sns.lineplot(df_pred, x="ds", y=model_name, hue="step", ax=ax)

    ax.set_title("Sunspot Forecast", fontsize=22)
    ax.set_ylabel("Sunspot Number", fontsize=20)
    ax.set_xlabel("Year", fontsize=20)
    ax.legend(prop={"size": 15})
    ax.grid()


df_y_hat_arima = forecast_test(
    lambda x: sf.predict(h=horizon, level=[95]), df_train=df_train, df_test=df_test
)

visualize_predictions(
    df_train=df_train, df_test=df_test, df_pred=df_y_hat_arima, model_name="AutoARIMA"
)

# ## Transformers

from neuralforecast import NeuralForecast
from neuralforecast.models import NBEATS, NHITS, VanillaTransformer, iTransformer

models = [
    VanillaTransformer(
        **{
            "hidden_size": 128,
            "n_head": 4,
            "learning_rate": 0.00010614524276500768,
            "scaler_type": "robust",
            "max_steps": 500,
            "batch_size": 32,
            "windows_batch_size": 512,
            "random_seed": 16,
            "input_size": 30,
            "step_size": 3,
            "h": horizon,
        }
    ),
    # iTransformer(
    #     input_size=history_length, h=horizon,
    #     # n_head=1,
    #     # conv_hidden_size=1,
    #     # encoder_layers=6,
    #     max_steps=50,
    #     n_series=1,
    #     #early_stop_patience_steps=5,
    # ),
    # NBEATS(input_size=history_length, h=horizon, max_steps=100, #early_stop_patience_steps=5
    #        ),
    # NHITS(input_size=history_length, h=horizon, max_steps=100, #early_stop_patience_steps=5
    #       )
]
nf = NeuralForecast(models=models, freq="d")

nf.fit(df=df_train)

train_test_split_date

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

# ## Evaluations

from datasetsforecast.evaluation import accuracy
from datasetsforecast.losses import mae, mape, mse, rmse, smape

df_y_hat_transformer.loc[df_y_hat_transformer.step == 0]

df_test

df_y_hat_transformer.loc[df_y_hat_transformer.step == 0]

# +
evaluation_df = accuracy(
    Y_hat_df=df_y_hat_transformer.loc[df_y_hat_transformer.step == 0],
    Y_test_df=df_test,
    Y_df=df_train,
    metrics=[mse, mae, rmse, mape, smape],
    agg_by=["unique_id"],
)

evaluation_df.head()

# -
accuracy(
    Y_hat_df=df_y_hat_arima.loc[df_y_hat_arima.step == 0],
    Y_test_df=df_test,
    Y_df=df_train,
    metrics=[mse, mae, rmse, mape, smape],
    agg_by=["unique_id"],
)
