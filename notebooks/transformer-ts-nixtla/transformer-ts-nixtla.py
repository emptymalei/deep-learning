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

import pandas as pd
import seaborn as sns

# ## Load Data

# +
data_source = "https://gist.githubusercontent.com/emptymalei/921a624ce44e6a60bb6ec637b195ceaf/raw/4cb52fa20dcc598e16891eae2749fd96a97750e9/sunspot.csv"

df = (
    pd.read_csv(data_source, parse_dates=["date"])
    .rename(columns={"date": "ds", "avg_sunspot_area": "y"})
    .assign(unique_id=1)
)

# -

df.describe()

df.plot(x="ds", y="y")

# ## Prepare Data

# +
train_test_split_date = "2000-01-01"

df_train = df[df.ds <= train_test_split_date]
df_test = df[df.ds > train_test_split_date]

horizon = 3
horizon
# -

# ## Baselines

from statsforecast import StatsForecast
from statsforecast.models import AutoARIMA

sf = StatsForecast(models=[AutoARIMA(season_length=12)], freq="YS")

sf.fit(df_train)

df_y_hat_arima = sf.predict(h=horizon, level=[95])
df_y_hat_arima

# +
import matplotlib.pyplot as plt

fig, ax = plt.subplots(1, 1, figsize=(20, 7))
df_chart = df_test.merge(df_y_hat_arima, how="left", on=["unique_id", "ds"])
df_chart = pd.concat([df_train, df_chart]).set_index("ds")

df_chart[
    [
        "y",
        "AutoARIMA",
        # 'NHITS'
    ]
].plot(ax=ax, linewidth=2)

ax.set_title("AirPassengers Forecast", fontsize=22)
ax.set_ylabel("Monthly Passengers", fontsize=20)
ax.set_xlabel("Timestamp [t]", fontsize=20)
ax.legend(prop={"size": 15})
ax.grid()
# -

# ## Transformers

from neuralforecast import NeuralForecast
from neuralforecast.auto import AutoVanillaTransformer
from neuralforecast.models import NBEATS, NHITS, VanillaTransformer, iTransformer

# +
# vt_config = dict(
#     max_steps=1, val_check_steps=1, input_size=12, hidden_size=8
# )
auto_vt_model = AutoVanillaTransformer(h=horizon, backend="optuna")

nf_auto = NeuralForecast(models=[auto_vt_model], freq="YS")
# -

nf_auto.fit(df=df_train, val_size=3)

results = nf_auto.models[0].results.trials_dataframe()
results.drop(columns="user_attrs_ALL_PARAMS")


df_y_hat_optuna = nf_auto.predict().reset_index()
df_y_hat_optuna.head()

# +
import matplotlib.pyplot as plt

fig, ax = plt.subplots(1, 1, figsize=(20, 7))
df_chart = df_test.merge(df_y_hat_optuna, how="left", on=["unique_id", "ds"])
df_chart = pd.concat([df_train, df_chart]).set_index("ds")

df_chart[
    [
        "y",
        "AutoVanillaTransformer",
    ]
].plot(ax=ax, linewidth=2)

ax.set_title("Forecast", fontsize=22)
ax.set_ylabel("Sunspot Area", fontsize=20)
ax.set_xlabel("Timestamp [t]", fontsize=20)
ax.legend(prop={"size": 15})
ax.grid()
# -

nf_auto.models[0].results.best_trial.params

# +
# nf_auto.save("lightning_logs/nf_auto_vanilla_transformer_sunspot")
# -

# ## Transformers

models = [
    VanillaTransformer(
        # input_size=9, h=horizon,
        # n_head=4,
        # windows_batch_size=512,
        # learning_rate=0.00010614524276500768,
        # # conv_hidden_size=2,
        # # encoder_layers=6,
        # max_steps=500,
        # #early_stop_patience_steps=5,
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
nf = NeuralForecast(models=models, freq="YS")

nf.fit(df=df_train)

train_test_split_date


dfs_pred = []
for i in range(len(df_test)):
    df_pred_input_i = pd.concat([df_train, df_test[:i]])
    df_pred_output_i = nf.predict(df_pred_input_i)
    df_pred_output_i["step"] = i
    dfs_pred.append(df_pred_output_i)
df_y_hat = pd.concat(dfs_pred).reset_index(drop=False)

df_y_hat

df_test

df_y_hat.ds.dt.freq

# +
import matplotlib.pyplot as plt

fig, ax = plt.subplots(1, 1, figsize=(20, 7))

sns.lineplot(df_train, x="ds", y="y", ax=ax)

sns.lineplot(df_test, x="ds", y="y", color="k", ax=ax)

sns.lineplot(df_y_hat, x="ds", y="VanillaTransformer", hue="step", ax=ax)


ax.set_title("Sunspot Forecast", fontsize=22)
ax.set_ylabel("Avg Sunspot Area", fontsize=20)
ax.set_xlabel("Year", fontsize=20)
ax.legend(prop={"size": 15})
ax.grid()
# -

from datasetsforecast.evaluation import accuracy
from datasetsforecast.losses import mae, mse, rmse

df_y_hat.loc[df_y_hat.step == 0]

df_test

# +
evaluation_df = accuracy(
    Y_hat_df=df_y_hat.loc[df_y_hat.step == 0],
    Y_test_df=df_test,
    Y_df=df_train,
    metrics=[mse, mae, rmse],
    agg_by=["unique_id"],
)

evaluation_df.head()
