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
from ts_dl_utils.datasets.pendulum import Pendulum

# ## Load Data

# +
pen = Pendulum(length=100)
df_pen = (
    pd.DataFrame(pen(10, 400, initial_angle=1, beta=0.00001))
    .reset_index()
    .rename(columns={"index": "ds", "theta": "y"})[["ds", "y"]]
)
df_pen["unique_id"] = 1

df_pen.head()
# -

df_pen.plot(x="ds", y="y")

# +
df_pen_train = df_pen[:-3]
df_pen_test = df_pen[-3:]

horizon_pen = len(df_pen_test)
# -

# ## Prepare Data

# ## Baselines

from statsforecast import StatsForecast
from statsforecast.models import AutoARIMA

sf_pen = StatsForecast(models=[AutoARIMA(season_length=12)], freq=1)

sf_pen.fit(df_pen_train)

df_pen_y_hat_arima = sf_pen.predict(h=horizon_pen, level=[95])
df_pen_y_hat_arima

# +
_, ax = plt.subplots()

df_pen_test.plot(x="ds", y="y", ax=ax)
df_pen_y_hat_arima.plot(x="ds", y="AutoARIMA", ax=ax)
# -


# ## Transformers

# +
from neuralforecast import NeuralForecast
from neuralforecast.auto import AutoVanillaTransformer
from neuralforecast.models import NBEATS, NHITS, VanillaTransformer, iTransformer

history_length_pen = 10

# -

models_pen = [
    VanillaTransformer(
        input_size=history_length_pen,
        h=horizon_pen,
        n_head=1,
        conv_hidden_size=1,
        encoder_layers=6,
        max_steps=100,
        # early_stop_patience_steps=5,
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
nf_pen = NeuralForecast(models=models_pen, freq=1)

nf_pen.fit(df=df_pen_train)

df_pen_y_hat = nf_pen.predict().reset_index()

# +
import matplotlib.pyplot as plt

fig, ax = plt.subplots(1, 1, figsize=(20, 7))
df_chart = df_pen_test.merge(df_pen_y_hat, how="left", on=["unique_id", "ds"])
df_chart = pd.concat([df_pen_train, df_chart]).set_index("ds")

df_chart[
    [
        "y",
        # 'NBEATS',
        "VanillaTransformer",
        # "iTransformer"
        # 'NHITS'
    ]
].plot(ax=ax, linewidth=2)

ax.set_title("AirPassengers Forecast", fontsize=22)
ax.set_ylabel("Monthly Passengers", fontsize=20)
ax.set_xlabel("Timestamp [t]", fontsize=20)
ax.legend(prop={"size": 15})
ax.grid()
