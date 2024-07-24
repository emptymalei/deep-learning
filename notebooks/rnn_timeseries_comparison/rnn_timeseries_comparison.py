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
#     display_name: .venv
#     language: python
#     name: python3
# ---

# # Comparing RNN Models for Time Series Forecasting
#
# In this notebook, we compare different RNN models for time series forecasting.

import dataclasses

import lightning as L
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from lightning.pytorch.callbacks.early_stopping import EarlyStopping
from torch import nn
from ts_bolt.datamodules.pandas import DataFrameDataModule
from ts_bolt.evaluation.evaluator import Evaluator
from ts_bolt.naive_forecasters.last_observation import LastObservationForecaster

# ## Data
#
# We prepare a toy dataset using `sin`


df = pd.DataFrame(
    {"t": np.linspace(0, 100, 501), "y": np.sin(np.linspace(0, 100, 501))}
)

df.head()

# +
_, ax = plt.subplots(figsize=(10, 6.18))

df.plot(x="t", y="y", ax=ax)


# -

# ## Model
#
# In this section, we create the RNN model.


# +
@dataclasses.dataclass
class TSRNNParams:
    """A dataclass to be served as our parameters
    for the model.

    :param hidden_size: number of dimensions in the hidden state
    :param input_size: input dim
    :param num_layers: number of units stacked
    """

    input_size: int
    hidden_size: int
    num_layers: int = 1


class TSRNN(nn.Module):
    """RNN for univaraite time series modeling.

    :param history_length: the length of the input history.
    :param horizon: the number of steps to be forecasted.
    :param rnn_params: the parameters for the RNN network.
    """

    def __init__(self, history_length: int, horizon: int, rnn_params: TSRNNParams):
        super().__init__()
        self.rnn_params = rnn_params
        self.history_length = history_length
        self.horizon = horizon

        self.regulate_input = nn.Linear(self.history_length, self.rnn_params.input_size)

        self.rnn = nn.RNN(
            input_size=self.rnn_params.input_size,
            hidden_size=self.rnn_params.hidden_size,
            num_layers=self.rnn_params.num_layers,
            batch_first=True,
        )

        self.regulate_output = nn.Linear(self.rnn_params.hidden_size, self.horizon)

    @property
    def rnn_config(self):
        return dataclasses.asdict(self.rnn_params)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.regulate_input(x)
        x, _ = self.rnn(x)

        return self.regulate_output(x)


# -

# ## Training

# We use [lightning](https://lightning.ai/docs/pytorch/stable/) to train our model.

# ### Training Utilities

history_length_1_step = 100
horizon_1_step = 1


# We will build a few utilities
#
# 1. To be able to feed the data into our model, we build a class (`DataFrameDataset`) that converts the pandas dataframe into a Dataset for pytorch.
# 2. To make the lightning training code simpler, we will build a [LightningDataModule](https://lightning.ai/docs/pytorch/stable/data/datamodule.html) (`DataFrameDataModule`) and a [LightningModule](https://lightning.ai/docs/pytorch/stable/common/lightning_module.html) (`RNNForecaster`).


class RNNForecaster(L.LightningModule):
    def __init__(self, rnn: nn.Module):
        super().__init__()
        self.rnn = rnn

    def configure_optimizers(self):
        optimizer = torch.optim.SGD(self.parameters(), lr=1e-3)
        return optimizer

    def training_step(self, batch, batch_idx):
        x, y = batch
        x = x.squeeze().type(self.dtype)
        y = y.squeeze(-1).type(self.dtype)

        y_hat = self.rnn(x)

        loss = nn.functional.l1_loss(y_hat, y)
        self.log_dict({"train_loss": loss}, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        x = x.squeeze().type(self.dtype)
        y = y.squeeze(-1).type(self.dtype)

        y_hat = self.rnn(x)

        loss = nn.functional.l1_loss(y_hat, y)
        self.log_dict({"val_loss": loss}, prog_bar=True)
        return loss

    def predict_step(self, batch, batch_idx):
        x, y = batch
        x = x.squeeze().type(self.dtype)
        y = y.squeeze(-1).type(self.dtype)

        y_hat = self.rnn(x)
        return x, y_hat

    def forward(self, x):
        x = x.squeeze().type(self.dtype)
        return x, self.rnn(x)


# ### Data, Model and Training

# #### DataModule

pdm_1_step = DataFrameDataModule(
    history_length=history_length_1_step,
    horizon=horizon_1_step,
    dataframe=df[["y"]],
)

# #### LightningModule

# +
ts_rnn_params_1_step = TSRNNParams(input_size=96, hidden_size=64, num_layers=1)

ts_rnn_1_step = TSRNN(
    history_length=history_length_1_step,
    horizon=horizon_1_step,
    rnn_params=ts_rnn_params_1_step,
)

ts_rnn_1_step
# -

rnn_forecaster_1_step = RNNForecaster(rnn=ts_rnn_1_step)

# #### Trainer

# +
logger_1_step = L.pytorch.loggers.TensorBoardLogger(
    save_dir="lightning_logs", name="rnn_ts_1_step"
)

trainer_1_step = L.Trainer(
    precision="64",
    max_epochs=100,
    min_epochs=5,
    callbacks=[
        EarlyStopping(monitor="val_loss", mode="min", min_delta=1e-7, patience=3)
    ],
    logger=logger_1_step,
)
# -

# #### Fitting

trainer_1_step.fit(model=rnn_forecaster_1_step, datamodule=pdm_1_step)

# #### Retrieving Predictions

predictions_1_step = trainer_1_step.predict(
    model=rnn_forecaster_1_step, datamodule=pdm_1_step
)

# ### Naive Forecaster

# +
trainer_naive_1_step = L.Trainer(precision="64")

lobs_forecaster_1_step = LastObservationForecaster(horizon=horizon_1_step)
lobs_1_step_predictions = trainer_naive_1_step.predict(
    model=lobs_forecaster_1_step, datamodule=pdm_1_step
)
# -

# ### Evaluations

evaluator_1_step = Evaluator(step=0)

# +
fig, ax = plt.subplots(figsize=(10, 6.18))

ax.plot(
    evaluator_1_step.y_true(dataloader=pdm_1_step.predict_dataloader()),
    "g-",
    label="truth",
)

ax.plot(evaluator_1_step.y(predictions_1_step), "r--", label="predictions")

ax.plot(evaluator_1_step.y(lobs_1_step_predictions), "b-.", label="naive predictions")

plt.legend()
# -

evaluator_1_step.metrics(predictions_1_step, pdm_1_step.predict_dataloader())

evaluator_1_step.metrics(lobs_1_step_predictions, pdm_1_step.predict_dataloader())

pd.merge(
    evaluator_1_step.metrics(predictions_1_step, pdm_1_step.predict_dataloader()),
    evaluator_1_step.metrics(lobs_1_step_predictions, pdm_1_step.predict_dataloader()),
    how="inner",
    left_index=True,
    right_index=True,
    suffixes=["_rnn", "_last_obs"],
)

# ## Real Life
#
# We take the [Airpassenger dataset](https://github.com/DataHerb/dataset-airpassenger).

df_ap = pd.read_csv(
    "https://raw.githubusercontent.com/DataHerb/dataset-airpassenger/main/dataset/AirPassengers.csv"
)
