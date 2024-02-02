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

# # RNN for Univariate Time Series Forecasting
#
# In this notebook, we build a RNN using pytorch to forecast $\sin$ function as a time series.

import dataclasses

# +
import math
from functools import cached_property
from typing import Dict, List, Tuple

import lightning as L
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from lightning.pytorch.callbacks.early_stopping import EarlyStopping
from loguru import logger
from torch import nn
from torch.utils.data import DataLoader, Dataset
from ts_dl_utils.datasets.pendulum import Pendulum, PendulumDataModule
from ts_dl_utils.evaluation.evaluator import Evaluator
from ts_dl_utils.naive_forecasters.last_observation import LastObservationForecaster

# -

# ## Data
#
# We create a dataset that models a damped pendulum. The pendulum is modelled as a damped harmonic oscillator, i.e.,
#
# $$
# \theta(t) = \theta(0) \cos(2 \pi t / p)\exp(-\beta t),
# $$
#
# where $\theta(t)$ is the angle of the pendulum at time $t$.
# The period $p$ is calculated using
#
# $$
# p = 2 \pi \sqrt(L / g),
# $$
#
# with $L$ being the length of the pendulum
# and $g$ being the surface gravity.


pen = Pendulum(length=100)

df = pd.DataFrame(pen(10, 400, initial_angle=1, beta=0.001))

# Since the damping constant is very small, the data generated is mostly a sin wave.

# +
_, ax = plt.subplots(figsize=(10, 6.18))

df.plot(x="t", y="theta", ax=ax)


# -

# ## Model
#
# In this section, we create the RNN model.


# +
@dataclasses.dataclass
class TSRNNParams:
    """A dataclass to be served as our parameters for the model.

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

# +
history_length_1_step = 100
horizon_1_step = 1

gap = 10
# -


# We will build a few utilities
#
# 1. To be able to feed the data into our model, we build a class (`DataFrameDataset`) that converts the pandas dataframe into a Dataset for pytorch.
# 2. To make the lightning training code simpler, we will build a [LightningDataModule](https://lightning.ai/docs/pytorch/stable/data/datamodule.html) (`PendulumDataModule`) and a [LightningModule](https://lightning.ai/docs/pytorch/stable/common/lightning_module.html) (`RNNForecaster`).


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

pdm_1_step = PendulumDataModule(
    history_length=history_length_1_step,
    horizon=horizon_1_step,
    gap=gap,
    dataframe=df[["theta"]],
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
        EarlyStopping(monitor="val_loss", mode="min", min_delta=1e-5, patience=2)
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

# ## Multi-horizon Forecast (h=3)

# ### Train a Model

history_length_m_step = 100
horizon_m_step = 3

pdm_m_step = PendulumDataModule(
    history_length=history_length_m_step,
    horizon=horizon_m_step,
    dataframe=df[["theta"]],
    gap=gap,
)

# +
ts_rnn_params_m_step = TSRNNParams(input_size=96, hidden_size=64, num_layers=1)

ts_rnn_m_step = TSRNN(
    history_length=history_length_m_step,
    horizon=horizon_m_step,
    rnn_params=ts_rnn_params_m_step,
)

ts_rnn_m_step
# -

rnn_forecaster_m_step = RNNForecaster(rnn=ts_rnn_m_step)

# +
logger_m_step = L.pytorch.loggers.TensorBoardLogger(
    save_dir="lightning_logs", name="rnn_ts_m_step"
)

trainer_m_step = L.Trainer(
    precision="64",
    max_epochs=100,
    min_epochs=5,
    callbacks=[
        EarlyStopping(monitor="val_loss", mode="min", min_delta=1e-5, patience=2)
    ],
    logger=logger_m_step,
)
# -

trainer_m_step.fit(model=rnn_forecaster_m_step, datamodule=pdm_m_step)

predictions_m_step = trainer_m_step.predict(
    model=rnn_forecaster_m_step, datamodule=pdm_m_step
)

# ### Naive Forecaster

# +
trainer_naive_m_step = L.Trainer(precision="64")

lobs_forecaster_m_step = LastObservationForecaster(horizon=horizon_m_step)
lobs_m_step_predictions = trainer_naive_m_step.predict(
    model=lobs_forecaster_m_step, datamodule=pdm_m_step
)
# -

# ### Evaluations

evaluator_m_step = Evaluator(step=2, gap=gap)

# +
fig, ax = plt.subplots(figsize=(10, 6.18))

ax.plot(
    evaluator_m_step.y_true(dataloader=pdm_m_step.predict_dataloader()),
    "g-",
    label="truth",
)

ax.plot(evaluator_m_step.y(predictions_m_step), "r--", label="predictions")

ax.plot(evaluator_m_step.y(lobs_m_step_predictions), "b-.", label="naive predictions")

plt.legend()

# +
fig, ax = plt.subplots(figsize=(10, 6.18))


for i in np.arange(0, 1000, 120):
    evaluator_m_step.plot_one_sample(ax=ax, predictions=predictions_m_step, idx=i)
# -

evaluator_m_step.metrics(predictions_m_step, pdm_m_step.predict_dataloader())

evaluator_m_step.metrics(lobs_m_step_predictions, pdm_m_step.predict_dataloader())
