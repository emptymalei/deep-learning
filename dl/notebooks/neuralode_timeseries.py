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

# # NeuralODE for Univariate Time Series Forecasting
#
# In this notebook, we build a NeuralODE using pytorch to forecast $\sin$ function as a time series.

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
from torchdyn.core import NeuralODE
from torchmetrics import MetricCollection
from torchmetrics.regression import (
    MeanAbsoluteError,
    MeanAbsolutePercentageError,
    MeanSquaredError,
    SymmetricMeanAbsolutePercentageError,
)

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


class Pendulum:
    """Class for generating time series data for a pendulum.

    The pendulum is modelled as a damped harmonic oscillator, i.e.,

    $$
    \theta(t) = \theta(0) \cos(2 \pi t / p)\exp(-\beta t),
    $$

    where $\theta(t)$ is the angle of the pendulum at time $t$.
    The period $p$ is calculated using

    $$
    p = 2 \pi \sqrt(L / g),
    $$

    with $L$ being the length of the pendulum
    and $g$ being the surface gravity.

    :param length: Length of the pendulum.
    :param gravity: Acceleration due to gravity.
    """

    def __init__(self, length: float, gravity: float = 9.81) -> None:
        self.length = length
        self.gravity = gravity

    @cached_property
    def period(self) -> float:
        """Calculate the period of the pendulum."""
        return 2 * math.pi * math.sqrt(self.length / self.gravity)

    def __call__(
        self,
        num_periods: int,
        num_samples_per_period: int,
        initial_angle: float = 0.1,
        beta: float = 0,
    ) -> Dict[str, List[float]]:
        """Generate time series data for the pendulum.

        Returns a list of floats representing the angle
        of the pendulum at each time step.

        :param num_periods: Number of periods to generate.
        :param num_samples_per_period: Number of samples per period.
        :param initial_angle: Initial angle of the pendulum.
        """
        time_step = self.period / num_samples_per_period
        steps = []
        time_series = []
        for i in range(num_periods * num_samples_per_period):
            t = i * time_step
            angle = (
                initial_angle
                * math.cos(2 * math.pi * t / self.period)
                * math.exp(-beta * t)
            )
            steps.append(t)
            time_series.append(angle)

        return {"t": steps, "theta": time_series}


pen = Pendulum(length=100)

df = pd.DataFrame(pen(10, 400, initial_angle=1, beta=0.001))

# Since the damping constant is very small, the data generated is mostly a sin wave.

# +
_, ax = plt.subplots(figsize=(10, 6.18))

df.plot(x="t", y="theta", ax=ax)


# -

# ## Model
#
# In this section, we create the NeuralODE model.


# +
@dataclasses.dataclass
class TSNODEParams:
    """A dataclass to be served as our parameters for the model.

    :param hidden_widths: list of dimensions for the hidden layers
    """

    hidden_widths: List[int]
    time_span: torch.Tensor


class TSNODE(nn.Module):
    """NeuralODE for univaraite time series modeling.

    :param history_length: the length of the input history.
    :param horizon: the number of steps to be forecasted.
    :param ffn_params: the parameters for the NODE network.
    """

    def __init__(self, history_length: int, horizon: int, model_params: TSNODEParams):
        super().__init__()
        self.model_params = model_params
        self.history_length = history_length
        self.horizon = horizon

        self.time_span = model_params.time_span

        self.regulate_input = nn.Linear(
            self.history_length, self.model_params.hidden_widths[0]
        )

        self.hidden_layers = nn.Sequential(
            *[
                self._linear_block(dim_in, dim_out)
                for dim_in, dim_out in zip(
                    self.model_params.hidden_widths[:-1],
                    self.model_params.hidden_widths[1:],
                )
            ]
        )

        self.regulate_output = nn.Linear(
            self.model_params.hidden_widths[-1], self.history_length
        )

        self.network = nn.Sequential(
            *[self.regulate_input, self.hidden_layers, self.regulate_output]
        )

    @property
    def node_config(self):
        return dataclasses.asdict(self.ffn_params)

    def _linear_block(self, dim_in, dim_out):
        return nn.Sequential(*[nn.Linear(dim_in, dim_out), nn.ReLU()])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.network(x)


# -

# ## Training

# We use [lightning](https://lightning.ai/docs/pytorch/stable/) to train our model.

# ### Training Utilities

history_length = 100
horizon = 1


# We will build a few utilities
#
# 1. To be able to feed the data into our model, we build a class (`DataFrameDataset`) that converts the pandas dataframe into a Dataset for pytorch.
# 2. To make the lightning training code simpler, we will build a [LightningDataModule](https://lightning.ai/docs/pytorch/stable/data/datamodule.html) (`PendulumDataModule`) and a [LightningModule](https://lightning.ai/docs/pytorch/stable/common/lightning_module.html) (`FFNForecaster`).


# +
class DataFrameDataset(Dataset):
    """A dataset from a pandas dataframe.

    For a given pandas dataframe, this generates a pytorch
    compatible dataset by sliding in time dimension.

    ```python
    ds = DataFrameDataset(
        dataframe=df, history_length=10, horizon=2
    )
    ```

    :param dataframe: input dataframe with a DatetimeIndex.
    :param history_length: length of input X in time dimension
        in the final Dataset class.
    :param horizon: number of steps to be forecasted.
    """

    def __init__(self, dataframe: pd.DataFrame, history_length: int, horizon: int):
        super().__init__()
        self.dataframe = dataframe
        self.history_length = history_length
        self.horzion = horizon
        self.dataframe_rows = len(self.dataframe)
        self.length = self.dataframe_rows - self.history_length - self.horzion

    def moving_slicing(self, idx: int) -> Tuple[np.ndarray, np.ndarray]:
        x, y = (
            self.dataframe[idx : self.history_length + idx].values,
            self.dataframe[
                self.history_length + idx : self.history_length + self.horzion + idx
            ].values,
        )
        return x, y

    def _validate_dataframe(self) -> None:
        """Validate the input dataframe.

        - We require the dataframe index to be DatetimeIndex.
        - This dataset is null aversion.
        - Dataframe index should be sorted.
        """

        if not isinstance(
            self.dataframe.index, pd.core.indexes.datetimes.DatetimeIndex
        ):
            raise TypeError(
                "Type of the dataframe index is not DatetimeIndex"
                f": {type(self.dataframe.index)}"
            )

        has_na = self.dataframe.isnull().values.any()

        if has_na:
            logger.warning("Dataframe has null")

        has_index_sorted = self.dataframe.index.equals(
            self.dataframe.index.sort_values()
        )

        if not has_index_sorted:
            logger.warning("Dataframe index is not sorted")

    def __getitem__(self, idx: int) -> Tuple[np.ndarray, np.ndarray]:
        if isinstance(idx, slice):
            if (idx.start < 0) or (idx.stop >= self.length):
                raise IndexError(f"Slice out of range: {idx}")
            step = idx.step if idx.step is not None else 1
            return [self.moving_slicing(i) for i in range(idx.start, idx.stop, step)]
        else:
            if idx >= self.length:
                raise IndexError("End of dataset")
            return self.moving_slicing(idx)

    def __len__(self) -> int:
        return self.length


class PendulumDataModule(L.LightningDataModule):
    def __init__(
        self,
        history_length: int,
        horizon: int,
        dataframe: pd.DataFrame,
        test_fraction: float = 0.3,
        val_fraction: float = 0.1,
        batch_size: int = 32,
    ):
        super().__init__()
        self.history_length = history_length
        self.horizon = horizon
        self.batch_size = batch_size
        self.dataframe = dataframe
        self.test_fraction = test_fraction
        self.val_fraction = val_fraction

        self.train_dataset, self.val_dataset = self.split_train_val(
            self.train_val_dataset
        )

    @cached_property
    def df_length(self):
        return len(self.dataframe)

    @cached_property
    def df_test_length(self):
        return int(self.df_length * self.test_fraction)

    @cached_property
    def df_train_val_length(self):
        return self.df_length - self.df_test_length

    @cached_property
    def train_val_dataframe(self):
        return self.dataframe.iloc[: self.df_train_val_length]

    @cached_property
    def test_dataframe(self):
        return self.dataframe.iloc[self.df_train_val_length :]

    @cached_property
    def train_val_dataset(self):
        return DataFrameDataset(
            dataframe=self.train_val_dataframe,
            history_length=self.history_length,
            horizon=self.horizon,
        )

    @cached_property
    def test_dataset(self):
        return DataFrameDataset(
            dataframe=self.test_dataframe,
            history_length=self.history_length,
            horizon=self.horizon,
        )

    def split_train_val(self, dataset: Dataset):
        return torch.utils.data.random_split(
            dataset, [1 - self.val_fraction, self.val_fraction]
        )

    def train_dataloader(self):
        return DataLoader(
            dataset=self.train_dataset, batch_size=self.batch_size, shuffle=True
        )

    def test_dataloader(self):
        return DataLoader(
            dataset=self.test_dataset, batch_size=self.batch_size, shuffle=True
        )

    def val_dataloader(self):
        return DataLoader(
            dataset=self.val_dataset, batch_size=self.batch_size, shuffle=True
        )

    def predict_dataloader(self):
        return DataLoader(
            dataset=self.test_dataset, batch_size=len(self.test_dataset), shuffle=False
        )


class NODEForecaster(L.LightningModule):
    def __init__(self, model: nn.Module):
        super().__init__()
        self.model = model

        self.neural_ode = NeuralODE(
            self.model.network,
            sensitivity="adjoint",
            solver="dopri5",
            atol_adjoint=1e-4,
            rtol_adjoint=1e-4,
        )
        self.time_span = self.model.time_span
        self.horizon = self.model.horizon

    def configure_optimizers(self):
        optimizer = torch.optim.SGD(self.parameters(), lr=1e-3)
        return optimizer

    def training_step(self, batch, batch_idx):
        x, y = batch
        x = x.squeeze().type(self.dtype)
        y = y.squeeze(-1).type(self.dtype)

        t_, y_hat = self.neural_ode(x, self.time_span)
        y_hat = y_hat[-1, ..., -self.horizon :]

        loss = nn.functional.mse_loss(y_hat, y)
        self.log_dict({"train_loss": loss}, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        x = x.squeeze().type(self.dtype)
        y = y.squeeze(-1).type(self.dtype)

        t_, y_hat = self.neural_ode(x, self.time_span)
        y_hat = y_hat[-1, ..., -self.horizon :]

        loss = nn.functional.mse_loss(y_hat, y)
        self.log_dict({"val_loss": loss}, prog_bar=True)
        return loss

    def predict_step(self, batch, batch_idx):
        x, y = batch
        x = x.squeeze().type(self.dtype)
        y = y.squeeze(-1).type(self.dtype)

        t_, y_hat = self.neural_ode(x, self.time_span)
        y_hat = y_hat[-1, ..., -self.horizon :]

        return x, y_hat

    def forward(self, x):
        x = x.squeeze().type(self.dtype)
        t_, y_hat = self.neural_ode(x, self.time_span)
        y_hat = y_hat[-1, ..., -self.horizon :]
        return x, y_hat


# -

# ### Data, Model and Training

# #### DataModule

ds = DataFrameDataset(dataframe=df, history_length=history_length, horizon=horizon)

len(ds)

pdm = PendulumDataModule(
    history_length=history_length, horizon=horizon, dataframe=df[["theta"]]
)

# #### LightningModule

# +
ts_model_params = TSNODEParams(hidden_widths=[256], time_span=torch.linspace(0, 1, 101))

ts_node = TSNODE(
    history_length=history_length,
    horizon=horizon,
    model_params=ts_model_params,
)

ts_node
# -

node_forecaster = NODEForecaster(model=ts_node)

# #### Trainer

trainer = L.Trainer(
    precision="32",
    max_epochs=10,
    min_epochs=5,
    callbacks=[
        EarlyStopping(monitor="val_loss", mode="min", min_delta=1e-4, patience=2)
    ],
)

# #### Fitting

trainer.fit(model=node_forecaster, datamodule=pdm)

# #### Retrieving Predictions

predictions = trainer.predict(model=node_forecaster, datamodule=pdm)

prediction_inputs = [i[0] for i in pdm.predict_dataloader()]
prediction_truths = [i[1].squeeze() for i in pdm.predict_dataloader()]

# ## Results

y_test_pred = predictions[0][1].squeeze().detach().numpy()
y_test_truth = prediction_truths[0].numpy()

# +
fig, ax = plt.subplots(figsize=(10, 6.18))

ax.plot(y_test_truth, label="truth")

ax.plot(y_test_pred, "--", label="predictions")

plt.legend()
# -

# To quantify the results, we compute a few metrics.

all_metrics = MetricCollection(
    MeanAbsoluteError(),
    MeanAbsolutePercentageError(),
    MeanSquaredError(),
    SymmetricMeanAbsolutePercentageError(),
)

all_metrics(predictions[0][1].squeeze().detach(), prediction_truths[0])
