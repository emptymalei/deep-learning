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
#     display_name: deep-learning-code
#     language: python
#     name: deep-learning-code
# ---

# # Transformer for Univariate Time Series Forecasting

# +
import math
from functools import cached_property
from typing import Dict, List, Optional

import pandas as pd

from torch.utils.data import Dataset


# +
class DataFrameDataset(Dataset):
    """A dataset from a pandas dataframe
    :param dataframe: input dataframe with a DatetimeIndex.
    :param contex_length: length of input in time dimension
    :param horizon: future length to be forecasted
    """

    def __init__(self, dataframe: pd.DataFrame, context_length: int, horizon: int):
        super().__init__()
        self.dataframe = dataframe
        self.context_length = context_length
        self.horzion = horizon
        self.dataframe_rows = len(self.dataframe)
        self.length = self.dataframe_rows - self.context_length - self.horzion

    def moving_slicing(self, idx):
        x, y = (
            self.dataframe[idx : self.context_length + idx].values,
            self.dataframe[
                self.context_length + idx : self.context_length + self.horzion + idx
            ].values,
        )
        return x, y

    def _validate_dataframe(self):
        """Validate the input dataframe.
        - We require the dataframe index to be DatetimeIndex.
        - This dataset is null aversion.
        - Dataframe index should be sorted.
        """

        if not isinstance(
            self.dataframe.index, pd.core.indexes.datetimes.DatetimeIndex
        ):
            raise TypeError(
                f"Type of the dataframe index is not DatetimeIndex: {type(self.dataframe.index)}"
            )

        has_na = self.dataframe.isnull().values.any()

        if has_na:
            logger.warning(f"Dataframe has null")

        has_index_sorted = self.dataframe.index.equals(
            self.dataframe.index.sort_values()
        )

        if not has_index_sorted:
            logger.warning(f"Dataframe index is not sorted")

    def __getitem__(self, idx):
        if isinstance(idx, slice):
            if (idx.start < 0) or (idx.stop >= self.length):
                raise IndexError(f"Slice out of range: {idx}")
            step = idx.step if idx.step is not None else 1
            return [self.moving_slicing(i) for i in range(idx.start, idx.stop, step)]
        else:
            if idx >= self.length:
                raise IndexError("End of dataset")
            return self.moving_slicing(idx)

    def __len__(self):
        return self.length


class Pendulum:
    """Class for generating time series data for a pendulum.

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


# -

pen = Pendulum(length=1)

df = pd.DataFrame(pen(100, 100, beta=0.001))

df.head()

df.plot(x="t", y="theta")


# ## Model

# +
from torch import nn
import torch

import dataclasses


# +
@dataclasses.dataclass
class TSTransformerParams:
    d_model: int = 512
    nhead: int = 8
    num_encoder_layers: int = 6
    dim_feedforward: int = 2048
    dropout: int = 0.1
    batch_first: bool = True
    norm_first: bool = True


@dataclasses.dataclass
class TSCovariateTransformerParams:
    d_model: int = 512
    nhead: int = 8
    num_encoder_layers: int = 6
    num_decoder_layers: int = 6
    dim_feedforward: int = 2048
    dropout: int = 0.1
    batch_first: bool = True
    norm_first: bool = True


# -


class PositionalEncoding(nn.Module):
    """
    From https://pytorch.org/tutorials/beginner/transformer_tutorial.html
    """

    def __init__(self, d_model: int, dropout: float = 0.1, max_len: int = 5000):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model)
        )
        pe = torch.zeros(max_len, d_model)
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer("pe", pe)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        :param x: Tensor, shape `[seq_len, batch_size, embedding_dim]`
        """
        x = x + self.pe[: x.size(0)]
        return self.dropout(x)


# +
class TSTransformer(nn.Module):
    def __init__(self, history_length: int, horizon: int, transformer_params):
        super().__init__()
        self.transformer_params = transformer_params
        self.history_length = history_length
        self.horizon = horizon

        self.regulate_input = nn.Linear(
            self.history_length, self.transformer_params.d_model
        )
        self.regulate_output = nn.Linear(self.transformer_params.d_model, self.horizon)
        self.positional_encoder = PositionalEncoding(self.transformer_params.d_model)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=512, nhead=8, batch_first=True
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=6)

    @property
    def transformer_config(self):
        return dataclasses.asdict(self.transformer_params)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.regulate_input(x)
        x = self.positional_encoder(x)

        encoder_state = self.encoder(x)

        return self.regulate_output(encoder_state)


# -


# ## Training

import lightning as L
from torch.utils.data import DataLoader

history_length = 100
horizon = 5

ts_transformer_params = TSTransformerParams()

# +
ts_transformer = TSTransformer(
    history_length=history_length,
    horizon=horizon,
    transformer_params=ts_transformer_params,
)

ts_transformer
# -

ds = DataFrameDataset(dataframe=df, context_length=history_length, horizon=horizon)

len(ds)


# +
class PendulumDataModule(L.LightningDataModule):
    def __init__(
        self,
        history_length: int,
        horizon: int,
        dataframe: pd.DataFrame,
        test_fraction: float = 0.9,
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
            context_length=self.history_length,
            horizon=self.horizon,
        )

    @cached_property
    def test_dataset(self):
        return DataFrameDataset(
            dataframe=self.test_dataframe,
            context_length=self.history_length,
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
        return DataLoader(dataset=self.test_dataset, batch_size=self.batch_size)


class TransformerForecaster(L.LightningModule):
    def __init__(self, transformer: nn.Module):
        super().__init__()
        self.transformer = transformer

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=1e-3)
        return optimizer

    def training_step(self, batch, batch_idx):
        x, y = batch
        x = x.squeeze().type(self.dtype)
        y = y.squeeze().type(self.dtype)

        y_hat = self.transformer(x)

        loss = nn.functional.mse_loss(y_hat, y)
        self.log_dict({"train_loss": loss}, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        x = x.squeeze().type(self.dtype)
        y = y.squeeze().type(self.dtype)

        y_hat = self.transformer(x)

        loss = nn.functional.mse_loss(y_hat, y)
        self.log_dict({"val_loss": loss}, prog_bar=True)
        return loss

    def predict_step(self, batch, batch_idx):
        x, y = batch
        x = x.squeeze().type(self.dtype)
        y = y.squeeze().type(self.dtype)

        y_hat = self.transformer(x)
        return x, y_hat

    def forward(self, x):
        x = x.squeeze().type(self.dtype)
        return x, self.transformer(x)


# -

pdm = PendulumDataModule(
    history_length=history_length, horizon=horizon, dataframe=df[["theta"]]
)

transformer_forecaster = TransformerForecaster(transformer=ts_transformer)

from lightning.pytorch.callbacks.early_stopping import EarlyStopping

trainer = L.Trainer(
    precision="32",
    max_epochs=100,
    min_epochs=10,
    callbacks=[
        EarlyStopping(monitor="val_loss", mode="min", min_delta=0.00, patience=3)
    ],
)

trainer.fit(model=transformer_forecaster, datamodule=pdm)

predictiosn = trainer.predict(model=transformer_forecaster, datamodule=pdm)

prediction_inputs = [i[0] for i in pdm.predict_dataloader()]
prediction_truths = [i[1].squeeze() for i in pdm.predict_dataloader()]

prediction_inputs[0].shape

transformer_forecaster(prediction_inputs[0])[-1]

# ## Check Results

import matplotlib.pyplot as plt
import numpy as np

# +
step = 0
batch_idx = 0

x_test, y_test_pred = transformer_forecaster(prediction_inputs[step])
x_test = x_test.detach()[batch_idx].numpy()
y_test_pred = y_test_pred.detach()[batch_idx].numpy()
y_test_truth = prediction_truths[0][batch_idx].numpy()
# -

x_test.shape, y_test_pred.shape, y_test_truth.shape

# +
fig, ax = plt.subplots(figsize=(10, 6.18))

ax.plot(np.concatenate([x_test, y_test_truth]), label="truth")

ax.plot(np.concatenate([x_test, y_test_pred]), label="predictions")

plt.legend()
# -


# ## Debug


class TSCovariateTransformer(nn.Module):
    def __init__(self, history_length: int, horizon: int, transformer_params):
        super().__init__()
        self.transformer_params = transformer_params
        self.history_length = history_length
        self.horizon = horizon

    @property
    def transformer_config(self):
        return dataclasses.asdict(self.transformer_params)

    @property
    def regulate_input(self):
        return nn.Linear(self.history_length, self.transformer_params.d_model)

    @property
    def regulate_covariate(self):
        return nn.Linear(self.horizon, self.transformer_params.d_model)

    @property
    def regulate_output(self):
        return nn.Linear(self.transformer_params.d_model, self.horizon)

    @property
    def positional_encoder(self):
        return PositionalEncoding(self.transformer_params.d_model)

    @property
    def encoder(self):
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=512, nhead=8, batch_first=True
        )
        return nn.TransformerEncoder(encoder_layer, num_layers=6)

    @property
    def decoder(self):
        decoder_layer = nn.TransformerDecoderLayer(d_model=512, nhead=8)
        return nn.TransformerDecoder(decoder_layer, num_layers=6)

    def forward(self, x: torch.Tensor, c: torch.Tensor) -> torch.Tensor:
        x = self.regulate_input(x)
        #         x = self.positional_encoder(x)

        # switch batch and time for attention
        # x = x.permute(1, 0, 2)

        encoder_state = self.encoder(x)

        c = self.regulate_covariate(c)

        decoder_state = self.decoder(c, encoder_state)

        return self.regulate_output(decoder_state)
