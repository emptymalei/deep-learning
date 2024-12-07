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

# # TimeVAE
#
# Reference:
# https://github.com/wangyz1999/timeVAE-pytorch

import dataclasses

# +
import os
from functools import cached_property
from typing import Dict, List, Tuple

import lightning as L
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from lightning.pytorch.callbacks.early_stopping import EarlyStopping
from loguru import logger
from torch import nn
from torch.utils.data import DataLoader, Dataset
from ts_dl_utils.datasets.dataset import DataFrameDataset
from ts_dl_utils.datasets.pendulum import Pendulum

# -

# ## Data

pen = Pendulum(length=100)

df = pd.DataFrame(pen(100, 100, initial_angle=1, beta=0.0001))

df["theta"] = df["theta"] + 2

# +
_, ax = plt.subplots(figsize=(10, 6.18))

df.plot(x="t", y="theta", ax=ax)
# -

df


def time_delay_embed(df: pd.DataFrame, window_size: int) -> pd.DataFrame:
    """embed time series into a time delay embedding space

    Time column `t` is required in the input data frame.

    :param df: original time series data frame
    :param window_size: window size for the time delay embedding
    """
    dfs_embedded = []

    for i in df.rolling(window_size):
        i_t = i.t.iloc[0]
        dfs_embedded.append(
            pd.DataFrame(i.reset_index(drop=True))
            .drop(columns=["t"])
            .T.reset_index(drop=True)
            # .rename(columns={"index": "name"})
            # .assign(t=i_t)
        )

    df_embedded = pd.concat(dfs_embedded[window_size - 1 :])

    return df_embedded


time_delay_embed(df, 3)


class TimeVAEDataset(Dataset):
    """A dataset from a pandas dataframe.

    For a given pandas dataframe, this generates a pytorch
    compatible dataset by sliding in time dimension.

    ```python
    ds = DataFrameDataset(
        dataframe=df, history_length=10, horizon=2
    )
    ```

    :param dataframe: input dataframe with a DatetimeIndex.
    :param window_size: length of time series slicing chunks
    """

    def __init__(
        self,
        dataframe: pd.DataFrame,
        window_size: int,
    ):
        super().__init__()
        self.dataframe = dataframe
        self.window_size = window_size
        self.dataframe_rows = len(self.dataframe)
        self.length = self.dataframe_rows - self.window_size + 1

    def moving_slicing(self, idx: int) -> np.ndarray:
        return self.dataframe[idx : self.window_size + idx].values

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


class TimeVAEDataModule(L.LightningDataModule):
    def __init__(
        self,
        window_size: int,
        dataframe: pd.DataFrame,
        test_fraction: float = 0.3,
        val_fraction: float = 0.1,
        batch_size: int = 32,
        num_workers: int = 0,
    ):
        super().__init__()
        self.window_size = window_size
        self.batch_size = batch_size
        self.dataframe = dataframe
        self.test_fraction = test_fraction
        self.val_fraction = val_fraction
        self.num_workers = num_workers

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
        return TimeVAEDataset(
            dataframe=self.train_val_dataframe,
            window_size=self.window_size,
        )

    @cached_property
    def test_dataset(self):
        return TimeVAEDataset(
            dataframe=self.test_dataframe,
            window_size=self.window_size,
        )

    def split_train_val(self, dataset: Dataset):
        return torch.utils.data.random_split(
            dataset, [1 - self.val_fraction, self.val_fraction]
        )

    def train_dataloader(self):
        return DataLoader(
            dataset=self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            persistent_workers=True if self.num_workers > 0 else False,
        )

    def test_dataloader(self):
        return DataLoader(
            dataset=self.test_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
        )

    def val_dataloader(self):
        return DataLoader(
            dataset=self.val_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            persistent_workers=True if self.num_workers > 0 else False,
        )

    def predict_dataloader(self):
        return DataLoader(
            dataset=self.test_dataset, batch_size=len(self.test_dataset), shuffle=False
        )


time_vae_dm_example = TimeVAEDataModule(
    window_size=30, dataframe=df[["theta"]], batch_size=32
)

len(list(time_vae_dm_example.train_dataloader()))

list(time_vae_dm_example.train_dataloader())[0].shape


# ## Model


@dataclasses.dataclass
class VAEParams:
    """Parameters for TimeVAE"""

    latent_size: int
    hidden_layer_sizes: List[int]


# +
@dataclasses.dataclass
class VAEParams:
    """Parameters for VAEEncoder and VAEDecoder"""

    hidden_layer_sizes: List[int]
    latent_size: int
    sequence_length: int
    n_features: int = 1

    @cached_property
    def data_size(self):
        return self.sequence_length * self.n_features


class VAEEncoder(nn.Module):
    """Encoder of TimeVAE"""

    def __init__(self, params: VAEParams):
        super().__init__()

        self.params = params

        encode_layer_sizes = [self.params.data_size] + self.params.hidden_layer_sizes
        self.encode = nn.Sequential(
            *[
                self._linear_block(size_in, size_out)
                for size_in, size_out in zip(
                    encode_layer_sizes[:-1], encode_layer_sizes[1:]
                )
            ]
        )
        self.z_mean_layer = nn.Linear(
            self.params.hidden_layer_sizes[-1], self.params.latent_size
        )
        self.z_log_var_layer = nn.Linear(
            self.params.hidden_layer_sizes[-1], self.params.latent_size
        )

    def forward(
        self, x: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        batch_size, _, _ = x.size()
        x = x.transpose(1, 2)
        x = self.encode(x)

        z_mean = self.z_mean_layer(x)
        z_log_var = self.z_log_var_layer(x)
        epsilon = torch.randn(
            batch_size, self.params.n_features, self.params.latent_size
        ).type_as(x)
        z = z_mean + torch.exp(0.5 * z_log_var) * epsilon

        return z_mean, z_log_var, z

    def _linear_block(self, size_in: int, size_out: int) -> nn.Module:
        return nn.Sequential(*[nn.Linear(size_in, size_out), nn.ReLU()])


# +
encoder = VAEEncoder(
    VAEParams(hidden_layer_sizes=[40, 30], latent_size=10, sequence_length=50)
)

[i.size() for i in encoder(torch.ones(32, 50, 1))], encoder(torch.ones(32, 50, 1))[-1]

# +
# @dataclasses.dataclass
# class VAEDecoderParams:
#     """Parameters for VAEDecoder"""
#     hidden_layer_sizes: List[int]
#     latent_size: int
#     sequence_length: int
#     n_features: int = 1

#     @property
#     def data_size(self):
#         return self.sequence_length * self.n_features


class VAEDecoder(nn.Module):
    """Decoder of TimeVAE"""

    def __init__(self, params: VAEParams):
        super().__init__()

        self.params = params

        decode_layer_sizes = (
            [self.params.latent_size]
            + self.params.hidden_layer_sizes
            + [self.params.data_size]
        )

        self.decode = nn.Sequential(
            *[
                self._linear_block(size_in, size_out)
                for size_in, size_out in zip(
                    decode_layer_sizes[:-1], decode_layer_sizes[1:]
                )
            ]
        )

    def forward(self, z):
        output = self.decode(z)
        return output.view(-1, self.params.sequence_length, self.params.n_features)

    def _linear_block(self, size_in: int, size_out: int) -> nn.Module:
        return nn.Sequential(*[nn.Linear(size_in, size_out), nn.ReLU()])


# +
decoder = VAEDecoder(
    VAEParams(hidden_layer_sizes=[30, 40], latent_size=10, sequence_length=50)
)

decoder(torch.ones(32, 1, 10)).size()


# -


class VAE(nn.Module):
    def __init__(self, encoder: nn.Module, decoder: nn.Module):
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder

    def forward(self, x):
        z_mean, z_log_var, z = self.encoder(x)
        x_reconstructed = self.decoder(z)
        return x_reconstructed, z_mean, z_log_var


class VAEModel(L.LightningModule):
    """VAE model using VAEEncoder, VAEDecoder, and VAE"""

    def __init__(self, model: VAE, reconstruction_weight: float = 1.0):
        super().__init__()
        self.model = model
        self.reconstruction_weight = reconstruction_weight

    def forward(self, x):
        return self.model(x)

    def training_step(self, batch: torch.Tensor, batch_idx: int) -> torch.Tensor:
        batch_reconstructed, z_mean, z_log_var = self.model(batch)
        loss_total, loss_reconstruction, loss_kl = self.loss(
            x=batch,
            x_reconstructed=batch_reconstructed,
            z_mean=z_mean,
            z_log_var=z_log_var,
        )

        self.log_dict(
            {
                "train_loss_total": loss_total,
                "train_loss_reconstruction": loss_reconstruction,
                "train_loss_kl": loss_kl,
            }
        )

        return loss_total

    def validation_step(self, batch: torch.Tensor, batch_idx: int) -> torch.Tensor:
        batch_reconstructed, z_mean, z_log_var = self.model(batch)
        loss_total, loss_reconstruction, loss_kl = self.loss(
            x=batch,
            x_reconstructed=batch_reconstructed,
            z_mean=z_mean,
            z_log_var=z_log_var,
        )
        self.log_dict(
            {
                "val_loss_total": loss_total,
                "val_loss_reconstruction": loss_reconstruction,
                "val_loss_kl": loss_kl,
            }
        )

        return loss_total

    def test_step(self, batch: torch.Tensor, batch_idx: int) -> torch.Tensor:
        batch_reconstructed, z_mean, z_log_var = self.model(batch)
        loss_total, loss_reconstruction, loss_kl = self.loss(
            x=batch,
            x_reconstructed=batch_reconstructed,
            z_mean=z_mean,
            z_log_var=z_log_var,
        )
        self.log_dict(
            {
                "test_loss_total": loss_total,
                "test_loss_reconstruction": loss_reconstruction,
                "test_loss_kl": loss_kl,
            }
        )
        return loss_total

    def loss(
        self,
        x: torch.Tensor,
        x_reconstructed: torch.Tensor,
        z_log_var: torch.Tensor,
        z_mean: torch.Tensor,
    ):
        loss_reconstruction = self.reconstruction_loss(x, x_reconstructed)
        loss_kl = -0.5 * torch.sum(1 + z_log_var - z_mean**2 - z_log_var.exp())
        loss_total = self.reconstruction_weight * loss_reconstruction + loss_kl

        return (
            loss_total / x.size(0),
            loss_reconstruction / x.size(0),
            loss_kl / x.size(0),
        )

    def reconstruction_loss(
        self, x: torch.Tensor, x_reconstructed: torch.Tensor
    ) -> torch.Tensor:
        """Reconstruction loss for VAE.

        $$
        \sum_{i=1}^{N} (x_i - x_{reconstructed_i})^2
        + \sum_{i=1}^{N} (\mu_i - \mu_{reconstructed_i})^2
        $$
        """
        loss = torch.sum((x - x_reconstructed) ** 2) + torch.sum(
            (torch.mean(x, dim=1) - torch.mean(x_reconstructed, dim=1)) ** 2
        )

        return loss

    def configure_optimizers(self) -> torch.optim.Optimizer:
        optimizer = torch.optim.SGD(self.parameters(), lr=1e-3)

        return optimizer


# # Training
#

# +
window_size = 24

time_vae_dm = TimeVAEDataModule(
    window_size=window_size, dataframe=df[["theta"]], batch_size=32
)

# +
vae = VAE(
    encoder=VAEEncoder(
        VAEParams(
            hidden_layer_sizes=[200, 100, 50],
            latent_size=8,
            sequence_length=window_size,
        )
    ),
    decoder=VAEDecoder(
        VAEParams(
            hidden_layer_sizes=[30, 50, 100], latent_size=8, sequence_length=window_size
        )
    ),
)

vae_model = VAEModel(vae, reconstruction_weight=1)
# -

trainer = L.Trainer(
    precision="64",
    max_epochs=2000,
    min_epochs=5,
    # callbacks=[
    #     EarlyStopping(monitor="val_loss_total", mode="min", min_delta=1e-10, patience=10)
    # ],
    logger=L.pytorch.loggers.TensorBoardLogger(
        save_dir="lightning_logs", name="time_vae_naive"
    ),
)

trainer.fit(model=vae_model, datamodule=time_vae_dm)

for i in time_vae_dm.predict_dataloader():
    print(i.size())
    i_pred = vae_model.model(i)
    break

i_pred[0].size()

import matplotlib.pyplot as plt

# +
_, ax = plt.subplots()

element = 100

ax.plot(i.detach().numpy()[element, :, 0])
ax.plot(i_pred[0].detach().numpy()[element, :, 0], "x-")
