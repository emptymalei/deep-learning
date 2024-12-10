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
# Use VAE to generate time series data. In this example, we will train a VAE model to sinusoidal time series data. The overall structure of the model is shown below:
#
# ```mermaid
# graph TD
# data["Time Series Chunks"] --> E[Encoder]
#     E --> L[Latent Space]
#     L --> D[Decoder]
#     D --> gen["Generated Time Series Chunks"]
# ```
#
# Reference:
# https://github.com/wangyz1999/timeVAE-pytorch

import dataclasses

# +
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
from ts_dl_utils.datasets.pendulum import Pendulum

# -

# ## Data
#
# We will reuse our classic pendulum dataset.

pen = Pendulum(length=100)

df = pd.DataFrame(pen(300, 30, initial_angle=1, beta=0.00001))

df["theta"] = df["theta"] + 2

# +
_, ax = plt.subplots(figsize=(10, 6.18))

df.head(100).plot(x="t", y="theta", ax=ax)

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
    """Lightning DataModule for Time Series VAE.

    This data module takes a pandas dataframe and generates
    the corresponding dataloaders for training, validation and
    testing.

    ```python
    time_vae_dm_example = TimeVAEDataModule(
        window_size=30, dataframe=df[["theta"]], batch_size=32
    )
    ```
    """

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
            persistent_workers=(True if self.num_workers > 0 else False),
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
            persistent_workers=(True if self.num_workers > 0 else False),
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
    """Parameters for VAEEncoder and VAEDecoder

    :param hidden_layer_sizes: list of hidden layer sizes
    :param latent_size: latent space dimension
    :param sequence_length: input sequence length
    :param n_features: number of features
    """

    hidden_layer_sizes: List[int]
    latent_size: int
    sequence_length: int
    n_features: int = 1

    @cached_property
    def data_size(self) -> int:
        """The dimension of the input data
        when flattened.
        """
        return self.sequence_length * self.n_features

    def asdict(self) -> dict:
        return dataclasses.asdict(self)


class VAEMLPEncoder(nn.Module):
    """MLP Encoder of TimeVAE"""

    def __init__(self, params: VAEParams):
        super().__init__()

        self.params = params

        encode_layer_sizes = [self.params.data_size] + self.params.hidden_layer_sizes
        self.layers_used_to_encode = [
            self._linear_block(size_in, size_out)
            for size_in, size_out in zip(
                encode_layer_sizes[:-1], encode_layer_sizes[1:]
            )
        ]
        self.encode = nn.Sequential(*self.layers_used_to_encode)
        encoded_size = self.params.hidden_layer_sizes[-1]
        self.z_mean_layer = nn.Linear(encoded_size, self.params.latent_size)
        self.z_log_var_layer = nn.Linear(encoded_size, self.params.latent_size)

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


class VAEEncoder(nn.Module):
    """Encoder of TimeVAE

    ```python
    encoder = VAEEncoder(
        VAEParams(
            hidden_layer_sizes=[40, 30],
            latent_size=10,
            sequence_length=50
        )
    )
    ```

    :param params: parameters for the encoder
    """

    def __init__(self, params: VAEParams):
        super().__init__()

        self.params = params
        self.hparams = params.asdict()

        encode_layer_sizes = [self.params.n_features] + self.params.hidden_layer_sizes
        self.layers_used_to_encode = [
            self._conv_block(size_in, size_out)
            for size_in, size_out in zip(
                encode_layer_sizes[:-1], encode_layer_sizes[1:]
            )
        ] + [nn.Flatten()]
        self.encode = nn.Sequential(*self.layers_used_to_encode)
        encoded_size = self.cal_conv1d_output_dim() * self.params.hidden_layer_sizes[-1]
        self.z_mean_layer = nn.Linear(encoded_size, self.params.latent_size)
        self.z_log_var_layer = nn.Linear(encoded_size, self.params.latent_size)

    def forward(
        self, x: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        batch_size, _, _ = x.size()
        x = x.transpose(1, 2)
        x = self.encode(x)

        z_mean = self.z_mean_layer(x).view(
            batch_size, self.params.n_features, self.params.latent_size
        )
        z_log_var = self.z_log_var_layer(x).view(
            batch_size, self.params.n_features, self.params.latent_size
        )
        epsilon = torch.randn(
            batch_size, self.params.n_features, self.params.latent_size
        ).type_as(x)
        z = z_mean + torch.exp(0.5 * z_log_var) * epsilon

        return z_mean, z_log_var, z

    def _linear_block(self, size_in: int, size_out: int) -> nn.Module:
        return nn.Sequential(*[nn.Linear(size_in, size_out), nn.ReLU()])

    def _conv_block(self, size_in: int, size_out: int) -> nn.Module:
        return nn.Sequential(
            *[
                nn.Conv1d(size_in, size_out, kernel_size=3, stride=2, padding=1),
                nn.ReLU(),
            ]
        )

    def cal_conv1d_output_dim(self) -> int:
        """the output dimension of all the Conv1d layers"""
        output_size = self.params.sequence_length * self.params.n_features

        for l in self.layers_used_to_encode:
            if l._get_name() == "Conv1d":
                output_size = self._conv1d_output_dim(l, output_size)
            elif l._get_name() == "Sequential":
                for l2 in l:
                    if l2._get_name() == "Conv1d":
                        output_size = self._conv1d_output_dim(l2, output_size)

        return output_size

    def _conv1d_output_dim(self, layer: nn.Module, input_size: int) -> int:
        """Formula to calculate
        the output size of Conv1d layer
        """
        return (
            (input_size + 2 * layer.padding[0] - layer.kernel_size[0])
            // layer.stride[0]
        ) + 1


# +
mlp_encoder = VAEMLPEncoder(
    VAEParams(hidden_layer_sizes=[40, 30], latent_size=10, sequence_length=50)
)

[i.size() for i in mlp_encoder(torch.ones(32, 50, 1))], mlp_encoder(
    torch.ones(32, 50, 1)
)[-1]

# +
encoder = VAEEncoder(
    VAEParams(hidden_layer_sizes=[40, 30], latent_size=10, sequence_length=50)
)

[i.size() for i in encoder(torch.ones(32, 50, 1))], encoder(torch.ones(32, 50, 1))[-1]


# -


class VAEDecoder(nn.Module):
    """Decoder of TimeVAE

    ```python
    decoder = VAEDecoder(
        VAEParams(
            hidden_layer_sizes=[30, 40],
            latent_size=10,
            sequence_length=50,
        )
    )
    ```

    :param params: parameters for the decoder
    """

    def __init__(self, params: VAEParams):
        super().__init__()

        self.params = params
        self.hparams = params.asdict()

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

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        output = self.decode(z)
        return output.view(-1, self.params.sequence_length, self.params.n_features)

    def _linear_block(self, size_in: int, size_out: int) -> nn.Module:
        """create linear block based on the specified sizes"""
        return nn.Sequential(*[nn.Linear(size_in, size_out), nn.Softplus()])


# +
decoder = VAEDecoder(
    VAEParams(hidden_layer_sizes=[30, 40], latent_size=10, sequence_length=50)
)

decoder(torch.ones(32, 1, 10)).size()


# -


class VAE(nn.Module):
    """VAE model with encoder and decoder

    :param encoder: encoder module
    :param decoder: decoder module
    """

    def __init__(self, encoder: nn.Module, decoder: nn.Module):
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.hparams = {
            **{f"encoder_{k}": v for k, v in self.encoder.hparams.items()},
            **{f"decoder_{k}": v for k, v in self.decoder.hparams.items()},
        }

    def forward(
        self, x: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        z_mean, z_log_var, z = self.encoder(x)
        x_reconstructed = self.decoder(z)
        return x_reconstructed, z_mean, z_log_var


class VAEModel(L.LightningModule):
    """VAE model using VAEEncoder, VAEDecoder, and VAE

    :param model: VAE model
    :param reconstruction_weight: weight for the reconstruction loss
    :param learning_rate: learning rate for the optimizer
    :param scheduler_max_epochs: maximum epochs for the scheduler
    """

    def __init__(
        self,
        model: VAE,
        reconstruction_weight: float = 1.0,
        learning_rate: float = 1e-3,
        scheduler_max_epochs: int = 10000,
    ):
        super().__init__()
        self.model = model
        self.reconstruction_weight = reconstruction_weight
        self.learning_rate = learning_rate
        self.scheduler_max_epochs = scheduler_max_epochs

        self.hparams.update(
            {
                **model.hparams,
                **{
                    "reconstruction_weight": reconstruction_weight,
                    "learning_rate": learning_rate,
                    "scheduler_max_epochs": scheduler_max_epochs,
                },
            }
        )
        self.save_hyperparameters(self.hparams)

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
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
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

    def configure_optimizers(self) -> dict:
        optimizer = torch.optim.SGD(self.parameters(), lr=self.learning_rate)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=self.scheduler_max_epochs
        )

        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "monitor": "train_loss",
                "interval": "step",
                "frequency": 1,
            },
        }


# # Training
#

# +
window_size = 24
max_epochs = 2000

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

vae_model = VAEModel(
    vae,
    reconstruction_weight=3,
    scheduler_max_epochs=max_epochs * len(time_vae_dm.train_dataloader()),
)
# -

trainer = L.Trainer(
    precision="64",
    max_epochs=max_epochs,
    min_epochs=5,
    callbacks=[
        EarlyStopping(
            monitor="val_loss_total", mode="min", min_delta=1e-10, patience=10
        )
    ],
    logger=L.pytorch.loggers.TensorBoardLogger(
        save_dir="lightning_logs", name="time_vae_naive"
    ),
)

trainer.fit(model=vae_model, datamodule=time_vae_dm)

# ## Fitted Model

checkpoint_path = (
    "lightning_logs/time_vae_naive/version_29/checkpoints/epoch=1999-step=354000.ckpt"
)
vae_model_reloaded = VAEModel.load_from_checkpoint(checkpoint_path, model=vae)


for i in time_vae_dm.predict_dataloader():
    print(i.size())
    i_pred = vae_model_reloaded.model(i.float().cuda())
    break

i_pred[0].size()

import matplotlib.pyplot as plt

# +
_, ax = plt.subplots()

element = 4

ax.plot(i.detach().numpy()[element, :, 0])
ax.plot(i_pred[0].cpu().detach().numpy()[element, :, 0], "x-")
# -

# Data generation using the decoder.

sampling_z = torch.randn(
    2, vae_model_reloaded.model.encoder.params.latent_size
).type_as(vae_model_reloaded.model.encoder.z_mean_layer.weight)
sampling_x = vae_model_reloaded.model.decoder(sampling_z)

sampling_x.size()

# +
_, ax = plt.subplots()

for i in range(min(len(sampling_x), 4)):
    ax.plot(sampling_x.cpu().detach().numpy()[i, :, 0], "x-")

# -
