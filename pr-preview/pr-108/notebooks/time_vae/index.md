# TimeVAE

Use VAE to generate time series data. In this example, we will train a VAE model to sinusoidal time series data. The overall structure of the model is shown below:

```
graph TD
data["Time Series Chunks"] --> E[Encoder]
    E --> L[Latent Space]
    L --> D[Decoder]
    D --> gen["Generated Time Series Chunks"]
```

Reference: [https://github.com/wangyz1999/timeVAE-pytorch](<https://github.com/wangyz1999/timeVAE-pytorch>)

```
import dataclasses
```

```
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
```

## Data

We will reuse our classic pendulum dataset.

```
pen = Pendulum(length=100)
```

```
df = pd.DataFrame(pen(300, 30, initial_angle=1, beta=0.00001))
```

```
df["theta"] = df["theta"] + 2
```

```
_, ax = plt.subplots(figsize=(10, 6.18))

df.head(100).plot(x="t", y="theta", ax=ax)
```

```
_, ax = plt.subplots(figsize=(10, 6.18))

df.plot(x="t", y="theta", ax=ax)
```

```
<Axes: xlabel='t'>
```

![output](<https://dl.leima.is/notebooks/assets/time_vae/cell8-1.png>)

```
df
```

|  | t | theta |
| --- | --- | --- |
| 0 | 0.000000 | 3.000000 |
| 1 | 0.200607 | 2.998007 |
| 2 | 0.401213 | 2.992075 |
| 3 | 0.601820 | 2.982228 |
| 4 | 0.802427 | 2.968505 |
| ... | ... | ... |
| 9995 | 2005.063647 | 2.778265 |
| 9996 | 2005.264254 | 2.792591 |
| 9997 | 2005.464861 | 2.803789 |
| 9998 | 2005.665467 | 2.811815 |
| 9999 | 2005.866074 | 2.816636 |

10000 rows × 2 columns

```
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
```

```
time_delay_embed(df, 3)
```

|  | 0 | 1 | 2 |
| --- | --- | --- | --- |
| 0 | 3.000000 | 2.998007 | 2.992075 |
| 0 | 2.998007 | 2.992075 | 2.982228 |
| 0 | 2.992075 | 2.982228 | 2.968505 |
| 0 | 2.982228 | 2.968505 | 2.950961 |
| 0 | 2.968505 | 2.950961 | 2.929665 |
| ... | ... | ... | ... |
| 0 | 2.740464 | 2.760867 | 2.778265 |
| 0 | 2.760867 | 2.778265 | 2.792591 |
| 0 | 2.778265 | 2.792591 | 2.803789 |
| 0 | 2.792591 | 2.803789 | 2.811815 |
| 0 | 2.803789 | 2.811815 | 2.816636 |

9998 rows × 3 columns

````
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
````

````
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
````

```
time_vae_dm_example = TimeVAEDataModule(
    window_size=30, dataframe=df[["theta"]], batch_size=32
)
```

```
len(list(time_vae_dm_example.train_dataloader()))
```

```
list(time_vae_dm_example.train_dataloader())[0].shape
```

```
197
```

## Model

```
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
```

```
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
```

````
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
````

```
mlp_encoder = VAEMLPEncoder(
    VAEParams(hidden_layer_sizes=[40, 30], latent_size=10, sequence_length=50)
)

[i.size() for i in mlp_encoder(torch.ones(32, 50, 1))], mlp_encoder(
    torch.ones(32, 50, 1)
)[-1]
```

```
encoder = VAEEncoder(
    VAEParams(hidden_layer_sizes=[40, 30], latent_size=10, sequence_length=50)
)

[i.size() for i in encoder(torch.ones(32, 50, 1))], encoder(torch.ones(32, 50, 1))[-1]
```

```
([torch.Size([32, 1, 10]), torch.Size([32, 1, 10]), torch.Size([32, 1, 10])],
 tensor([[[ 0.7512, -0.2464, -1.5799, -0.3591,  0.3991,  0.8255, -0.6925,
           -0.2539,  0.1998, -0.1732]],

         [[-0.1898,  0.6519,  1.3573,  0.4369, -0.7016, -0.2748, -0.3117,
            1.2254,  1.2868, -1.4186]],

         [[-0.7966, -0.9899, -0.9354,  1.0457, -1.8151,  0.9798, -0.4217,
            0.4817,  0.6745, -1.1424]],

         [[ 0.7722,  1.0069, -0.1242, -0.1579,  3.1850,  0.3993,  1.6327,
           -2.2345,  0.8184, -1.8291]],

         [[ 0.6430, -0.5761,  0.9650,  1.1978, -0.0064, -1.2288,  0.2022,
            1.5347, -1.2524,  1.2249]],

         [[-0.2916,  0.6617, -2.6156, -0.4596,  0.4995,  0.2771, -1.0398,
            1.6406, -0.5922, -0.8694]],

         [[-0.7358, -0.6580,  0.4102,  1.0043, -1.1620,  1.7000,  0.6870,
           -2.5108,  1.3517, -1.2280]],

         [[-2.0802, -0.0234, -0.7984,  0.7578,  0.8384, -0.6166,  1.0604,
           -0.2086,  0.3403, -0.2937]],

         [[ 1.5776,  0.2911, -0.3327, -0.5813, -2.0566,  1.2946, -0.4778,
           -1.2812, -0.9236,  0.4816]],

         [[ 0.3073,  0.1833, -1.0332, -0.9702, -0.1414, -0.3407, -0.4733,
            0.0359, -0.9915,  0.6481]],

         [[ 0.8189,  0.8177, -1.5599,  0.8721, -0.0073, -0.3176,  0.9012,
            0.9449,  0.5305,  0.8372]],

         [[-1.2151, -0.1610, -0.7059,  0.9138,  1.2397,  1.2807, -0.5464,
           -0.9129,  1.5476, -1.7536]],

         [[ 1.4121, -1.0689, -2.2029,  0.1702,  2.0124,  0.1258, -0.2963,
           -1.5757,  0.9311, -0.3869]],

         [[-1.7114,  0.0985,  1.3061, -0.3364,  0.9112,  1.3963,  0.7253,
            1.1977, -0.2153, -0.2129]],

         [[-0.0213, -1.7831,  0.0819, -1.6534, -0.0161,  0.6107, -0.1698,
            1.0649, -2.1362,  0.1544]],

         [[-1.6537,  0.3799,  0.6110, -0.2150,  1.5642, -1.3898, -0.6645,
            0.1034, -1.9392, -0.7975]],

         [[-2.3527, -0.9559, -0.0141,  0.0087, -0.6443, -1.2371,  0.3701,
            0.3806, -0.3413, -0.1965]],

         [[ 0.3955,  1.0094, -0.0092, -1.3375,  1.2194, -0.2048, -0.0285,
           -0.6014, -0.4264,  0.7233]],

         [[-0.7148,  1.4312,  0.1843, -0.5086, -0.6581, -0.3653, -0.7701,
           -0.5338, -0.1859, -0.7277]],

         [[-0.0872,  0.5067, -1.7811, -1.2165,  0.7887,  0.9671, -0.8803,
           -0.8274,  0.5897,  0.9491]],

         [[ 1.4125,  1.0227, -1.7007,  0.2469,  0.8499,  0.5016,  0.1287,
           -1.9992, -1.1804,  0.1232]],

         [[-0.8195,  0.3970, -0.1237,  0.3717, -0.2569, -1.0386,  0.3947,
            0.0796, -0.6385, -1.2369]],

         [[-0.8912,  0.0222,  0.1747, -0.0088, -1.7169, -1.2536, -0.7857,
            0.1664,  0.4925, -0.2927]],

         [[ 0.0163,  0.4311, -0.4547,  0.3656, -0.1896, -0.0171,  0.3149,
            0.7186,  0.1870, -1.0573]],

         [[-0.5779, -1.1715, -0.1330,  0.6739, -1.2915, -2.6458,  0.5481,
           -1.4053,  0.2551, -1.8555]],

         [[-0.5961, -0.2198, -0.6367, -1.1569,  0.3316, -0.2596, -0.1730,
            0.0328,  1.1898,  0.9066]],

         [[ 0.2193, -0.2441,  1.6414, -1.4026, -0.5100, -0.5528, -3.2889,
           -1.3575,  0.1371,  0.1426]],

         [[-0.8817,  0.4769, -0.5507,  0.6523,  1.1832, -1.3601, -1.2146,
           -0.8998,  0.4066, -1.2781]],

         [[-1.7669, -0.5484, -0.5710, -1.3032, -0.6533, -0.1837,  0.1647,
            0.0847,  0.6987, -1.9335]],

         [[ 0.9263,  0.4973,  0.2373, -0.4195, -1.3269, -0.6560,  0.1802,
            0.5625, -0.2468, -1.2317]],

         [[-0.1505, -0.5553,  0.7748,  0.6992,  1.1973, -0.6109,  0.6687,
           -0.6004, -0.1196, -0.6409]],

         [[ 0.2727,  0.3247, -1.4497,  0.1855,  1.1862,  0.3328, -0.6967,
            1.2210, -0.7328, -0.1885]]], grad_fn=<AddBackward0>))
```

````
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
````

```
decoder = VAEDecoder(
    VAEParams(hidden_layer_sizes=[30, 40], latent_size=10, sequence_length=50)
)

decoder(torch.ones(32, 1, 10)).size()
```

```
torch.Size([32, 50, 1])
```

```
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
```

```
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
        r"""Reconstruction loss for VAE.

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
```

# Training

```
window_size = 24
max_epochs = 2000

time_vae_dm = TimeVAEDataModule(
    window_size=window_size, dataframe=df[["theta"]], batch_size=32
)
```

```
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
```

```
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
```

```
GPU available: True (mps), used: True
TPU available: False, using: 0 TPU cores
IPU available: False, using: 0 IPUs
HPU available: False, using: 0 HPUs
```

```
trainer.fit(model=vae_model, datamodule=time_vae_dm)
```

## Fitted Model

```
IS_RELOAD = True
```

```
if IS_RELOAD:
    checkpoint_path = "lightning_logs/time_vae_naive/version_29/checkpoints/epoch=1999-step=354000.ckpt"
    vae_model_reloaded = VAEModel.load_from_checkpoint(checkpoint_path, model=vae)
else:
    vae_model_reloaded = vae_model
```

```
for pred_batch in time_vae_dm.predict_dataloader():
    print(pred_batch.size())
    i_pred = vae_model_reloaded.model(pred_batch.float().cuda())
    break
```

```
i_pred[0].size()
```

```
import matplotlib.pyplot as plt
```

```
_, ax = plt.subplots()

element = 4

ax.plot(pred_batch.detach().numpy()[element, :, 0])
ax.plot(i_pred[0].cpu().detach().numpy()[element, :, 0], "x-")
```

Data generation using the decoder.

```
sampling_z = torch.randn(
    pred_batch.size(0), vae_model_reloaded.model.encoder.params.latent_size
).type_as(vae_model_reloaded.model.encoder.z_mean_layer.weight)
generated_samples_x = (
    vae_model_reloaded.model.decoder(sampling_z).cpu().detach().numpy().squeeze()
)
```

```
generated_samples_x.shape
```

```
_, ax = plt.subplots()

for i in range(min(len(generated_samples_x), 4)):
    ax.plot(generated_samples_x[i, :], "x-")
```

```
from openTSNE import TSNE
```

```
n_tsne_samples = 100
```

```
original_samples = pred_batch.cpu().detach().numpy().squeeze()[:n_tsne_samples]
original_samples.shape
```

```
tsne = TSNE(
    perplexity=30,
    metric="euclidean",
    n_jobs=8,
    random_state=42,
    verbose=True,
)
```

```
original_samples_embedding = tsne.fit(original_samples)
```

```
generated_samples_x[:n_tsne_samples]
```

```
generated_samples_embedding = original_samples_embedding.transform(
    generated_samples_x[:n_tsne_samples]
)
```

```
fig, ax = plt.subplots(figsize=(7, 7))

ax.scatter(
    original_samples_embedding[:, 0],
    original_samples_embedding[:, 1],
    color="black",
    marker=".",
    label="original",
)

ax.scatter(
    generated_samples_embedding[:, 0],
    generated_samples_embedding[:, 1],
    color="red",
    marker="x",
    label="generated",
)

ax.set_title("t-SNE of original and generated samples")
ax.set_xlabel("t-SNE 1")
ax.set_ylabel("t-SNE 2")
```

## Investigation of Embeddings

```
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.express as px
from torch.utils.data import DataLoader
from torchdr import PCA, TSNE, UMAP
from ts_dl_utils.datasets.dataset import DataFrameDataset
```

```
vae_model_reloaded
```

```
def embedding_extractor(forecaster: VAEModel, x: torch.Tensor) -> tuple[torch.Tensor]:
    """compute the embeddings based on the input

    :param forecaster: the trained forecaster
    :param x: input historical time series,
    """
    forecaster.model.to(x.device)
    z_mean, z_log_var, z = forecaster.model.encoder(
        x.type_as(forecaster.model.encoder.z_mean_layer.weight)
    )

    return z_mean, z_log_var, z
```

```
investigation_dl = DataLoader(
    dataset=DataFrameDataset(
        dataframe=df[["theta"]], history_length=window_size, horizon=1, gap=0
    ),
    batch_size=400,
    shuffle=False,
)

investigation_dl
```

```
input_example = list(investigation_dl)[0][0]
input_example.shape
```

```
z_mean_example, z_log_var_example, z_example = embedding_extractor(
    vae_model_reloaded, input_example
)
z_example.shape
```

```
(z_example.shape, z_mean_example.shape, z_log_var_example.shape)
```

```
n_components = 3

dr_input_result = TSNE(
    # dr_input_result = PCA(
    # n_neighbors=30, backend='torch',
    perplexity=30,
    n_components=n_components,
).fit_transform(input_example.detach().squeeze())

dr_input_result.shape
```

```
dr_z_mean_result = TSNE(
    # dr_z_result = PCA(
    # n_neighbors=30, backend='torch',
    # perplexity=30,
    n_components=n_components
).fit_transform(
    # z_example.detach().squeeze()
    z_mean_example.detach().squeeze()
)

dr_z_mean_result.shape
```

```
input_example.detach().shape
```

```
def create_embedding_dataframe(
    dr_result: torch.Tensor,
    n_batches: int,
    input_example: torch.Tensor,
    n_components: int,
) -> pd.DataFrame:

    dr_df = pd.DataFrame(
        dr_result.detach().numpy(), columns=[f"DR_{i+1}" for i in range(n_components)]
    )

    dr_df["batch"] = sum(
        [[i] * (len(dr_df) // n_batches) for i in range(n_batches)], []
    )

    dr_df["sample_idx"] = list(range(len(dr_df) // n_batches)) * n_batches

    # dr_df["input"] = np.concatenate(
    #     input_example[:n_batches].detach().numpy().astype("float32")
    # )

    dr_df = dr_df.merge(
        pd.DataFrame(
            input_example[:n_batches].detach()[:, 0, 0].numpy(),
            columns=["batch_first_value"],
        )
        .reset_index()
        .rename(columns={"index": "batch"}),
        how="left",
        on="batch",
    )

    return dr_df
```

```
dr_input_df = create_embedding_dataframe(
    dr_result=dr_input_result,
    n_batches=input_example.shape[0],
    input_example=input_example,
    n_components=n_components,
)
```

```
px.scatter(
    dr_input_df,
    x="DR_1",
    y="DR_2",
    # z='DR_3',
    # z='batch_first_value',
    # color='sample_idx',
    # symbol='type',
    # color="batch",
    # color="DR_3",
    color="batch_first_value",
    title="Embedding of Input Time Series",
    height=600,
    width=800,
).update_layout(legend=dict(itemsizing="constant", orientation="h", y=-0.2)).show()
```

```
dr_z_df = create_embedding_dataframe(
    dr_result=dr_z_mean_result,
    n_batches=z_example.shape[0],
    input_example=input_example,
    n_components=n_components,
)
```

```
dr_z_df
```

```
px.scatter(
    dr_z_df,
    x="DR_1",
    # y="batch_first_value",
    y="DR_2",
    # z='DR_3',
    # z='batch_first_value',
    # color='sample_idx',
    # symbol='type',
    # color="batch",
    # color="DR_2",
    color="batch_first_value",
    title="UMAP Embedding of Encoder Encoded Time Series",
    height=600,
    width=800,
).update_layout(legend=dict(itemsizing="constant", orientation="h", y=-0.2)).show()
```

```
px.scatter(
    dr_z_df,
    x="DR_1",
    y="DR_2",
    # z='DR_3',
    # z='batch_first_value',
    # color='sample_idx',
    # symbol='type',
    # color="batch",
    # color="DR_3",
    color="batch_first_value",
    title="UMAP Embedding of Encoder Encoded Time Series",
    height=600,
    width=800,
).update_layout(legend=dict(itemsizing="constant", orientation="h", y=-0.2)).show()
```

```
px.scatter_3d(
    dr_z_df,
    x="DR_1",
    y="DR_2",
    z="DR_3",
    # z='batch_first_value',
    # color='sample_idx',
    # symbol='type',
    # color="batch",
    color="batch_first_value",
    title="UMAP Embedding of Encoder Encoded Time Series",
    height=600,
    width=800,
).update_layout(legend=dict(itemsizing="constant", orientation="h", y=-0.2)).show()
```

```
px.imshow(input_example.squeeze().T, aspect="auto")
```

```
px.imshow(dr_input_result.T.detach().numpy(), aspect="auto")
```

```
px.imshow(dr_z_mean_result.T.detach().numpy(), aspect="auto")
```

```
px.imshow(dr_z_df[["DR_1", "DR_2", "DR_3"]].T, aspect="auto")
```
