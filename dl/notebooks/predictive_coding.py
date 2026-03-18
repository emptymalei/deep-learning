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

# # CPC for Time Series Data
#
# References:
#
# - [mqwfrog/ULTS/tree/main/models/CPC](https://github.com/mqwfrog/ULTS/tree/main/models/CPC)
# - [ECG200](https://www.timeseriesclassification.com/description.php?Dataset=ECG200)
# - R. T. Olszewski, “Generalized feature extraction for structural pattern recognition in time-series data,” Carnegie Mellon University, Pittsburgh, PA, 2001. Available: https://www.cs.cmu.edu/~bobski/pubs/tr01108-twosided.pdf
#
#

# configs

from pathlib import Path
from typing import Literal

import yaml
from pydantic import BaseModel, Field

# +


class DataConfig(BaseModel):
    """Dataset location and dataset identity settings."""

    dataset_name: str = "ECG200"
    data_dir: str = "data"


class DataloaderConfig(BaseModel):
    """PyTorch DataLoader settings for train and validation loaders."""

    batch_size: int = 64
    drop_last: bool = False
    num_workers: int = 0


class DataModuleConfig(BaseModel):
    """Top-level configuration object for CPC training."""

    data: DataConfig = Field(default_factory=DataConfig)
    dataloader: DataloaderConfig = Field(default_factory=DataloaderConfig)


# -

# Data

# +
import os

import lightning as L
import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset


class CPCDataset(Dataset):
    """Simple dataset wrapper for CPC tensors.

    Expected input object format:
    - ``samples``: array/tensor with shape ``[N, T]`` or ``[N, C, T]`` or
      ``[N, T, C]``.
    - ``labels``: array/tensor with shape ``[N]``.
    """

    def __init__(self, dataset_obj):
        """Initialize dataset tensors.

        :param dataset_obj: Dictionary containing ``samples`` and ``labels``
            arrays/tensors. Samples are converted to ``float32`` and labels
            to ``int64``.
        """
        super().__init__()

        samples = dataset_obj["samples"]
        labels = dataset_obj["labels"]

        if isinstance(samples, np.ndarray):
            samples = torch.from_numpy(samples)
        if isinstance(labels, np.ndarray):
            labels = torch.from_numpy(labels)

        if samples.ndim == 2:
            samples = samples.unsqueeze(1)
        elif samples.ndim == 3:
            # Keep channels in dim=1.
            if list(samples.shape).index(min(samples.shape)) != 1:
                samples = samples.permute(0, 2, 1)

        self.samples = samples.float()
        self.labels = labels.long()

    def __len__(self):
        """Return number of samples ``N``."""
        return self.samples.shape[0]

    def __getitem__(self, index):
        """Return one sample-label pair.

        :param index: Sample index.
        :returns: Tuple ``(x, y)`` where ``x`` has shape ``[C, T]`` and ``y``
            is a scalar class label tensor.
        """
        return self.samples[index], self.labels[index]


class CPCDataModule(L.LightningDataModule):
    """Lightning data module for CPC train/validation loaders."""

    def __init__(self, config: DataModuleConfig):
        """Initialize data module.

        :param config: datamodule configuration containing data and dataloader
            settings.
        """
        super().__init__()
        self.config = config
        self.train_loader = None
        self.val_loader = None
        self.input_channels = None

    def setup(self, stage=None):
        """Load serialized datasets and build data loaders.

        :param stage: Lightning setup stage. Present for Lightning
            compatibility.
        """
        if self.train_loader is not None:
            return

        dataset_path = os.path.join(
            self.config.data.data_dir, self.config.data.dataset_name
        )
        train_obj = torch.load(os.path.join(dataset_path, "train.pt"))
        test_obj = torch.load(os.path.join(dataset_path, "test.pt"))

        train_ds = CPCDataset(train_obj)
        test_ds = CPCDataset(test_obj)

        self.input_channels = int(train_ds.samples.shape[1])
        self.train_loader = DataLoader(
            train_ds,
            batch_size=self.config.dataloader.batch_size,
            shuffle=True,
            drop_last=self.config.dataloader.drop_last,
            num_workers=self.config.dataloader.num_workers,
        )
        self.val_loader = DataLoader(
            test_ds,
            batch_size=self.config.dataloader.batch_size,
            shuffle=False,
            drop_last=False,
            num_workers=self.config.dataloader.num_workers,
        )

    def train_dataloader(self):
        """Return training data loader.

        :returns: Loader yielding batches ``(x, y)`` with ``x`` shape
            ``[B, C_in, T]``.
        """
        return self.train_loader

    def val_dataloader(self):
        """Return validation data loader.

        :returns: Loader yielding batches ``(x, y)`` with ``x`` shape
            ``[B, C_in, T]``.
        """
        return self.val_loader


# -

# Modules


# +
class ConvLayerConfig(BaseModel):
    """Single 1D convolution specification.

    Parameters define one convolution block as
    ``Conv1d(in_channels, out_channels, kernel_size, stride, padding)``.
    """

    kernel_size: int
    stride: int
    padding: int


class EncoderConfig(BaseModel):
    """Encoder architecture and normalization settings."""

    hidden_size: int = 256
    norm_mode: Literal["batchNorm", "instanceNorm", "ID", "layerNorm"] = "layerNorm"
    conv_layers: list[ConvLayerConfig] = Field(
        default_factory=lambda: [
            ConvLayerConfig(kernel_size=10, stride=5, padding=3),
            ConvLayerConfig(kernel_size=8, stride=4, padding=2),
            ConvLayerConfig(kernel_size=4, stride=2, padding=1),
            ConvLayerConfig(kernel_size=4, stride=2, padding=1),
            ConvLayerConfig(kernel_size=4, stride=2, padding=1),
        ]
    )


class ARConfig(BaseModel):
    """Autoregressive model selection and parameters.

    Supports recurrent modes (GRU/LSTM/RNN), Transformer mode, and
    identity mode (``no_ar``).
    """

    mode: Literal["GRU", "LSTM", "RNN", "transformer", "no_ar"] = "LSTM"
    hidden_size: int = 256
    n_levels: int = 1
    reverse: bool = False


class LossConfig(BaseModel):
    """CPC/InfoNCE objective configuration."""

    n_predictions: int = 12
    negative_sampling_ext: int = 128
    temperature: float = 0.07


import math

# +
from functools import cached_property
from typing import Callable, Literal

import torch
import torch.nn as nn
import torch.nn.functional as F


class CPCEncoder(nn.Module):
    """Convolutional encoder for time-series CPC.

    Input is expected as ``[B, C_in, T]`` and output as ``[B, H, T_enc]``,
    where ``H`` is ``size_hidden`` and ``T_enc`` depends on convolution specs.
    """

    def __init__(
        self,
        input_channels: int = 1,
        size_hidden: int = 256,
        conv_specs: list[tuple[int, int, int]] | None = None,
    ):
        """Build the encoder stack.

        :param input_channels: Number of input channels ``C_in``.
        :param size_hidden: Number of output channels for each convolution
            block.
        :param conv_specs: List of convolution settings as
            ``(kernel_size, stride, padding)``. If ``None``, a default
            5-layer downsampling stack is used.
        """
        super().__init__()

        norm_factory: Callable[[int], nn.Module]
        norm_factory = nn.BatchNorm1d

        if conv_specs is None:
            conv_specs = [
                (10, 5, 3),
                (8, 4, 2),
                (4, 2, 1),
                (4, 2, 1),
                (4, 2, 1),
            ]
        blocks: list[nn.Module] = []
        in_channels = input_channels
        for kernel_size, stride, padding in conv_specs:
            blocks.extend(
                [
                    nn.Conv1d(
                        in_channels,
                        size_hidden,
                        kernel_size,
                        stride=stride,
                        padding=padding,
                    ),
                    norm_factory(size_hidden),
                    nn.ReLU(),
                ]
            )
            in_channels = size_hidden

        self.network = nn.Sequential(*blocks)
        self.output_dim = size_hidden

    @cached_property
    def output_dimension(self) -> int:
        """Return the encoder feature dimension ``H``."""
        return self.output_dim

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Encode raw sequence features.

        :param x: Input tensor with shape ``[B, C_in, T]``.
        :returns: Encoded tensor with shape ``[B, H, T_enc]``.
        """
        return self.network(x)


class CPCAR(nn.Module):
    """Recurrent autoregressive context model for CPC.

    Operates on encoded sequences in ``[B, T, C]`` format and returns
    contextualized representations in ``[B, T, C_ctx]``.
    """

    def __init__(
        self,
        dim_encoded: int,
        dim_output: int,
        n_levels_gru: int = 1,
        rnn_type: Literal["GRU", "LSTM", "RNN"] = "LSTM",
        # reverse: bool = False,
    ):
        """Initialize recurrent AR module.

        :param dim_encoded: Input feature size ``C`` from the encoder.
        :param dim_output: Hidden/context size ``C_ctx`` of the recurrent
            module.
        :param n_levels_gru: Number of recurrent layers.
        :param rnn_type: Recurrent cell type.
        # :param reverse: If ``True``, run AR in reverse time and flip output
            back.
        """
        super().__init__()
        # self.reverse = reverse

        rnn_cls = {"LSTM": nn.LSTM, "RNN": nn.RNN, "GRU": nn.GRU}[rnn_type]
        self.rnn = rnn_cls(
            dim_encoded, dim_output, num_layers=n_levels_gru, batch_first=True
        )

    @cached_property
    def output_dimension(self) -> int:
        """Return context feature dimension ``C_ctx``."""
        return self.rnn.hidden_size

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Run recurrent AR on encoded sequence.

        :param x: Encoded tensor with shape ``[B, T, C]``.
        :returns: Context tensor with shape ``[B, T, C_ctx]``.
        """
        # if self.reverse:
        #     x = torch.flip(x, [1])
        # try:
        #     self.rnn.flatten_parameters()
        # except RuntimeError:
        #     pass
        x, _ = self.rnn(x)
        # if self.reverse:
        #     x = torch.flip(x, [1])
        return x


class CPCModel(nn.Module):
    """Compose encoder and autoregressive context model."""

    def __init__(self, encoder: nn.Module, ar: nn.Module):
        """Initialize CPC model.

        :param encoder: Module mapping ``[B, C_in, T]`` to
            ``[B, C_enc, T_enc]``.
        :param ar: Module mapping ``[B, T_enc, C_enc]`` to context features
            ``[B, T_enc, C_ctx]``.
        """
        super().__init__()
        self.encoder = encoder
        self.ar = ar

    def forward(self, batch_data: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Run encoder then AR network.

        :param batch_data: Input batch with shape ``[B, C_in, T]``.
        :returns: Tuple ``(context, encoded)`` where ``context`` has shape
            ``[B, T_enc, C_ctx]`` and ``encoded`` has shape
            ``[B, T_enc, C_enc]``.
        """
        encoded = self.encoder(batch_data).permute(0, 2, 1)
        context = self.ar(encoded)
        return context, encoded


class CPCCriterion(nn.Module):
    """InfoNCE loss used by Contrastive Predictive Coding.

    For each prediction horizon ``k``, the criterion predicts future encoded
    vectors from context, then contrasts one positive against sampled negatives.
    """

    def __init__(
        self,
        dim_context: int,
        dim_encoded: int,
        n_predictions: int = 12,
        n_negatives: int = 128,
        temperature: float = 0.07,
    ):
        """Initialize CPC criterion.

        :param dim_context: Context feature dimension ``C_ctx``.
        :param dim_encoded: Encoded target feature dimension ``C_enc``.
        :param n_predictions: Number of future steps to predict.
        :param n_negatives: Number of negative samples per query.
        :param temperature: Logit temperature scaling factor.
        """
        super().__init__()
        self.n_predictions = n_predictions
        self.n_negatives = n_negatives
        self.temperature = temperature
        self.predictors = nn.ModuleList(
            [
                nn.Linear(dim_context, dim_encoded, bias=False)
                for _ in range(n_predictions)
            ]
        )

    def _sample_negatives(
        self, encoded_pool: torch.Tensor, n_queries: int
    ) -> torch.Tensor:
        """Sample negative encoded vectors.

        :param encoded_pool: Flattened encoded with shape ``[B*T, C_enc]``.
        :param n_queries: Number of query vectors to match.
        :returns: Negative samples with shape
            ``[n_queries, n_negatives, C_enc]``.
        """
        idx = torch.randint(
            low=0,
            high=encoded_pool.size(0),
            size=(n_queries, self.n_negatives),
            device=encoded_pool.device,
        )
        return encoded_pool[idx]

    def forward(self, context: torch.Tensor, encoded: torch.Tensor) -> torch.Tensor:
        """Compute averaged InfoNCE loss across valid prediction horizons.

        :param context: Context tensor with shape ``[B, T, C_ctx]``.
        :param encoded: Encoded tensor with shape ``[B, T, C_enc]``.
        :returns: Scalar loss tensor.
        :raises RuntimeError: If no prediction horizon is valid, usually
            because ``T`` is too short relative to ``n_predictions``.
        """
        _, seq_len, _ = encoded.shape
        encoded_pool = F.normalize(encoded.flatten(0, 1), dim=-1)

        total_loss: torch.Tensor | None = None
        used_steps = 0

        for k in range(1, self.n_predictions + 1):
            if seq_len - k <= 0:
                break

            c_t = context[:, : seq_len - k, :]
            z_pos = encoded[:, k:, :]

            z_hat = self.predictors[k - 1](c_t)
            z_hat = F.normalize(z_hat.flatten(0, 1), dim=-1)
            z_pos = F.normalize(z_pos.flatten(0, 1), dim=-1)

            n_queries = z_hat.size(0)
            pos_logits = torch.sum(z_hat * z_pos, dim=-1, keepdim=True)

            z_neg = self._sample_negatives(encoded_pool, n_queries)
            neg_logits = torch.einsum("nc,nkc->nk", z_hat, z_neg)

            logits = torch.cat([pos_logits, neg_logits], dim=1) / self.temperature
            targets = torch.zeros(n_queries, dtype=torch.long, device=logits.device)

            step_loss = F.cross_entropy(logits, targets)
            total_loss = step_loss if total_loss is None else total_loss + step_loss
            used_steps += 1

        if used_steps == 0:
            raise RuntimeError(
                "CPCCriterion has no valid prediction steps: encoded sequence length is too short "
                f"(seq_len={seq_len}) for n_predictions={self.n_predictions}. "
                "Use fewer downsampling conv layers, smaller strides, or lower n_predictions."
            )

        if total_loss is None:
            raise RuntimeError(
                "CPCCriterion failed to compute loss: total_loss is None after processing all prediction steps. "
            )

        return total_loss / used_steps


# -

# Model

# +


class OptimizerConfig(BaseModel):
    """Optimizer hyperparameters."""

    learning_rate: float = 2e-4
    weight_decay: float = 1e-6


class TrainerConfig(BaseModel):
    """Trainer runtime options, including epoch count and output path."""

    epochs: int = 100
    output_dir: str = "lightning_logs/ts_cpc"


class RuntimeConfig(BaseModel):
    """Execution environment settings such as seed and device."""

    seed: int = 0
    accelerator: Literal["auto", "cpu", "gpu", "mps"] = "auto"
    devices: int = 1


class TrainConfig(BaseModel):
    """Top-level configuration object for CPC training."""

    data: DataConfig = Field(default_factory=DataConfig)
    dataloader: DataloaderConfig = Field(default_factory=DataloaderConfig)
    encoder: EncoderConfig = Field(default_factory=EncoderConfig)
    ar: ARConfig = Field(default_factory=ARConfig)
    loss: LossConfig = Field(default_factory=LossConfig)
    optimizer: OptimizerConfig = Field(default_factory=OptimizerConfig)
    trainer: TrainerConfig = Field(default_factory=TrainerConfig)
    runtime: RuntimeConfig = Field(default_factory=RuntimeConfig)

    @classmethod
    def from_yaml(cls, path: Path) -> "TrainConfig":
        """Load a training configuration from YAML.

        :param path: Path to a YAML file containing any subset of
            ``TrainConfig`` fields.
        :returns: Parsed configuration with defaults applied to omitted
            fields.
        """
        with open(path) as f:
            data = yaml.safe_load(f)
        return cls(**(data or {}))


# +
import os

import lightning as L
import torch
from lightning.pytorch.callbacks import ModelCheckpoint


class CPCLightningModule(L.LightningModule):
    """Lightning wrapper around CPC model and criterion.

    This module handles model construction, train/validation steps, and
    optimizer configuration.
    """

    def __init__(self, config: TrainConfig, input_channels: int):
        """Initialize the Lightning module.

        :param config: Hierarchical training configuration.
        :param input_channels: Number of input channels in each sample ``x``
            where ``x`` has shape ``[B, C_in, T]``.
        """
        super().__init__()
        self.config = config
        self.save_hyperparameters(config.model_dump())

        encoder = CPCEncoder(
            input_channels=input_channels,
            size_hidden=config.encoder.hidden_size,
            # norm_mode=config.encoder.norm_mode,
            conv_specs=[
                (l.kernel_size, l.stride, l.padding) for l in config.encoder.conv_layers
            ],
        )

        ar = CPCAR(
            dim_encoded=config.encoder.hidden_size,
            dim_output=config.ar.hidden_size,
            n_levels_gru=config.ar.n_levels,
            rnn_type=config.ar.mode,
            # reverse=config.ar.reverse,
        )

        self.model = CPCModel(encoder, ar)
        self.criterion = CPCCriterion(
            dim_context=ar.output_dimension,
            dim_encoded=encoder.output_dimension,
            n_predictions=config.loss.n_predictions,
            n_negatives=config.loss.negative_sampling_ext,
            temperature=config.loss.temperature,
        )

    def _shared_step(self, batch, split: str):
        """Run one shared forward/loss step for train or validation.

        :param batch: Tuple ``(x, y)`` where ``x`` has shape ``[B, C_in, T]``
            and ``y`` is the label tensor.
        :param split: Metric prefix, typically ``"train"`` or ``"val"``.
        :returns: Scalar CPC loss.
        """
        batch_x, _ = batch
        batch_x = batch_x.float()
        context, encoded = self.model(batch_x)
        loss = self.criterion(context, encoded)
        self.log(
            f"{split}_loss",
            loss,
            on_step=False,
            on_epoch=True,
            prog_bar=True,
            batch_size=batch_x.size(0),
        )
        return loss

    def training_step(self, batch, batch_idx):
        """Run one training step.

        :param batch: Mini-batch tuple ``(x, y)``.
        :param batch_idx: Batch index within the epoch.
        :returns: Scalar training loss.
        """
        return self._shared_step(batch, "train")

    def validation_step(self, batch, batch_idx):
        """Run one validation step.

        :param batch: Mini-batch tuple ``(x, y)``.
        :param batch_idx: Batch index within the validation epoch.
        """
        self._shared_step(batch, "val")

    def predict_step(
        self, data, batch_idx: int = 0
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Run one prediction step.

        :param data: Mini-batch tuple ``(x, y)``.
        :param batch_idx: Batch index within the prediction epoch.
        :returns: Tuple of context and encoded tensors.
        """
        x = data.float()
        context, encoded = self.model(x)
        return context, encoded

    def configure_optimizers(self):
        """Create optimizer for model and criterion parameters.

        :returns: Adam optimizer configured from ``config.optimizer``.
        """
        return torch.optim.Adam(
            list(self.model.parameters()) + list(self.criterion.parameters()),
            lr=self.config.optimizer.learning_rate,
            weight_decay=self.config.optimizer.weight_decay,
        )


# +
config = TrainConfig.from_yaml(Path("configs/predictive_coding/config.ecg200.yaml"))

L.seed_everything(config.runtime.seed, workers=True)
os.makedirs(config.trainer.output_dir, exist_ok=True)

data_module = CPCDataModule(
    DataModuleConfig(
        data=config.data,
        dataloader=config.dataloader,
    )
)
data_module.setup(stage="fit")
lightning_module = CPCLightningModule(
    config=config, input_channels=data_module.input_channels
)

trainer = L.Trainer(
    max_epochs=config.trainer.epochs,
    accelerator=config.runtime.accelerator,
    devices=config.runtime.devices,
    default_root_dir=config.trainer.output_dir,
    log_every_n_steps=5,
)
trainer.fit(lightning_module, datamodule=data_module)

# -


# ## Load Artifacts and Interpret

for i in data_module.train_dataloader():
    pred_data, pred_label = i
    break

pred_data.shape, pred_label.shape

pred_label.numpy()

pred_context, pred_encoded = trainer.model.predict_step(pred_data)

pred_context.shape, pred_encoded.shape
