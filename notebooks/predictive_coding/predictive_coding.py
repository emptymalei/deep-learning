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
    val_split: float = 0.1


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
from torch.utils.data import DataLoader, Dataset, Subset


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


def _stratified_split(
    labels: np.ndarray, val_fraction: float, random_state: int = 42
) -> tuple[np.ndarray, np.ndarray]:
    """Stratified train/val index split without sklearn."""
    rng = np.random.default_rng(random_state)
    train_idx, val_idx = [], []
    for cls in np.unique(labels):
        cls_indices = np.where(labels == cls)[0]
        rng.shuffle(cls_indices)
        n_val = max(1, round(len(cls_indices) * val_fraction))
        val_idx.extend(cls_indices[:n_val].tolist())
        train_idx.extend(cls_indices[n_val:].tolist())
    return np.array(train_idx), np.array(val_idx)


class CPCDataModule(L.LightningDataModule):
    def __init__(self, config: DataModuleConfig):
        super().__init__()
        self.config = config
        self.train_loader = None
        self.val_loader = None
        self.test_loader = None
        self.input_channels = None

    def setup(self, stage=None):
        if self.train_loader is not None:
            return

        dataset_path = os.path.join(
            self.config.data.data_dir, self.config.data.dataset_name
        )
        train_obj = torch.load(os.path.join(dataset_path, "train.pt"))
        test_obj = torch.load(os.path.join(dataset_path, "test.pt"))

        full_train_ds = CPCDataset(train_obj)
        test_ds = CPCDataset(test_obj)

        self.input_channels = int(full_train_ds.samples.shape[1])

        labels = full_train_ds.labels.numpy()
        train_idx, val_idx = _stratified_split(
            labels,
            val_fraction=self.config.dataloader.val_split,
            random_state=42,
        )
        train_ds = Subset(full_train_ds, train_idx)
        val_ds = Subset(full_train_ds, val_idx)

        dl_kwargs = dict(
            batch_size=self.config.dataloader.batch_size,
            num_workers=self.config.dataloader.num_workers,
        )
        self.train_loader = DataLoader(
            train_ds,
            shuffle=True,
            drop_last=self.config.dataloader.drop_last,
            **dl_kwargs,
        )
        self.val_loader = DataLoader(
            val_ds, shuffle=False, drop_last=False, **dl_kwargs
        )
        self.test_loader = DataLoader(
            test_ds, shuffle=False, drop_last=False, **dl_kwargs
        )

    def train_dataloader(self):
        return self.train_loader

    def val_dataloader(self):
        return self.val_loader

    def test_dataloader(self):
        return self.test_loader


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
        # norm_factory = nn.Identity

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
        reverse: bool = False,
    ):
        """Initialize recurrent AR module.

        :param dim_encoded: Input feature size ``C`` from the encoder.
        :param dim_output: Hidden/context size ``C_ctx`` of the recurrent
            module.
        :param n_levels_gru: Number of recurrent layers.
        :param rnn_type: Recurrent cell type.
        :param reverse: If ``True``, run AR in reverse time and flip output
            back.
        """
        super().__init__()
        self.reverse = reverse

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
        if self.reverse:
            x = torch.flip(x, [1])
        try:
            self.rnn.flatten_parameters()
        except RuntimeError:
            pass
        x, _ = self.rnn(x)
        if self.reverse:
            x = torch.flip(x, [1])
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
            reverse=config.ar.reverse,
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
# config = TrainConfig.from_yaml(
#     Path("configs/predictive_coding/config.ecg200.yaml")
# )
# config = TrainConfig.from_yaml(
# Path("configs/predictive_coding/config.ecg5000.yaml")
# )
# config = TrainConfig.from_yaml(
# Path("configs/predictive_coding/config.forda.yaml")
# )
# config = TrainConfig.from_yaml(Path("configs/predictive_coding/config.sleep.yaml"))

config = TrainConfig.from_yaml(
    Path("configs/predictive_coding/config.binaryheartbeat.yaml")
)
# -

config.model_dump()

# +


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

import numpy as np
import pandas as pd
import plotly.express as px
import torch
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

# ### Visualize One Batch

for i in data_module.train_dataloader():
    pred_data, pred_label = i
    break

pred_data.shape, pred_label.shape

# +
data_pca2 = PCA(
    n_components=2,
).fit_transform(pred_data.detach().numpy()[:, 0, :])

df_data_pca = pd.DataFrame(data_pca2, columns=["pc_1", "pc_2"])

df_data_pca["label"] = pred_label.numpy().astype(str)

df_data_pca.head()
# -

px.scatter(
    df_data_pca,
    x="pc_1",
    y="pc_2",
    color="label",
    title="PCA of CPC Input Data",
    width=800,
    height=600,
    color_discrete_sequence=px.colors.qualitative.Set2,
)

pred_label.numpy()

pred_context, pred_encoded = trainer.model.predict_step(pred_data)

pred_context.shape, pred_encoded.shape

from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.manifold import TSNE
from sklearn.metrics import silhouette_score
from sklearn.model_selection import StratifiedKFold, cross_val_score

# +
context_pca2 = PCA(
    n_components=2,
).fit_transform(pred_context.detach().numpy()[:, 0, :])

df_context = pd.DataFrame(context_pca2, columns=["pc_1", "pc_2"])

df_context["label"] = pred_label.numpy().astype(str)

df_context.head()
# -

px.scatter(
    df_context,
    x="pc_1",
    y="pc_2",
    color="label",
    title="PCA of CPC Context Representations",
    width=800,
    height=600,
    color_discrete_sequence=px.colors.qualitative.Set2,
)

# +
context_tsne2d = TSNE(
    n_components=2,
    perplexity=15,
    random_state=0,
    # init="pca"
).fit_transform(
    # PCA(
    #     n_components=10,
    # ).fit_transform(
    #     pred_context.detach().numpy()[:,0,:]
    # )
    pred_context.detach().numpy()[:, 0, :]
)

df_context_tsne = pd.DataFrame(context_tsne2d, columns=["tsne_1", "tsne_2"])

df_context_tsne["label"] = pred_label.numpy().astype(str)

df_context_tsne.head()
# -

px.scatter(
    df_context_tsne,
    x="tsne_1",
    y="tsne_2",
    color="label",
    title="t-SNE of CPC Context Representations",
    width=800,
    height=600,
    color_discrete_sequence=px.colors.qualitative.Set2,
)

# +
encoded_pca2 = PCA(
    n_components=2,
).fit_transform(pred_encoded.detach().numpy()[:, 0, :])

df_encoded_pca = pd.DataFrame(encoded_pca2, columns=["pc_1", "pc_2"])

df_encoded_pca["label"] = pred_label.numpy().astype(str)

df_encoded_pca.head()
# -

px.scatter(
    df_encoded_pca,
    x="pc_1",
    y="pc_2",
    color="label",
    title="PCA of CPC Encoded Representations",
    width=800,
    height=600,
    color_discrete_sequence=px.colors.qualitative.Set2,
)

# +
encoded_tsne2d = TSNE(
    n_components=2,
    perplexity=15,
    random_state=0,
    # init="pca"
).fit_transform(
    # PCA(
    #     n_components=10,
    # ).fit_transform(
    #     pred_encoded.detach().numpy()[:,0,:]
    # )
    pred_encoded.detach().numpy()[:, 0, :]
)

df_encoded_tsne = pd.DataFrame(encoded_tsne2d, columns=["tsne_1", "tsne_2"])

df_encoded_tsne["label"] = pred_label.numpy().astype(str)

df_encoded_tsne.head()
# -

px.scatter(
    df_encoded_tsne,
    x="tsne_1",
    y="tsne_2",
    color="label",
    title="t-SNE of CPC Encoded Representations",
    width=800,
    height=600,
    color_discrete_sequence=px.colors.qualitative.Set2,
)

# ### Visualizez All

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from sklearn.decomposition import PCA

# +
# Faceted comparison: raw vs context vs encoded for the same reduction method
# Produces two figures:
# 1) PCA with facet rows (raw/context/encoded)
# 2) t-SNE with facet rows (raw/context/encoded)


# Rebuild tensors if they are not already in memory
if (
    "raw_all" not in globals()
    or "context_all" not in globals()
    or "encoded_all" not in globals()
):
    module = trainer.model if hasattr(trainer, "model") else lightning_module
    module.eval()
    device = module.device

    all_raw, all_context, all_encoded, all_labels = [], [], [], []
    with torch.no_grad():
        for x, y_batch in data_module.train_dataloader():
            x = x.float().to(device)
            context, encoded = module.model(x)

            all_raw.append(x.cpu())
            all_context.append(context.cpu())
            all_encoded.append(encoded.cpu())
            all_labels.append(y_batch.cpu())

    raw_all = torch.cat(all_raw, dim=0)  # [N, C_in, T]
    context_all = torch.cat(all_context, dim=0)  # [N, T_enc, C]
    encoded_all = torch.cat(all_encoded, dim=0)  # [N, T_enc, C]
    y = torch.cat(all_labels, dim=0).numpy()
    y_str = y.astype(str)

# Choose one view per family for fair visual comparison
Z_by_family = {
    "raw": raw_all.reshape(raw_all.shape[0], -1).numpy(),  # flatten_all_channels
    "context": context_all.reshape(context_all.shape[0], -1).numpy(),  # flatten_tokens
    "encoded": encoded_all.reshape(encoded_all.shape[0], -1).numpy(),  # flatten_tokens
    # "encoded": (
    #     encoded_all[:, 0, :]
    #     .reshape(encoded_all.shape[0], -1)
    #     .numpy(),  # flatten_tokens
    # )
}

family_order = ["raw", "context", "encoded"]


def reduce_family(Z, method="pca", random_state=0):
    if method == "pca":
        return make_pipeline(
            StandardScaler(),
            PCA(n_components=2, random_state=random_state),
        ).fit_transform(Z)
    if method == "tsne":
        n = Z.shape[0]
        perp = max(5, min(30, (n - 1) // 3))
        return TSNE(
            n_components=2,
            perplexity=perp,
            random_state=random_state,
            init="pca",
        ).fit_transform(Z)
    raise ValueError(f"Unknown method: {method}")


def build_facet_df(method):
    rows = []
    for fam in family_order:
        Z2 = reduce_family(Z_by_family[fam], method=method, random_state=0)
        df_f = pd.DataFrame(
            {
                "dim1": Z2[:, 0],
                "dim2": Z2[:, 1],
                "label": y_str,
                "family": fam,
            }
        )
        rows.append(df_f)
    return pd.concat(rows, ignore_index=True)


def plot_facet(dataframe, method, width=700, height=1100):
    df_plot = dataframe.copy()
    title = f"{method.upper()} comparison by family (facet rows)"
    fig = px.scatter(
        df_plot,
        x="dim1",
        y="dim2",
        color="label",
        facet_row="family",
        category_orders={"family": family_order},
        title=title,
        width=width,
        height=height,
        # color_discrete_sequence=px.colors.qualitative.Set2,
        opacity=0.8,
    )
    fig.update_traces(marker=dict(size=5))
    fig.for_each_annotation(lambda a: a.update(text=a.text.split("=")[-1]))
    fig.update_layout(legend_title_text="label")
    fig.update_xaxes(matches=None)
    fig.update_yaxes(matches=None)
    fig.show()


# Show both methods with the same faceting layout
df_plot_pca = build_facet_df("pca")
plot_facet(df_plot_pca, "pca")
df_plot_tsne = build_facet_df("tsne")
plot_facet(df_plot_tsne, "tsne")
# -
df_dr = pd.merge(
    df_plot_pca[["dim1", "dim2"]],
    df_plot_tsne,
    how="left",
    left_index=True,
    right_index=True,
    suffixes=("_pca", "_tsne"),
)
df_dr

# +
import numpy as np
import pandas as pd

# Expect df_dr columns from your merge:
# dim1_pca, dim2_pca, dim1_tsne, dim2_tsne, label, family

required = {"dim1_pca", "dim2_pca", "label", "family"}
missing = required - set(df_dr.columns)
if missing:
    raise ValueError(f"df_dr is missing required columns: {missing}")

# Keep only families needed for this vector
df_tmp = df_dr[df_dr["family"].isin(["raw", "encoded"])].copy()

# Create per-sample index inside each family block so raw and encoded can be paired
df_tmp["idx"] = df_tmp.groupby("family").cumcount()

# Split and rename PCA columns
raw_df = (
    df_tmp[df_tmp["family"] == "raw"][["idx", "label", "dim1_pca", "dim2_pca"]]
    .rename(columns={"dim1_pca": "raw_x", "dim2_pca": "raw_y"})
    .reset_index(drop=True)
)

enc_df = (
    df_tmp[df_tmp["family"] == "encoded"][["idx", "label", "dim1_pca", "dim2_pca"]]
    .rename(columns={"dim1_pca": "enc_x", "dim2_pca": "enc_y", "label": "label_enc"})
    .reset_index(drop=True)
)

# Pair raw and encoded rows by idx (and optionally validate labels)
df_vec = raw_df.merge(enc_df, on="idx", how="inner")

# Optional sanity check: label consistency
if not (df_vec["label"] == df_vec["label_enc"]).all():
    print("Warning: label mismatch found between raw and encoded pairing.")
df_vec = df_vec.drop(columns=["label_enc"])

# Vector components and norm
df_vec["vec_x"] = df_vec["enc_x"] - df_vec["raw_x"]
df_vec["vec_y"] = df_vec["enc_y"] - df_vec["raw_y"]
df_vec["vec_norm"] = np.linalg.norm(df_vec[["vec_x", "vec_y"]].to_numpy(), axis=1)

df_vec.head()


# +
labels = sorted(df_vec["label"].unique())
label_symbols = ["circle", "square", "triangle-up", "star", "hexagram"]

raw_color = "rgba(31, 119, 180, 0.85)"  # blue
enc_color = "rgba(214, 39, 40, 0.85)"  # red

line_x = np.column_stack(
    [df_vec["raw_x"], df_vec["enc_x"], np.full(len(df_vec), np.nan)]
).ravel()
line_y = np.column_stack(
    [df_vec["raw_y"], df_vec["enc_y"], np.full(len(df_vec), np.nan)]
).ravel()

fig = go.Figure()

fig.add_trace(
    go.Scatter(
        x=line_x,
        y=line_y,
        mode="lines",
        line=dict(color="rgba(80,80,80,0.20)", width=1),
        hoverinfo="skip",
        showlegend=False,
    )
)

for i, lbl in enumerate(labels):
    sym = label_symbols[i % len(label_symbols)]
    mask = df_vec["label"] == lbl
    subset = df_vec[mask]

    # raw — blue, label symbol
    fig.add_trace(
        go.Scatter(
            x=subset["raw_x"],
            y=subset["raw_y"],
            mode="markers",
            marker=dict(
                size=9, symbol=sym, color=raw_color, line=dict(width=1, color="white")
            ),
            name=f"raw (label={lbl})",
            legendgroup=f"label_{lbl}",
            text=[f"idx={r}, label={lbl}" for r in subset["idx"]],
            hovertemplate="RAW<br>%{text}<br>x=%{x:.3f}, y=%{y:.3f}<extra></extra>",
        )
    )

    # encoded — red, same label symbol
    fig.add_trace(
        go.Scatter(
            x=subset["enc_x"],
            y=subset["enc_y"],
            mode="markers",
            marker=dict(
                size=9, symbol=sym, color=enc_color, line=dict(width=1, color="white")
            ),
            name=f"encoded (label={lbl})",
            legendgroup=f"label_{lbl}",
            text=[
                f"idx={r}, label={lbl}, norm={n:.3f}"
                for r, n in zip(subset["idx"], subset["vec_norm"])
            ],
            hovertemplate="ENCODED<br>%{text}<br>x=%{x:.3f}, y=%{y:.3f}<extra></extra>",
        )
    )

fig.update_layout(
    title=f"Per-index vectors — color: raw(blue)/encoded(red), symbol: label (N={len(df_vec)})",
    xaxis_title="PC1 (raw space) / PC1 (encoded space)",
    yaxis_title="PC2",
    width=980,
    height=720,
    legend=dict(groupclick="toggleitem"),
)
fig.show()

# +
# Pick one label value to plot
target_label = 1.0  # change this to the label you want

# For float/continuous labels, exact equality can be brittle, so use tolerance
tol = 1e-8
subset = df_vec[
    np.isclose(df_vec["label"].astype(float), float(target_label), atol=tol)
].copy()

if subset.empty:
    raise ValueError(f"No rows found for target_label={target_label} (tol={tol})")

raw_color = "rgba(31, 119, 180, 0.85)"  # blue
enc_color = "rgba(214, 39, 40, 0.85)"  # red

# Connector lines for only the selected label
line_x = np.column_stack(
    [subset["raw_x"], subset["enc_x"], np.full(len(subset), np.nan)]
).ravel()
line_y = np.column_stack(
    [subset["raw_y"], subset["enc_y"], np.full(len(subset), np.nan)]
).ravel()

fig = go.Figure()

fig.add_trace(
    go.Scatter(
        x=line_x,
        y=line_y,
        mode="lines",
        line=dict(color="rgba(80,80,80,0.20)", width=1),
        hoverinfo="skip",
        showlegend=False,
    )
)

fig.add_trace(
    go.Scatter(
        x=subset["raw_x"],
        y=subset["raw_y"],
        mode="markers",
        marker=dict(
            size=9, symbol="circle", color=raw_color, line=dict(width=1, color="white")
        ),
        name=f"raw (label={target_label})",
        text=[
            f"idx={r}, label={lbl}" for r, lbl in zip(subset["idx"], subset["label"])
        ],
        hovertemplate="RAW<br>%{text}<br>x=%{x:.3f}, y=%{y:.3f}<extra></extra>",
    )
)

fig.add_trace(
    go.Scatter(
        x=subset["enc_x"],
        y=subset["enc_y"],
        mode="markers",
        marker=dict(
            size=9, symbol="diamond", color=enc_color, line=dict(width=1, color="white")
        ),
        name=f"encoded (label={target_label})",
        text=[
            f"idx={r}, label={lbl}, norm={n:.3f}"
            for r, lbl, n in zip(subset["idx"], subset["label"], subset["vec_norm"])
        ],
        hovertemplate="ENCODED<br>%{text}<br>x=%{x:.3f}, y=%{y:.3f}<extra></extra>",
    )
)

fig.update_layout(
    title=f"Per-index vectors for label={target_label} (N={len(subset)})",
    xaxis_title="PC1 (raw space) / PC1 (encoded space)",
    yaxis_title="PC2",
    width=980,
    height=720,
)
fig.show()

# +


# Derive vec_x / vec_y from whichever df_vec is in scope
vec_x = (df_vec["enc_x"] - df_vec["raw_x"]).values
vec_y = (df_vec["enc_y"] - df_vec["raw_y"]).values
V = np.stack([vec_x, vec_y], axis=1)  # [N, 2]
labels = df_vec["label"].values

# 1) Mean displacement and residuals
v_mean = V.mean(axis=0)
residuals = V - v_mean
residual_norms = np.linalg.norm(residuals, axis=1)
vector_norms = np.linalg.norm(V, axis=1)
explained_by_mean = 1 - (residual_norms / (vector_norms + 1e-9))

print(f"Mean displacement vector: ({v_mean[0]:.3f}, {v_mean[1]:.3f})")
print(f"Mean |v|:        {vector_norms.mean():.3f}  ±  {vector_norms.std():.3f}")
print(f"Mean |residual|: {residual_norms.mean():.3f}  ±  {residual_norms.std():.3f}")
print(f"Fraction explained by global shift: {explained_by_mean.mean():.3f}")

# 2) Angle relative to mean direction
v_mean_unit = v_mean / (np.linalg.norm(v_mean) + 1e-9)
cos_sim = np.clip(V @ v_mean_unit / (vector_norms + 1e-9), -1, 1)
angles_deg = np.degrees(np.arccos(cos_sim))

# 3) PCA of displacement vectors
pca_v = PCA(n_components=2).fit(V)
print(
    f"\nPCA of displacement vectors — variance explained: {pca_v.explained_variance_ratio_}"
)

df_angle = pd.DataFrame(
    {
        "angle_deg": angles_deg,
        "residual_norm": residual_norms,
        "vec_norm": vector_norms,
        "label": labels,
    }
)

fig = px.histogram(
    df_angle,
    x="angle_deg",
    color="label",
    nbins=30,
    barmode="overlay",
    opacity=0.7,
    title="Distribution of displacement angle relative to mean vector (degrees)<br>"
    "<sup>Near 0° = moves with the crowd; large angle = outlier movement</sup>",
    labels={"angle_deg": "Angle to mean displacement (°)"},
    width=850,
    height=450,
)
fig.show()

fig2 = px.scatter(
    df_angle,
    x="vec_norm",
    y="residual_norm",
    color="label",
    title="Vector norm vs residual norm after removing global shift<br>"
    "<sup>Near x-axis = moved with global trend; high residual = distinctive movement</sup>",
    labels={"vec_norm": "|v|", "residual_norm": "|v - v̄|"},
    width=800,
    height=500,
    opacity=0.8,
)
fig2.add_hline(y=0, line_dash="dot", line_color="gray")
fig2.show()
# -

# ## Downstream Tasks

import lightning as L

# +
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from lightning.pytorch.callbacks import EarlyStopping
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader, TensorDataset


class SmallMLPClassifier(L.LightningModule):
    def __init__(
        self,
        input_dim: int,
        n_classes: int,
        hidden_dim: int = 128,
        dropout: float = 0.2,
        lr: float = 1e-3,
        weight_decay: float = 1e-4,
        class_weights: torch.Tensor | None = None,
    ):
        super().__init__()
        self.save_hyperparameters(ignore=["class_weights"])

        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, n_classes),
        )
        self.loss_fn = nn.CrossEntropyLoss(weight=class_weights)

    def forward(self, x):
        return self.net(x)

    def training_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)
        loss = self.loss_fn(logits, y)
        self.log("train_loss", loss, prog_bar=False, on_epoch=True, on_step=False)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)
        loss = self.loss_fn(logits, y)
        self.log("val_loss", loss, prog_bar=False, on_epoch=True, on_step=False)
        return loss

    def predict_step(self, batch, batch_idx, dataloader_idx=0):
        x, _ = batch
        return self(x)

    def configure_optimizers(self):
        return torch.optim.AdamW(
            self.parameters(),
            lr=self.hparams.lr,
            weight_decay=self.hparams.weight_decay,
        )


def collect_raw_and_encoded(loader, model_module):
    x_raw, x_enc, y_all = [], [], []

    model_module.eval()
    device = model_module.device

    with torch.no_grad():
        for x_batch, y_batch in loader:
            x_batch = x_batch.float()

            raw_feat = x_batch.reshape(x_batch.size(0), -1).cpu().numpy()
            _, encoded = model_module.model(x_batch.to(device))
            # encoded is [B, T_enc, C]; pool over time (second dimension) -> [B, C]
            # enc_feat = encoded.mean(dim=1).cpu().numpy()
            # flatten the encoded
            enc_feat = encoded.reshape(encoded.size(0), -1).cpu().numpy()

            x_raw.append(raw_feat)
            x_enc.append(enc_feat)
            y_all.append(y_batch.cpu().numpy())

    X_raw = np.concatenate(x_raw, axis=0)
    X_enc = np.concatenate(x_enc, axis=0)
    y = np.concatenate(y_all, axis=0)
    return X_raw, X_enc, y


def fit_predict_mlp_on_fold(
    X_train,
    y_train,
    X_val,
    y_val,
    seed: int = 0,
    max_epochs: int = 80,
    batch_size: int = 32,
):
    # Scale inside each fold to avoid leakage
    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train).astype(np.float32)
    X_val_s = scaler.transform(X_val).astype(np.float32)

    y_train_t = torch.from_numpy(y_train.astype(np.int64))
    y_val_t = torch.from_numpy(y_val.astype(np.int64))

    train_ds = TensorDataset(torch.from_numpy(X_train_s), y_train_t)
    val_ds = TensorDataset(torch.from_numpy(X_val_s), y_val_t)

    train_loader = DataLoader(
        train_ds, batch_size=batch_size, shuffle=True, num_workers=0
    )
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=0)

    n_classes = int(np.unique(y_train).size)
    counts = np.bincount(y_train, minlength=n_classes).astype(np.float32)
    class_weights = counts.sum() / np.maximum(counts, 1.0)
    class_weights = class_weights / class_weights.mean()
    class_weights_t = torch.tensor(class_weights, dtype=torch.float32)

    L.seed_everything(seed, workers=True)

    model = SmallMLPClassifier(
        input_dim=X_train_s.shape[1],
        n_classes=n_classes,
        hidden_dim=64,
        dropout=0.2,
        lr=1e-3,
        weight_decay=1e-4,
        class_weights=class_weights_t,
    )

    trainer = L.Trainer(
        max_epochs=max_epochs,
        accelerator="auto",
        devices=1,
        logger=False,
        enable_checkpointing=False,
        enable_progress_bar=False,
        deterministic=True,
        callbacks=[EarlyStopping(monitor="val_loss", mode="min", patience=8)],
    )
    trainer.fit(model, train_dataloaders=train_loader, val_dataloaders=val_loader)

    pred_batches = trainer.predict(model, dataloaders=val_loader)
    logits = torch.cat(pred_batches, dim=0)
    y_pred = torch.argmax(logits, dim=1).cpu().numpy()
    return y_pred


def evaluate_representation_cv_mlp(X, y, splits, rep_name):
    rows = []
    for split_id, (idx_tr, idx_va) in enumerate(splits, start=1):
        y_pred = fit_predict_mlp_on_fold(
            X_train=X[idx_tr],
            y_train=y[idx_tr],
            X_val=X[idx_va],
            y_val=y[idx_va],
            seed=split_id,
            max_epochs=80,
            batch_size=32,
        )

        rows.append(
            {
                "representation": rep_name,
                "split_id": split_id,
                "repeat": (split_id - 1) // 5 + 1,
                "fold": (split_id - 1) % 5 + 1,
                "accuracy": accuracy_score(y[idx_va], y_pred),
                "f1_macro": f1_score(y[idx_va], y_pred, average="macro"),
                "f1_weighted": f1_score(y[idx_va], y_pred, average="weighted"),
            }
        )
    return pd.DataFrame(rows)


# Fixed CPC model
module = trainer.model if hasattr(trainer, "model") else lightning_module

# Feature extraction once (CPC frozen)
train_loader = data_module.train_dataloader()
test_loader = data_module.val_dataloader()

X_train_raw, X_train_enc, y_train = collect_raw_and_encoded(train_loader, module)
X_test_raw, X_test_enc, y_test = collect_raw_and_encoded(test_loader, module)

USE_TEST_IN_CV = False
if USE_TEST_IN_CV:
    X_raw = np.concatenate([X_train_raw, X_test_raw], axis=0)
    X_enc = np.concatenate([X_train_enc, X_test_enc], axis=0)
    y_all = np.concatenate([y_train, y_test], axis=0)
else:
    X_raw = X_train_raw
    X_enc = X_train_enc
    y_all = y_train

cv = RepeatedStratifiedKFold(n_splits=5, n_repeats=5, random_state=42)
splits = list(cv.split(X_raw, y_all))

df_raw = evaluate_representation_cv_mlp(X_raw, y_all, splits, "raw")
df_enc = evaluate_representation_cv_mlp(X_enc, y_all, splits, "encoded")
df_cv = pd.concat([df_raw, df_enc], ignore_index=True)

summary = (
    df_cv.groupby("representation")[["accuracy", "f1_macro", "f1_weighted"]]
    .agg(["mean", "std"])
    .round(4)
)

paired = df_raw[["split_id", "accuracy", "f1_macro", "f1_weighted"]].merge(
    df_enc[["split_id", "accuracy", "f1_macro", "f1_weighted"]],
    on="split_id",
    suffixes=("_raw", "_enc"),
)
paired["acc_delta_enc_minus_raw"] = paired["accuracy_enc"] - paired["accuracy_raw"]
paired["f1m_delta_enc_minus_raw"] = paired["f1_macro_enc"] - paired["f1_macro_raw"]
paired["f1w_delta_enc_minus_raw"] = (
    paired["f1_weighted_enc"] - paired["f1_weighted_raw"]
)

print("=== MLP Repeated Stratified 5x5 CV Summary (mean ± std) ===")
print(summary)

print("\n=== Paired Delta Across Same Splits (encoded - raw) ===")
print(
    paired[
        [
            "acc_delta_enc_minus_raw",
            "f1m_delta_enc_minus_raw",
            "f1w_delta_enc_minus_raw",
        ]
    ]
    .agg(["mean", "std", "min", "max"])
    .round(4)
)

print("\n=== Win Rate (encoded > raw) ===")
print(
    {
        "accuracy": float((paired["acc_delta_enc_minus_raw"] > 0).mean()),
        "f1_macro": float((paired["f1m_delta_enc_minus_raw"] > 0).mean()),
        "f1_weighted": float((paired["f1w_delta_enc_minus_raw"] > 0).mean()),
    }
)

df_cv.head()

# +


from scipy.stats import binomtest, ttest_rel, wilcoxon


def summarize_paired(metric_name, col_raw, col_enc):
    d = paired[col_enc] - paired[col_raw]
    n = len(d)

    t_res = ttest_rel(paired[col_enc], paired[col_raw], alternative="two-sided")
    w_res = wilcoxon(d, zero_method="wilcox", alternative="two-sided", correction=False)

    wins = int((d > 0).sum())
    losses = int((d < 0).sum())
    ties = int((d == 0).sum())

    # Sign test ignores ties
    n_sign = wins + losses
    sign_p = np.nan
    if n_sign > 0:
        sign_p = binomtest(k=wins, n=n_sign, p=0.5, alternative="two-sided").pvalue

    mean_delta = float(d.mean())
    std_delta = float(d.std(ddof=1))
    se = std_delta / np.sqrt(n)
    ci95_low = mean_delta - 1.96 * se
    ci95_high = mean_delta + 1.96 * se

    return {
        "metric": metric_name,
        "n_splits": n,
        "mean_delta_enc_minus_raw": mean_delta,
        "std_delta": std_delta,
        "ci95_low": ci95_low,
        "ci95_high": ci95_high,
        "wins": wins,
        "losses": losses,
        "ties": ties,
        "ttest_p": float(t_res.pvalue),
        "wilcoxon_p": float(w_res.pvalue),
        "sign_test_p": float(sign_p) if not np.isnan(sign_p) else np.nan,
    }


stats_rows = [
    summarize_paired("accuracy", "accuracy_raw", "accuracy_enc"),
    summarize_paired("f1_macro", "f1_macro_raw", "f1_macro_enc"),
    summarize_paired("f1_weighted", "f1_weighted_raw", "f1_weighted_enc"),
]

df_stats = pd.DataFrame(stats_rows).round(6)

print("=== Paired significance tests: encoded vs raw (same 25 splits) ===")
print(df_stats.to_string(index=False))

alpha = 0.05
print("\n=== Quick significance flags at alpha=0.05 ===")
for _, r in df_stats.iterrows():
    print(
        f"{r['metric']}: "
        f"ttest={r['ttest_p'] < alpha}, "
        f"wilcoxon={r['wilcoxon_p'] < alpha}, "
        f"sign_test={r['sign_test_p'] < alpha if not np.isnan(r['sign_test_p']) else False}"
    )
# -

#
