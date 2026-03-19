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

import numpy as np
import pandas as pd
import plotly.express as px

# +
import yaml
from pydantic import BaseModel, Field

# -

df = pd.DataFrame(
    {"t": np.linspace(0, 100, 501), "y": np.sin(np.linspace(0, 100, 501))}
)
df = df.astype(np.float32)

px.line(df, x="t", y="y")


# Data

from ts_bolt.datamodules.pandas import DataFrameDataModule

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
        # norm_factory = nn.BatchNorm1d
        norm_factory = nn.Identity

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


# -


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
        batch_x = batch_x.permute(0, 2, 1).float()
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
        x = data.permute(0, 2, 1).float()
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


# config = TrainConfig.from_yaml(
#     Path("configs/predictive_coding/config.ecg200.yaml")
# )
# config = TrainConfig.from_yaml(
# Path("configs/predictive_coding/config.ecg5000.yaml")
# )
# config = TrainConfig.from_yaml(
#     Path("configs/predictive_coding/config.forda.yaml")
# )
config = TrainConfig.from_yaml(Path("configs/predictive_coding/config.synth.yaml"))

config.model_dump()

# +
history_length_1_step = 100
horizon_1_step = 1

pdm_1_step = DataFrameDataModule(
    history_length=history_length_1_step,
    horizon=horizon_1_step,
    dataframe=df[["y"]],
)
# -

for i in pdm_1_step.train_dataloader():
    print(i[0].shape, i[1].shape)
    break

# +


L.seed_everything(config.runtime.seed, workers=True)
os.makedirs(config.trainer.output_dir, exist_ok=True)

lightning_module = CPCLightningModule(config=config, input_channels=1)

trainer = L.Trainer(
    max_epochs=config.trainer.epochs,
    accelerator=config.runtime.accelerator,
    devices=config.runtime.devices,
    default_root_dir=config.trainer.output_dir,
    log_every_n_steps=5,
    precision="32-true",  # Enforce float32 for MPS compatibility
)
trainer.fit(lightning_module, datamodule=pdm_1_step)
# -


# ## Load Artifacts and Interpret

import pandas as pd
import plotly.express as px
import torch
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

# ### Visualize One Batch

for i in pdm_1_step.train_dataloader():
    pred_data, pred_label = i
    break

pred_data.shape, pred_label.shape

# +
data_pca2 = PCA(
    n_components=2,
).fit_transform(pred_data.detach().numpy()[:, :, 0])

df_data_pca = pd.DataFrame(data_pca2, columns=["pc_1", "pc_2"])

df_data_pca["label"] = pred_label[:, 0, 0].numpy()

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
    color_continuous_scale="Viridis",
)

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

df_context["label"] = pred_label[:, 0, 0].numpy()

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
    color_continuous_scale="Viridis",
)

# +
context_tsne2d = TSNE(
    n_components=2,
    perplexity=20,
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

df_context_tsne["label"] = pred_label[:, 0, 0].numpy()

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
    color_continuous_scale="Viridis",
)

# +
encoded_pca2 = PCA(
    n_components=2,
).fit_transform(pred_encoded.detach().numpy()[:, 0, :])

df_encoded_pca = pd.DataFrame(encoded_pca2, columns=["pc_1", "pc_2"])

df_encoded_pca["label"] = pred_label[:, 0, 0].numpy()

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
    color_continuous_scale="Viridis",
)

# +
encoded_tsne2d = TSNE(
    n_components=2,
    perplexity=20,
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

df_encoded_tsne["label"] = pred_label[:, 0, 0].numpy()

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
    color_continuous_scale="Viridis",
)

# ### Visualizez All

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

module = trainer.model if hasattr(trainer, "model") else lightning_module
module.eval()
device = module.device

all_raw, all_context, all_encoded, all_labels = [], [], [], []
with torch.no_grad():
    for x, y_batch in pdm_1_step.train_dataloader():
        x = x.permute(0, 2, 1).float().to(device)
        context, encoded = module.model(x)

        all_raw.append(x.cpu())
        all_context.append(context.cpu())
        all_encoded.append(encoded.cpu())
        all_labels.append(y_batch[:, 0, 0].cpu())

raw_all = torch.cat(all_raw, dim=0)  # [N, C_in, T]
context_all = torch.cat(all_context, dim=0)  # [N, T_enc, C]
encoded_all = torch.cat(all_encoded, dim=0)  # [N, T_enc, C]
y = torch.cat(all_labels, dim=0).numpy().astype(np.float32)

# Choose one view per family for fair visual comparison
Z_by_family = {
    "raw": raw_all.reshape(raw_all.shape[0], -1).numpy(),  # flatten_all_channels
    "context": context_all.reshape(context_all.shape[0], -1).numpy(),  # flatten_tokens
    # "encoded": encoded_all.reshape(encoded_all.shape[0], -1).numpy(),# flatten_tokens
    "encoded": encoded_all[:, 1, :]
    .reshape(encoded_all.shape[0], -1)
    .numpy(),  # flatten_tokens
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
                "label": y,
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
        color_continuous_scale="Viridis",
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
label_values = df_vec["label"].astype(np.float32).to_numpy()

raw_color = "rgba(31, 119, 180, 0.2)"  # blue-ish connector tint

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
        line=dict(color=raw_color, width=1),
        hoverinfo="skip",
        showlegend=False,
    )
)

fig.add_trace(
    go.Scatter(
        x=df_vec["raw_x"],
        y=df_vec["raw_y"],
        mode="markers",
        marker=dict(
            size=8,
            symbol="circle",
            color=label_values,
            coloraxis="coloraxis",
            line=dict(width=1, color="white"),
        ),
        name="raw",
        text=[
            f"idx={r}, label={lbl:.4f}" for r, lbl in zip(df_vec["idx"], label_values)
        ],
        hovertemplate="RAW<br>%{text}<br>x=%{x:.3f}, y=%{y:.3f}<extra></extra>",
    )
)

fig.add_trace(
    go.Scatter(
        x=df_vec["enc_x"],
        y=df_vec["enc_y"],
        mode="markers",
        marker=dict(
            size=8,
            symbol="diamond",
            color=label_values,
            coloraxis="coloraxis",
            line=dict(width=1, color="white"),
        ),
        name="encoded",
        text=[
            f"idx={r}, label={lbl:.4f}, norm={n:.3f}"
            for r, lbl, n in zip(df_vec["idx"], label_values, df_vec["vec_norm"])
        ],
        hovertemplate="ENCODED<br>%{text}<br>x=%{x:.3f}, y=%{y:.3f}<extra></extra>",
    )
)

fig.update_layout(
    title=f"Per-index vectors — marker color: continuous label, shape: raw/encoded (N={len(df_vec)})",
    xaxis_title="PC1 (raw space) / PC1 (encoded space)",
    yaxis_title="PC2",
    width=980,
    height=720,
    coloraxis=dict(colorscale="Viridis", colorbar=dict(title="label")),
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


class MLPCForecaster(L.LightningModule):
    def __init__(
        self,
        input_dim: int,
        forecast_steps: int,
        hidden_dim: int = 128,
        dropout: float = 0.2,
        lr: float = 1e-3,
        pre_trained_model: nn.Module | None = None,
    ):
        super().__init__()
        self.save_hyperparameters(ignore=["pre_trained_model"])

        if pre_trained_model is not None:
            self.pre_net = pre_trained_model
        else:
            self.pre_net = nn.Identity()

        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, forecast_steps),
        )
        self.loss = nn.MSELoss()

    def forward(self, x):
        x = self.pre_net(x)
        return self.net(x)

    def training_step(self, batch, batch_idx):
        x, y = batch
        forecast = self(x.permute(0, 2, 1))
        loss = self.loss(forecast, y.squeeze())
        self.log("train_loss", loss, prog_bar=False, on_epoch=True, on_step=False)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        forecast = self(x.permute(0, 2, 1))
        loss = self.loss(forecast, y.squeeze())
        self.log("val_loss", loss, prog_bar=False, on_epoch=True, on_step=False)
        return loss

    def predict_step(self, batch, batch_idx, dataloader_idx=0):
        x, _ = batch
        return self(x.permute(0, 2, 1))

    def configure_optimizers(self):
        return torch.optim.AdamW(
            self.parameters(),
            lr=self.hparams.lr,
        )


model = MLPCForecaster(
    input_dim=history_length_1_step,
    forecast_steps=horizon_1_step,
    hidden_dim=64,
    dropout=0.2,
    lr=1e-3,
)

forecaster_trainer_naive = L.Trainer(
    max_epochs=100,
    accelerator="auto",
    devices=1,
    logger=False,
    enable_checkpointing=False,
    enable_progress_bar=False,
    deterministic=True,
    callbacks=[EarlyStopping(monitor="val_loss", mode="min", patience=8)],
)
forecaster_trainer_naive.fit(model, datamodule=pdm_1_step)

pred_batches = forecaster_trainer_naive.predict(model, datamodule=pdm_1_step)
y_pred = torch.cat(pred_batches, dim=0)


# -

#
