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
#     display_name: deep-learning-py3.10
#     language: python
#     name: python3
# ---

# # Transformer for Univariate Time Series Forecasting
#
# In this notebook, we build a transformer using pytorch to forecast $\sin$ function as a time series.

import dataclasses

# +
import math

import lightning as L
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from lightning.pytorch.callbacks.early_stopping import EarlyStopping
from torch import nn
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


pen = Pendulum(length=10000)

df = pd.DataFrame(pen(100, 400, initial_angle=1, beta=0.000001))

# Since the damping constant is very small, the data generated is mostly a sin wave.

# +
_, ax = plt.subplots(figsize=(10, 6.18))

df.plot(x="t", y="theta", ax=ax)


# -

# ## Model
#
# In this section, we create the transformer model.

# Since we do not deal with future covariates, we do not need a decoder. In this example, we build a simple transformer that only contains attention in encoder.


# +
@dataclasses.dataclass
class TSTransformerParams:
    """A dataclass that contains all
    the parameters for the transformer model.
    """

    d_model: int = 512
    nhead: int = 8
    num_encoder_layers: int = 6
    dropout: int = 0.1


class PositionalEncoding(nn.Module):
    """Positional encoding to be added to
    input embedding.

    :param d_model: hidden dimension of the encoder
    :param dropout: rate of dropout
    :param max_len: maximum length of our positional
        encoder. The encoder can not encode sequence
        length longer than max_len.
    """

    def __init__(
        self,
        d_model: int,
        dropout: float = 0.1,
        max_len: int = 5000,
    ):
        super().__init__()
        self.max_len = max_len
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
        :param x: input embedded time series,
            shape `[batch_size, seq_len, embedding_dim]`
        """
        history_length = x.size(1)
        x = x + self.pe[:history_length]

        return self.dropout(x)


class TSTransformer(nn.Module):
    """Transformer for univaraite time series modeling.

    :param history_length: the length of the input history.
    :param horizon: the number of steps to be forecasted.
    :param transformer_params: all the parameters.
    """

    def __init__(
        self,
        history_length: int,
        horizon: int,
        transformer_params: TSTransformerParams,
    ):
        super().__init__()
        self.transformer_params = transformer_params
        self.history_length = history_length
        self.horizon = horizon

        self.embedding = nn.Linear(1, self.transformer_params.d_model)

        self.positional_encoding = PositionalEncoding(
            d_model=self.transformer_params.d_model
        )

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=self.transformer_params.d_model,
            nhead=self.transformer_params.nhead,
            batch_first=True,
        )
        self.encoder = nn.TransformerEncoder(
            encoder_layer, num_layers=self.transformer_params.num_encoder_layers
        )

        self.reverse_embedding = nn.Linear(self.transformer_params.d_model, 1)

        self.decoder = nn.Linear(self.history_length, self.horizon)

    @property
    def transformer_config(self) -> dict:
        """all the param in dict format"""
        return dataclasses.asdict(self.transformer_params)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        :param x: input historical time series,
            shape `[batch_size, seq_len, n_var]`
        """
        x = self.embedding(x)
        x = self.positional_encoding(x)

        encoder_state = self.encoder(x)

        decoder_in = self.reverse_embedding(encoder_state).squeeze(-1)

        return self.decoder(decoder_in)


# -

# ## Training

# We use [lightning](https://lightning.ai/docs/pytorch/stable/) to train our model.

# ### Training Utilities

# +
history_length_1_step = 100
horizon_1_step = 1

gap = 0
# -


# We will build a few utilities
#
# 1. To be able to feed the data into our model, we build a class (`DataFrameDataset`) that converts the pandas dataframe into a Dataset for pytorch.
# 2. To make the lightning training code simpler, we will build a [LightningDataModule](https://lightning.ai/docs/pytorch/stable/data/datamodule.html) (`PendulumDataModule`) and a [LightningModule](https://lightning.ai/docs/pytorch/stable/common/lightning_module.html) (`TransformerForecaster`).


class TransformerForecaster(L.LightningModule):
    """Transformer forecasting training, validation,
    and prediction all collected in one class.

    :param transformer: pre-defined transformer model
    """

    def __init__(self, transformer: nn.Module):
        super().__init__()
        self.transformer = transformer
        self.save_hyperparameters()

    def configure_optimizers(self) -> torch.optim.Optimizer:
        optimizer = torch.optim.SGD(self.parameters(), lr=1e-3)

        return optimizer

    def training_step(self, batch: tuple[torch.Tensor], batch_idx: int) -> torch.Tensor:
        x, y = batch
        y = y.squeeze(-1).type(self.dtype)

        y_hat = self.transformer(x)

        loss = nn.functional.mse_loss(y_hat, y)
        self.log_dict({"train_loss": loss}, prog_bar=True)
        return loss

    def validation_step(
        self, batch: tuple[torch.Tensor], batch_idx: int
    ) -> torch.Tensor:
        x, y = batch
        y = y.squeeze(-1).type(self.dtype)

        y_hat = self.transformer(x)

        loss = nn.functional.mse_loss(y_hat, y)
        self.log_dict({"val_loss": loss}, prog_bar=True)

        return loss

    def predict_step(
        self, batch: list[torch.Tensor], batch_idx: int
    ) -> tuple[torch.Tensor]:
        x, y = batch
        y = y.squeeze(-1).type(self.dtype)

        y_hat = self.transformer(x)

        return x, y_hat

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor]:
        return x, self.transformer(x)


# ### Data, Model and Training

# #### DataModule

pdm_1_step = PendulumDataModule(
    history_length=history_length_1_step,
    horizon=horizon_1_step,
    dataframe=df[["theta"]],
    gap=gap,
)

# #### LightningModule

# +
ts_transformer_params_1_step = TSTransformerParams(
    d_model=192, nhead=6, num_encoder_layers=1
)

ts_transformer_1_step = TSTransformer(
    history_length=history_length_1_step,
    horizon=horizon_1_step,
    transformer_params=ts_transformer_params_1_step,
)

ts_transformer_1_step

# +
transformer_forecaster_1_step = TransformerForecaster(transformer=ts_transformer_1_step)

transformer_forecaster_1_step
# -

# #### Trainer

# +
logger_1_step = L.pytorch.loggers.TensorBoardLogger(
    save_dir="lightning_logs", name="transformer_ts_1_step"
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

demo_x = list(pdm_1_step.train_dataloader())[0][0].type(
    transformer_forecaster_1_step.dtype
)
demo_x.shape

nn.Linear(
    1,
    ts_transformer_1_step.transformer_params.d_model,
    dtype=transformer_forecaster_1_step.dtype,
)(demo_x).shape

ts_transformer_1_step.encoder(ts_transformer_1_step.embedding(demo_x)).shape

trainer_1_step.fit(model=transformer_forecaster_1_step, datamodule=pdm_1_step)

# #### Retrieving Predictions

predictions_1_step = trainer_1_step.predict(
    model=transformer_forecaster_1_step, datamodule=pdm_1_step
)

# ### Naive Forecaster

# +
trainer_naive_1_step = L.Trainer(precision="64")

lobs_forecaster_1_step = LastObservationForecaster(horizon=horizon_1_step)
lobs_1_step_predictions = trainer_naive_1_step.predict(
    model=lobs_forecaster_1_step, datamodule=pdm_1_step
)
# -

# ## Evaluations

evaluator_1_step = Evaluator(step=0)

# +
fig, ax = plt.subplots(figsize=(50, 6.18))

ax.plot(
    evaluator_1_step.y_true(dataloader=pdm_1_step.predict_dataloader()),
    "g-",
    label="truth",
)

ax.plot(evaluator_1_step.y(predictions_1_step), "r--", label="predictions")

ax.plot(evaluator_1_step.y(lobs_1_step_predictions), "b-.", label="naive predictions")

plt.legend()

# +
fig, ax = plt.subplots(figsize=(10, 6.18))

inspection_slice_length = 200

ax.plot(
    evaluator_1_step.y_true(dataloader=pdm_1_step.predict_dataloader())[
        :inspection_slice_length
    ],
    "g-",
    label="truth",
)

ax.plot(
    evaluator_1_step.y(predictions_1_step)[:inspection_slice_length],
    "r--",
    label="predictions",
)

ax.plot(
    evaluator_1_step.y(lobs_1_step_predictions)[:inspection_slice_length],
    "b-.",
    label="naive predictions",
)

plt.legend()
# -

# To quantify the results, we compute a few metrics.

pd.merge(
    evaluator_1_step.metrics(predictions_1_step, pdm_1_step.predict_dataloader()),
    evaluator_1_step.metrics(lobs_1_step_predictions, pdm_1_step.predict_dataloader()),
    how="left",
    left_index=True,
    right_index=True,
    suffixes=["_transformer", "_naive"],
)

# Here SMAPE is better because of better forecasts for larger values

# ## Forecasting (horizon=3)

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
ts_transformer_params_m_step = TSTransformerParams(
    d_model=192, nhead=6, num_encoder_layers=1
)

ts_transformer_m_step = TSTransformer(
    history_length=history_length_m_step,
    horizon=horizon_m_step,
    transformer_params=ts_transformer_params_m_step,
)

ts_transformer_m_step
# -

transformer_forecaster_m_step = TransformerForecaster(transformer=ts_transformer_m_step)

# +
logger_m_step = L.pytorch.loggers.TensorBoardLogger(
    save_dir="lightning_logs", name="transformer_ts_m_step"
)


trainer_m_step = L.Trainer(
    precision="64",
    max_epochs=100,
    min_epochs=5,
    callbacks=[
        EarlyStopping(monitor="val_loss", mode="min", min_delta=1e-7, patience=3)
    ],
    logger=logger_m_step,
)
# -

trainer_m_step.fit(model=transformer_forecaster_m_step, datamodule=pdm_m_step)

predictions_m_step = trainer_m_step.predict(
    model=transformer_forecaster_m_step, datamodule=pdm_m_step
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

# # Investigations
#
# We dive deeper into the intermediate results of the model.

# +
from pathlib import Path

import pandas as pd
import plotly.express as px
from torch.utils.data import DataLoader
from ts_dl_utils.datasets.dataset import DataFrameDataset

# +
load_from_checkpoint = (
    Path("lightning_logs/transformer_ts_1_step/version_9") / "checkpoints"
)

# load_from_checkpoint = Path(logger_1_step.log_dir) / "checkpoints"
load_from_checkpoint, logger_1_step.log_dir

# +
transformer_forecaster_1_step_re = TransformerForecaster.load_from_checkpoint(
    # load_from_checkpoint / "checkpoints/epoch=11-step=5495.ckpt"
    list(load_from_checkpoint.iterdir())[0]
)

transformer_forecaster_1_step_re


# -

# ## Visualize Embeddings of Intermediate Layers


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

    dr_df["input"] = np.concatenate(
        input_example[:n_batches].detach().numpy().astype("float32")
    )

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


def embedding_extractor(
    forecaster: TransformerForecaster, x: torch.Tensor
) -> tuple[torch.Tensor]:
    """compute the embeddings based on the input

    :param forecaster: the trained forecaster
    :param x: input historical time series,
    """
    forecaster.transformer.to(x.device)
    x_embedding = forecaster.transformer.embedding(
        x.type_as(forecaster.transformer.embedding.weight)
    )
    x_positional = forecaster.transformer.positional_encoding(x_embedding)

    encoder_state = forecaster.transformer.encoder(x_positional)

    reversed = forecaster.transformer.reverse_embedding(encoder_state).squeeze(-1)

    return x_embedding, x_positional, encoder_state, reversed


# Prepare input data for embedding visualization.

# +
investigation_dl = DataLoader(
    dataset=DataFrameDataset(
        dataframe=df[["theta"]],
        history_length=history_length_1_step,
        horizon=horizon_1_step,
        gap=gap,
    ),
    batch_size=400,
    shuffle=False,
)

investigation_dl
# -

input_example = list(investigation_dl)[0][0]
input_example.shape

(embedding_example, positional_example, encoder_example, reversed_example) = (
    embedding_extractor(transformer_forecaster_1_step, input_example)
)

(
    embedding_example.shape,
    positional_example.shape,
    encoder_example.shape,
    reversed_example.shape,
)

n_batches, seq_len, _ = input_example.shape
df_example = pd.DataFrame(
    {
        "input": input_example.squeeze(-1).detach().cpu().numpy().reshape(-1),
        "reversed": reversed_example.detach().cpu().numpy().reshape(-1),
        "sample": np.repeat(np.arange(n_batches), seq_len),
    }
)
df_example


px.scatter(
    df_example, x="input", y="reversed", color="sample", height=600, width=600
).update_layout(yaxis_scaleanchor="x")

import numpy as np
from torchdr import PCA, TSNE, UMAP

# Latent space visualization

# +
n_components = 3

dr_reversed_result = TSNE(
    # n_neighbors=30, backend='torch',
    perplexity=30,
    n_components=n_components,
).fit_transform(
    np.concatenate(
        [
            input_example.squeeze(-1).numpy().astype("float32"),
            reversed_example.detach().numpy().astype("float32"),
        ],
        axis=0,
    )
)
# -
dr_reversed_df = pd.DataFrame(
    dr_reversed_result, columns=[f"DR_{i+1}" for i in range(n_components)]
)
dr_reversed_df["type"] = ["input"] * (len(dr_reversed_df) // 2) + ["embedded"] * (
    len(dr_reversed_df) - len(dr_reversed_df) // 2
)
dr_reversed_df["sample_idx"] = list(range(len(dr_reversed_df) // 2)) + list(
    range(len(dr_reversed_df) // 2)
)


px.scatter_3d(
    dr_reversed_df,
    x="DR_1",
    y="DR_2",
    z="DR_3",
    color="sample_idx",
    # color="DR_4",
    symbol="type",
    title="Embedding of Input and Encoded Time Series",
    height=600,
    width=800,
).update_layout(legend=dict(itemsizing="constant", orientation="h", y=-0.2)).show()

px.scatter_3d(
    dr_reversed_df.loc[dr_reversed_df["type"] == "embedded"],
    x="DR_1",
    y="DR_2",
    z="DR_3",
    color="sample_idx",
    # color="DR_4",
    symbol="type",
    title="Embedding of Input Time Series",
    height=600,
    width=800,
).update_layout(legend=dict(itemsizing="constant", orientation="h", y=-0.2)).show()

# Embedding outout

embedding_example.detach()[0].shape, input_example.detach()[0].shape

# +
embedding_result = TSNE(
    # n_neighbors=30, backend='torch',
    perplexity=30,
    n_components=n_components,
).fit_transform(
    # np.concatenate(
    #     embedding_example.detach().numpy().astype("float32")
    # )
    embedding_example.detach().reshape(-1, embedding_example.shape[-1])
)

embedding_result.shape
# -

dr_embedding_df = create_embedding_dataframe(
    dr_result=embedding_result,
    n_batches=embedding_example.shape[0],
    input_example=input_example,
    n_components=n_components,
)

dr_embedding_df

px.scatter(
    dr_embedding_df,
    x="DR_1",
    y="DR_2",
    # z='DR_3',
    # color='sample_idx',
    # symbol='type',
    color="batch_first_value",
    # color="input",
    title="Dimension Reduction for Embedding of Input Time Series",
    height=600,
    width=800,
).update_layout(legend=dict(itemsizing="constant", orientation="h", y=-0.2)).show()

px.scatter_3d(
    dr_embedding_df,
    x="DR_1",
    y="DR_2",
    z="input",
    # color='sample_idx',
    # symbol='type',
    # color="batch",
    # color="batch_first_value",
    title="UMAP Embedding of Input Time Series",
    height=600,
    width=800,
).update_layout(legend=dict(itemsizing="constant", orientation="h", y=-0.2)).show()

px.scatter(
    pd.DataFrame(input_example.detach()[:, 0, :].numpy()).reset_index(),
    x="index",
    y=0,
    height=600,
    width=800,
)

# Positional encoding output

positional_example_sample_size = 100

# +
positional_result = TSNE(
    # n_neighbors=30, backend='torch',
    perplexity=30,
    n_components=n_components,
).fit_transform(
    positional_example[:positional_example_sample_size]
    .detach()
    .reshape(-1, positional_example.shape[-1])
)

positional_result.shape
# -

dr_positional_df = create_embedding_dataframe(
    dr_result=positional_result,
    n_batches=positional_example_sample_size,
    input_example=input_example,
    n_components=n_components,
)

# +
dr_positional_df = pd.DataFrame(
    positional_result.detach().numpy(),
    columns=[f"DR_{i+1}" for i in range(n_components)],
)

dr_positional_df["batch"] = sum(
    [
        [i] * (len(dr_positional_df) // positional_example_sample_size)
        for i in range(positional_example_sample_size)
    ],
    [],
)

dr_positional_df["sample_idx"] = (
    list(range(len(dr_positional_df) // positional_example_sample_size))
    * positional_example_sample_size
)

dr_positional_df["input"] = np.concatenate(
    input_example[:positional_example_sample_size].detach().numpy().astype("float32")
)

dr_positional_df = dr_positional_df.merge(
    pd.DataFrame(
        input_example[:positional_example_sample_size].detach()[:, 0, 0].numpy(),
        columns=["batch_first_value"],
    )
    .reset_index()
    .rename(columns={"index": "batch"}),
    how="left",
    on="batch",
)
# -


dr_positional_df

px.scatter_3d(
    dr_positional_df,
    x="DR_1",
    y="DR_2",
    z="input",
    # color='sample_idx',
    # symbol='type',
    # color="batch",
    color="batch_first_value",
    title="UMAP Embedding of Positional Encoded Time Series",
    height=600,
    width=800,
).update_layout(legend=dict(itemsizing="constant", orientation="h", y=-0.2)).show()

px.scatter(
    dr_positional_df,
    x="DR_1",
    y="DR_2",
    # z='input',
    # symbol='type',
    color="batch",
    # color="input",
    # color="batch_first_value",
    title="UMAP Embedding of Positional Encoded Time Series",
    height=600,
    width=800,
).update_layout(legend=dict(itemsizing="constant", orientation="h", y=-0.2)).show()

# Encoder output

# +
encoder_result = TSNE(
    # n_neighbors=30, backend='torch',
    perplexity=30,
    n_components=n_components,
).fit_transform(encoder_example.detach().reshape(-1, encoder_example.shape[-1]))

encoder_result.shape
# -

dr_encoder_df = create_embedding_dataframe(
    dr_result=encoder_result,
    n_batches=encoder_example.shape[0],
    input_example=input_example,
    n_components=n_components,
)

px.scatter_3d(
    dr_encoder_df,
    x="DR_1",
    y="DR_2",
    z="input",
    # color='sample_idx',
    # symbol='type',
    # color="batch",
    color="batch_first_value",
    title="UMAP Embedding of Encoder Encoded Time Series",
    height=600,
    width=800,
).update_layout(legend=dict(itemsizing="constant", orientation="h", y=-0.2)).show()
