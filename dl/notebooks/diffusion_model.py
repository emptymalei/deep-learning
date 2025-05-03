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

# # Diffusion Model
#
# References:
# 1. [`pts/model/time_grad/time_grad_network.py`](https://github.com/zalandoresearch/pytorch-ts/blob/master/pts/model/time_grad/time_grad_network.py)

# +
import dataclasses
from functools import cached_property

import numpy as np
import pandas as pd
import plotly.express as px
import torch
import torch.nn as nn
from loguru import logger


# +
@dataclasses.dataclass
class DiffusionPocessParams:
    """Parameter that defines a diffusion process.

    :param steps: Number of steps in the diffusion process.
    :param beta: Beta parameter for the diffusion process.
    """

    steps: int
    beta: float

    @cached_property
    def alpha(self) -> float:
        r"""$\alpha = 1 - \beta$"""
        return 1.0 - self.beta

    @cached_property
    def beta_by_step(self) -> np.ndarray:
        """the beta parameter for each step in the diffusion process."""
        return np.array([self.beta] * self.steps)

    @cached_property
    def alpha_by_step(self) -> np.ndarray:
        """the alpha parameter for each step in the diffusion process."""
        return np.array([self.alpha] * self.steps)


def gaussian_noise(n_var: int, length: int) -> torch.Tensor:
    """Generate a Gaussian noise tensor.

    :param n_var: Number of variables.
    :param length: Length of the tensor.
    """
    return torch.normal(mean=0, std=1, size=(n_var, length))


class DiffusionProcess:
    """
    Diffusion process.

    :param params: DiffusionParams that defines how the diffusion process works
    :param noise: noise tensor, shape is (batch_size, params.steps)
    """

    def __init__(
        self,
        params: DiffusionPocessParams,
        noise: torch.Tensor,
        dtype: torch.dtype = torch.float32,
    ):
        self.params = params
        self.noise = noise
        self.dtype = dtype

    @cached_property
    def alpha_by_step(self) -> torch.Tensor:
        return torch.tensor(self.params.alpha_by_step, dtype=self.dtype)

    def _forward_process_by_step(self, state: torch.Tensor, step: int) -> torch.Tensor:
        r"""Assuming that we know the noise at step $t$,

        $$
        x(t) = \sqrt{\alpha(t)}x(t-1) + \sqrt{1 - \alpha(t)}\epsilon(t)
        $$
        """
        return (
            torch.sqrt(self.alpha_by_step[step]) * state
            + torch.sqrt(1 - self.alpha_by_step[step]) * self.noise[:, step]
        )

    def _inverse_process_by_step(self, state: torch.Tensor, step: int) -> torch.Tensor:
        r"""Assuming that we know the noise at step $t$,

        $$
        x(t-1) = \frac{1}{\sqrt{\alpha(t)}}
        (x(t) - \sqrt{1 - \alpha(t)}\epsilon(t))
        $$
        """
        return (
            state - torch.sqrt(1 - self.alpha_by_step[step]) * self.noise[:, step]
        ) / torch.sqrt(self.alpha_by_step[step])


# +
diffusion_process_params = DiffusionPocessParams(
    steps=100,
    beta=0.005,
    # beta=0,
)
diffusion_batch_size = 1000
# diffusion_batch_size = 2

noise = gaussian_noise(diffusion_batch_size, diffusion_process_params.steps)

diffusion_process = DiffusionProcess(diffusion_process_params, noise=noise)


# +
# diffusion_initial_x = torch.sin(
#     torch.linspace(0, 1, diffusion_batch_size)
#     .reshape(diffusion_batch_size)
# )

diffusion_initial_x = torch.rand(diffusion_batch_size)
# diffusion_initial_x = (
#     torch.distributions.Beta(torch.tensor([0.5]), torch.tensor([0.5]))
#     .sample((diffusion_batch_size, 1))
#     .reshape(diffusion_batch_size)
# )

diffusion_initial_x
# -


# ## Forward process step by step

# +
diffusion_steps_step_by_step = [diffusion_initial_x.detach().numpy()]

for i in range(0, diffusion_process_params.steps):
    logger.info(f"step {i}")
    i_state = (
        diffusion_process._forward_process_by_step(
            torch.from_numpy(diffusion_steps_step_by_step[-1]), step=i
        )
        .detach()
        .numpy()
    )
    logger.info(f"i_state {i_state[:2]}")
    diffusion_steps_step_by_step.append(i_state)

# -

px.histogram(diffusion_initial_x)

px.histogram(diffusion_steps_step_by_step[0])

px.histogram(diffusion_steps_step_by_step[-1])

# ## Reverse step by step

# +
diffusion_steps_reverse = [diffusion_steps_step_by_step[-1]]

for i in range(diffusion_process_params.steps - 1, -1, -1):
    logger.info(f"step {i}")
    i_state = (
        diffusion_process._inverse_process_by_step(
            torch.from_numpy(diffusion_steps_reverse[-1]), step=i
        )
        .detach()
        .numpy()
    )
    logger.info(f"i_state {i_state[:2]}")
    diffusion_steps_reverse.append(i_state)

# -

px.histogram(diffusion_steps_reverse[0])

px.histogram(diffusion_steps_reverse[-1])

# ## Diffusion Distributions

# +
df_diffusion_example = pd.DataFrame(
    {i: v for i, v in enumerate(diffusion_steps_step_by_step)}
).T
df_diffusion_example["step"] = df_diffusion_example.index

df_diffusion_example_melted = df_diffusion_example.melt(
    id_vars=["step"], var_name="variable", value_name="value"
)
df_diffusion_example_melted.tail()
# -

px.histogram(
    df_diffusion_example_melted,
    x="value",
    histnorm="probability density",
    animation_frame="step",
)

px.violin(
    df_diffusion_example_melted.loc[
        df_diffusion_example_melted["step"].isin(
            [0, 10, 20, 30, 40, 50, 60, 70, 80, 90]
        )
    ],
    x="step",
    y="value",
)

px.line(
    df_diffusion_example_melted,
    x="step",
    y="value",
    color="variable",
)

# ## Create Visuals

import matplotlib.pyplot as plt
import seaborn as sns

# +
_, ax = plt.subplots(figsize=(10, 6))
sns.histplot(
    df_diffusion_example_melted.loc[df_diffusion_example_melted["step"] == 0],
    x="value",
    stat="probability",
    color="k",
    label="Initial Distribution",
    ax=ax,
)

ax.set_title("Initial Distribution")
ax.set_xlabel("Position")

# +
_, ax = plt.subplots(figsize=(10, 6))

sns.histplot(
    df_diffusion_example_melted.loc[
        df_diffusion_example_melted["step"] == max(df_diffusion_example_melted["step"])
    ],
    x="value",
    stat="probability",
    color="k",
    ax=ax,
)

ax.set_title("Final Distribution")
ax.set_xlabel("Position")
# -


# # Model


# We create a naive model based on the idea of diffusion.
#
# 1. Connect the real data to the latent space through diffusion process.
# 2. We forecast in the latent space.

import lightning.pytorch as pl
from lightning import LightningModule


# +
@dataclasses.dataclass
class LatentRNNParams:
    """Parameters for Diffusion process.

    :param latent_size: latent space dimension
    :param history_length: input sequence length
    :param n_features: number of features
    """

    history_length: int
    latent_size: int = 100
    num_layers: int = 2
    n_features: int = 1
    initial_state: torch.Tensor = None

    @cached_property
    def data_size(self) -> int:
        """The dimension of the input data
        when flattened.
        """
        return self.sequence_length * self.n_features

    def asdict(self) -> dict:
        return dataclasses.asdict(self)


class LatentRNN(nn.Module):
    """Forecasting the next step in latent space."""

    def __init__(self, params: LatentRNNParams):
        super().__init__()

        self.params = params
        self.hparams = params.asdict()

        self.rnn = nn.GRU(
            input_size=self.params.history_length,
            hidden_size=self.params.latent_size,
            num_layers=self.params.num_layers,
            batch_first=True,
        )

    def forward(
        self, x: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """

        :param x: input data, shape (batch_size, history_length * n_features)
        """

        outputs, _ = self.rnn(x, self.params.initial_state)

        return outputs


class DiffusionDecoder(nn.Module):
    """Decoding the time series into the latent space."""

    def __init__(
        self,
        params: DiffusionPocessParams,
        noise: torch.Tensor,
    ):
        super().__init__()
        self.params = params
        self.noise = noise

    @staticmethod
    def _forward_process_by_step(
        state: torch.Tensor, alpha_by_step: torch.Tensor, noise: torch.Tensor, step: int
    ) -> torch.Tensor:
        r"""Assuming that we know the noise at step $t$,

        $$
        x(t) = \sqrt{\alpha(t)}x(t-1) + \sqrt{1 - \alpha(t)}\epsilon(t)
        $$
        """
        batch_size = state.shape[0]
        return torch.sqrt(alpha_by_step[step]) * state + (
            torch.sqrt(1 - alpha_by_step[step]) * noise[:batch_size, step]
        ).reshape(batch_size, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Encoding the latent space into a distribution.

        :param x: input data, shape (batch_size, history_length, n_features)
        """

        alpha_by_step = torch.tensor(self.params.alpha_by_step).to(x)
        self.noise = self.noise.to(x)
        # logger.debug(
        #     f"alpha_by_step: {alpha_by_step.shape}"
        #     f"noise: {self.noise.shape}"
        #     f"x: {x.shape}"
        # )

        diffusion_steps_step_by_step = [x]

        for i in range(0, self.params.steps):
            i_state = self._forward_process_by_step(
                diffusion_steps_step_by_step[-1],
                alpha_by_step=alpha_by_step,
                noise=self.noise,
                step=i,
            )
            diffusion_steps_step_by_step.append(i_state)

        return diffusion_steps_step_by_step[-1]


class DiffusionEncoder(nn.Module):
    """Encode the latent space into a distribution."""

    def __init__(
        self,
        params: DiffusionPocessParams,
        noise: torch.Tensor,
    ):
        super().__init__()
        self.params = params
        self.noise = noise

    @staticmethod
    def _inverse_process_by_step(
        state: torch.Tensor, alpha_by_step: torch.Tensor, noise: torch.Tensor, step: int
    ) -> torch.Tensor:
        r"""Assuming that we know the noise at step $t$,

        $$
        x(t-1) = \frac{1}{\sqrt{\alpha(t)}}
        (x(t) - \sqrt{1 - \alpha(t)}\epsilon(t))
        $$
        """
        batch_size = state.shape[0]
        return (
            state
            - (torch.sqrt(1 - alpha_by_step[step]) * noise[:batch_size, step]).reshape(
                batch_size, 1
            )
        ) / torch.sqrt(alpha_by_step[step])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Encoding the latent space into a distribution.

        :param x: input data, shape (batch_size, history_length, n_features)
        """
        alpha_by_step = torch.tensor(self.params.alpha_by_step).to(x)
        self.noise = self.noise.to(x)

        diffusion_steps_reverse = [x]

        for i in range(self.params.steps - 1, -1, -1):
            i_state = self._inverse_process_by_step(
                state=diffusion_steps_reverse[-1],
                alpha_by_step=alpha_by_step,
                noise=self.noise,
                step=i,
            )
            diffusion_steps_reverse.append(i_state)

        return diffusion_steps_reverse[-1]


class NaiveDiffusionModel(nn.Module):
    """A naive diffusion model that explicitly calculates
    the diffusion process.
    """

    def __init__(
        self,
        rnn: LatentRNN,
        diffusion_decoder: DiffusionDecoder,
        diffusion_encoder: DiffusionEncoder,
        horizon: int = 1,
    ):
        super().__init__()
        self.rnn = rnn
        self.diffusion_decoder = diffusion_decoder
        self.diffusion_encoder = diffusion_encoder
        self.horizon = horizon
        self.scale = nn.Linear(
            in_features=self.rnn.params.latent_size,
            out_features=self.horizon,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:

        # logger.debug(f"x.squeeze(-1): {x.squeeze(-1).shape=}")
        x_latent = self.diffusion_encoder(x.squeeze(-1))
        # logger.debug(f"x_latent: {x_latent.shape=}")
        y_latent = self.rnn(x_latent)
        # logger.debug(f"y_latent: {y_latent.shape=}")
        y_hat = self.diffusion_decoder(y_latent)
        # logger.debug(f"y_hat: {y_hat.shape=}")
        y_hat = self.scale(y_hat)
        # logger.debug(f"scaled y_hat: {y_hat.shape=}")

        return y_hat


class NaiveDiffusionForecaster(LightningModule):
    """A assembled lightning module for the naive diffusion model."""

    def __init__(
        self,
        model: NaiveDiffusionModel,
        loss: nn.Module = nn.MSELoss(),
    ):
        super().__init__()
        self.model = model
        self.loss = loss

    def configure_optimizers(self):
        optimizer = torch.optim.SGD(self.parameters(), lr=1e-3)
        return optimizer

    def training_step(self, batch, batch_idx):
        x, y = batch
        x = x.type(self.dtype)
        y = y.type(self.dtype)
        batch_size = x.shape[0]

        y_hat = self.model(x)[:batch_size, :].reshape_as(y)

        loss = self.loss(y_hat, y).mean()
        self.log_dict({"train_loss": loss}, prog_bar=True)

        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        x = x.type(self.dtype)
        y = y.type(self.dtype)
        batch_size = x.shape[0]

        y_hat = self.model(x)[:batch_size, :].reshape_as(y)

        loss = self.loss(y_hat, y).mean()
        self.log_dict({"val_loss": loss}, prog_bar=True)
        return loss

    def predict_step(self, batch, batch_idx):
        x, y = batch
        x = x.type(self.dtype)
        y = y.type(self.dtype)
        batch_size = x.shape[0]

        y_hat = self.model(x)[:batch_size, :].reshape_as(y)
        return x, y_hat

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.to(self.model.rnn.rnn.weight_ih_l0)
        return self.model(x)


# +
df = pd.DataFrame(
    {"t": np.linspace(0, 100, 501), "y": np.sin(np.linspace(0, 100, 501))}
)

_, ax = plt.subplots(figsize=(10, 6.18))

df.plot(x="t", y="y", ax=ax)
# -


# ## Traning

from ts_bolt.datamodules.pandas import DataFrameDataModule

# +
history_length_1_step = 100
horizon_1_step = 1
training_batch_size = 64

training_noise = gaussian_noise(training_batch_size, diffusion_process_params.steps)
# -

diffusion_process_params.alpha_by_step.shape, training_noise.shape

test_state = torch.rand(training_batch_size, diffusion_process_params.steps)
test_state.shape

torch.sqrt(torch.from_numpy(diffusion_process_params.alpha_by_step)[0])


(
    test_state
    - torch.sqrt(torch.from_numpy(diffusion_process_params.alpha_by_step)[0])
    * training_noise[:, 0].reshape(training_batch_size, 1)
).shape

pdm_1_step = DataFrameDataModule(
    history_length=history_length_1_step,
    horizon=horizon_1_step,
    dataframe=df[["y"]].astype(np.float32),
    batch_size=training_batch_size,
)

diffusion_decoder = DiffusionDecoder(diffusion_process_params, training_noise)
diffusion_encoder = DiffusionEncoder(diffusion_process_params, training_noise)

# +
latent_rnn_params = LatentRNNParams(
    history_length=history_length_1_step,
    latent_size=diffusion_process_params.steps,
)

latent_rnn = LatentRNN(latent_rnn_params)
# -

naive_diffusion_model = NaiveDiffusionModel(
    rnn=latent_rnn,
    diffusion_decoder=diffusion_decoder,
    diffusion_encoder=diffusion_encoder,
)
naive_diffusion_forecaster = NaiveDiffusionForecaster(
    model=naive_diffusion_model.float(),
)

naive_diffusion_forecaster

# +
logger_1_step = pl.loggers.TensorBoardLogger(
    save_dir="lightning_logs", name="naive_diffusion_ts_1_step"
)

trainer_1_step = pl.Trainer(
    precision="32",
    max_epochs=5000,
    min_epochs=5,
    # callbacks=[
    #     pl.callbacks.early_stopping.EarlyStopping(monitor="val_loss", mode="min", min_delta=1e-8, patience=4)
    # ],
    logger=logger_1_step,
    accelerator="mps",
)
# -

trainer_1_step.fit(model=naive_diffusion_forecaster, datamodule=pdm_1_step)

# # Evaluation

from ts_bolt.evaluation.evaluator import Evaluator
from ts_bolt.naive_forecasters.last_observation import LastObservationForecaster

evaluator_1_step = Evaluator(step=0)

predictions_1_step = trainer_1_step.predict(
    model=naive_diffusion_forecaster, datamodule=pdm_1_step
)

# +
fig, ax = plt.subplots(figsize=(10, 6.18))

ax.plot(
    evaluator_1_step.y_true(dataloader=pdm_1_step.predict_dataloader()),
    "g-",
    label="truth",
)

ax.plot(evaluator_1_step.y(predictions_1_step), "r--", label="predictions")

# ax.plot(evaluator_1_step.y(lobs_1_step_predictions), "b-.", label="naive predictions")

plt.legend()

# +
trainer_naive_1_step = pl.Trainer(precision="32")

lobs_forecaster_1_step = LastObservationForecaster(horizon=horizon_1_step)
lobs_1_step_predictions = trainer_naive_1_step.predict(
    model=lobs_forecaster_1_step, datamodule=pdm_1_step
)
