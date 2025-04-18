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

# -


# +
@dataclasses.dataclass
class DiffusionParams:
    """ """

    steps: int
    beta: float

    @cached_property
    def alpha(self) -> float:
        """ """
        return 1.0 - self.beta

    @cached_property
    def alpha_bar_by_step(self) -> np.ndarray:
        """ """
        return np.cumprod(self.alpha_by_step)

    @cached_property
    def beta_by_step(self) -> np.ndarray:
        """ """
        return np.array([self.beta] * self.steps)

    @cached_property
    def alpha_by_step(self) -> np.ndarray:
        """ """
        return np.array([self.alpha] * self.steps)

    # @cached_property
    # def sigma_q_squared(self) -> float:
    #     r"""

    #     $$
    #     \sigma_q^2(t) = \frac{
    #         (1 - \alpha(t))(1 - \bar\alpha(t-1))
    #     }{1 - \bar\alpha(t)}
    #     $$
    #     """
    #     sigma_q_squared = (
    #         1 - self.alpha_by_step[1:]
    #     ) * (1 - self.alpha_bar_by_step[:-1]) / (1 - self.alpha_bar_by_step[1:])
    #     return np.insert(sigma_q_squared, 0, 0)


class Diffusion(nn.Module):
    def __init__(
        self,
        batch_size: int,
        params: DiffusionParams,
        dtype: torch.dtype = torch.float32,
    ):
        self.batch_size = batch_size
        self.params = params
        self.dtype = dtype

    @cached_property
    def alpha_bar_by_step(self) -> torch.Tensor:
        return torch.tensor(self.params.alpha_bar_by_step, dtype=self.dtype).detach()

    @cached_property
    def alpha_by_step(self) -> torch.Tensor:
        return torch.tensor(self.params.alpha_by_step, dtype=self.dtype).detach()

    # @cached_property
    # def sigma_q_squared(self) -> torch.Tensor:
    #     return torch.tensor(self.params.sigma_q_squared, dtype=self.dtype)

    def likelihood(self):
        pass

    def _gaussian(self, n_var: int, length: int) -> torch.Tensor:
        return torch.normal(mean=0, std=1, size=(n_var, length))

    @cached_property
    def noise(self) -> torch.Tensor:
        return self._gaussian(self.batch_size, self.params.steps).detach()

    def _forward_process(self, initial: torch.Tensor) -> torch.Tensor:
        return (
            torch.outer(
                initial,
                torch.sqrt(self.alpha_bar_by_step),
            )
            + torch.sqrt(1 - self.alpha_bar_by_step) * self.noise
        )

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

    def forward(self):
        pass


# -

diffusion_params = DiffusionParams(
    steps=100,
    beta=0.005,
    # beta=0,
)
diffusion_batch_size = 1000
# diffusion_batch_size = 2
diffusion_process = Diffusion(diffusion_batch_size, diffusion_params)


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

for i in range(0, diffusion_params.steps):
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

# ## Calculate State at time t in one batch

# +
diffusion_steps = diffusion_process._forward_process(diffusion_initial_x)

diffusion_initial_x.shape, diffusion_steps.shape
# -

px.histogram(diffusion_steps[:, -1].detach().numpy().squeeze())

# ## Reverse step by step

# +
diffusion_steps_reverse = [diffusion_steps_step_by_step[-1]]

for i in range(diffusion_params.steps - 1, -1, -1):
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

px.histogram(diffusion_steps_reverse[-1])

px.histogram(diffusion_steps_reverse[0])

# ## Diffusion Distributions

px.line(diffusion_params.alpha_bar_by_step)

# +
df_diffusion_example = pd.DataFrame(
    np.concatenate(
        [
            diffusion_initial_x.reshape(1, diffusion_batch_size).detach().numpy(),
            diffusion_steps.T.detach().numpy(),
        ]
    ),
    columns=[f"x_{i}" for i in range(len(diffusion_steps))],
)
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


from lightning import LightningModule


# +
@dataclasses.dataclass
class DiffusionEncoderParams:
    """Parameters for VAEEncoder and VAEDecoder

    :param latent_size: latent space dimension
    :param history_length: input sequence length
    :param n_features: number of features
    """

    latent_size: int = 40
    num_layers: int = 2
    history_length: int
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


class Encoder(nn.Module):
    """ """

    def __init__(self, params: DiffusionEncoderParams):
        super().__init__()

        self.params = params
        self.hparams = params.asdict()

        self.rnn = nn.GRU(
            input_size=self.params.history_length,
            hidden_size=self.params.latent_size,
            num_layers=self.params.num_layers,
        )

    def forward(
        self, x: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """

        :param x: input data, shape (batch_size, history_length, n_features)
        """
        batch_size, _, _ = x.size()
        x = x.transpose(1, 2)

        outputs, state = self.rnn(x, self.params.initial_state)

        return outputs, state


class Distribution(nn.Module):
    """ """

    def __init__(self, params: DiffusionEncoderParams, samples: int = 100):
        super().__init__()

        self.params = params
        self.hparams = params.asdict()
        self.samples = samples

        self.linear = nn.Linear(self.params.latent_size, samples)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.linear(x)


class DiffusionModel(nn.Module):
    """ """

    def __init__(
        self,
        encoder_params: DiffusionEncoderParams,
        diffusion_params: DiffusionParams,
        batch_size: int,
    ):
        super().__init__()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        outputs, state = self.encoder(x)
        distribution = self.distribution(outputs)

        return distribution


class Model(LightningModule):
    def __init__(
        self,
        encoder_params: DiffusionEncoderParams,
        diffusion_params: DiffusionParams,
        batch_size: int,
    ):
        self.encoder = Encoder(encoder_params)
        self.diffusion = Diffusion(batch_size, diffusion_params)
        self.distribution = Distribution(encoder_params, samples=100)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        outputs, state = self.encoder(x)
        distribution = self.distribution(outputs)

        return distribution

    def training_step(self, batch: torch.Tensor, batch_idx: int) -> torch.Tensor:
        x = batch["x"]
        y = batch["y"]

        distribution = self(x)

        loss = self.loss(distribution, y)

        return loss

    def loss(self, y: torch.Tensor, y_hat: torch.Tensor) -> torch.Tensor:

        return


# -

diffusion_params.steps

# +
enc_params = DiffusionEncoderParams(
    latent_size=10,
    num_layers=1,
    history_length=5,
    n_features=1,
    initial_state=None,
)
encoder = Encoder(
    params=enc_params,
)

encoder(
    x=torch.rand(8, 5, 1),
)[0].shape, encoder(
    x=torch.rand(8, 5, 1),
)[-1].shape
# -


# +
class DiffusionEmbedding(nn.Module):
    def __init__(self, dim, proj_dim, max_steps=500):
        super().__init__()
        self.register_buffer(
            "embedding", self._build_embedding(dim, max_steps), persistent=False
        )
        self.projection1 = nn.Linear(dim * 2, proj_dim)
        self.projection2 = nn.Linear(proj_dim, proj_dim)

    def forward(self, diffusion_step):
        x = self.embedding[diffusion_step]
        x = self.projection1(x)
        x = F.silu(x)
        x = self.projection2(x)
        x = F.silu(x)
        return x

    def _build_embedding(self, dim, max_steps):
        steps = torch.arange(max_steps).unsqueeze(1)  # [T,1]
        dims = torch.arange(dim).unsqueeze(0)  # [1,dim]
        table = steps * 10.0 ** (dims * 4.0 / dim)  # [T,dim]
        table = torch.cat([torch.sin(table), torch.cos(table)], dim=1)
        return table


class EpsilonTheta(nn.Module):
    def __init__(
        self,
        target_dim,
        cond_length,
        time_emb_dim=16,
        residual_layers=8,
        residual_channels=8,
        dilation_cycle_length=2,
        residual_hidden=64,
    ):
        super().__init__()
        self.input_projection = nn.Conv1d(
            1, residual_channels, 1, padding=2, padding_mode="circular"
        )
        self.diffusion_embedding = DiffusionEmbedding(
            time_emb_dim, proj_dim=residual_hidden
        )
        self.cond_upsampler = CondUpsampler(
            target_dim=target_dim, cond_length=cond_length
        )
        self.residual_layers = nn.ModuleList(
            [
                ResidualBlock(
                    residual_channels=residual_channels,
                    dilation=2 ** (i % dilation_cycle_length),
                    hidden_size=residual_hidden,
                )
                for i in range(residual_layers)
            ]
        )
        self.skip_projection = nn.Conv1d(residual_channels, residual_channels, 3)
        self.output_projection = nn.Conv1d(residual_channels, 1, 3)

        nn.init.kaiming_normal_(self.input_projection.weight)
        nn.init.kaiming_normal_(self.skip_projection.weight)
        nn.init.zeros_(self.output_projection.weight)

    def forward(self, inputs, time, cond):
        x = self.input_projection(inputs)
        x = F.leaky_relu(x, 0.4)

        diffusion_step = self.diffusion_embedding(time)
        cond_up = self.cond_upsampler(cond)
        skip = []
        for layer in self.residual_layers:
            x, skip_connection = layer(x, cond_up, diffusion_step)
            skip.append(skip_connection)

        x = torch.sum(torch.stack(skip), dim=0) / math.sqrt(len(self.residual_layers))
        x = self.skip_projection(x)
        x = F.leaky_relu(x, 0.4)
        x = self.output_projection(x)
        return x


# +
def extract(a, t, x_shape):
    b, *_ = t.shape
    out = a.gather(-1, t)
    mm = len(x_shape) - 1
    return out.reshape(b, *((1,) * mm))


def q_sample(x_start, t, noise=None):
    noise = default(noise, lambda: torch.randn_like(x_start))

    return (
        extract(sqrt_alphas_cumprod, t, x_start.shape) * x_start
        + extract(sqrt_one_minus_alphas_cumprod, t, x_start.shape) * noise
    )


def p_losses(x_start, cond, t, noise=None):
    loss_type = "l2"
    noise = default(noise, lambda: torch.randn_like(x_start))

    x_noisy = q_sample(x_start=x_start, t=t, noise=noise)
    x_recon = denoise_fn(x_noisy, t, cond=cond)

    if loss_type == "l1":
        loss = torch.nn.functional.l1_loss(x_recon, noise)
    elif loss_type == "l2":
        loss = torch.nn.functional.mse_loss(x_recon, noise)
    elif loss_type == "huber":
        loss = torch.nn.functional.smooth_l1_loss(x_recon, noise)
    else:
        raise NotImplementedError()

    return loss


def log_prob(self, x, cond, *args, **kwargs):
    if self.scale is not None:
        x /= self.scale

    B, T, _ = x.shape

    time = torch.randint(0, self.num_timesteps, (B * T,), device=x.device).long()
    loss = p_losses(
        x.reshape(B * T, 1, -1), cond.reshape(B * T, 1, -1), time, *args, **kwargs
    )

    return loss


# -

tmp_x_start = torch.ones((2, 3))
tmp_x_start

tmp_a = torch.Tensor([1, 2, 3, 4, 5])
tmp_a

extract(tmp_a, tmp_t, x_shape=tmp_x_start.shape)
