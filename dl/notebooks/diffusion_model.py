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

import dataclasses
from functools import cached_property

import numpy as np
import pandas as pd
import plotly.express as px
import torch
import torch.nn as nn


# +
@dataclasses.dataclass
class DiffusionEncoderParams:
    """Parameters for VAEEncoder and VAEDecoder

    :param latent_size: latent space dimension
    :param history_length: input sequence length
    :param n_features: number of features
    """

    latent_size: int
    num_layers: int
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
        batch_size, _, _ = x.size()
        x = x.transpose(1, 2)
        x = self.encode(x)

        outputs, state = self.rnn(x, self.params.initial_state)

        return outputs, state


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
)
diffusion_batch_size = 1000
diffusion_process = Diffusion(diffusion_batch_size, diffusion_params)


# diffusion_initial_x = torch.sin(
#     torch.linspace(0, 10, 50)
#     .reshape(50, 1)
# )
diffusion_initial_x = (
    torch.distributions.Beta(torch.tensor([0.5]), torch.tensor([0.5]))
    .sample((diffusion_batch_size, 1))
    .reshape(diffusion_batch_size)
)


# ## Forward process step by step

# +
diffusion_steps_step_by_step = []

diffusion_steps_sbs_state = diffusion_initial_x
for i in range(0, diffusion_params.steps):
    i_state = diffusion_process._forward_process_by_step(
        diffusion_steps_sbs_state, step=i
    )
    diffusion_steps_step_by_step.append(i_state.detach().numpy())
    diffusion_steps_sbs_state = i_state
# -

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
diffusion_steps_reverse = []

diffusion_steps_reverse_state = diffusion_steps[:, -1]
for i in range(diffusion_params.steps - 1, 0, -1):
    i_state = diffusion_process._inverse_process_by_step(
        diffusion_steps_reverse_state, step=i
    )
    diffusion_steps_reverse.append(i_state.detach().numpy())
    diffusion_steps_reverse_state = i_state
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


def extract(a, t, x_shape):
    b, *_ = t.shape
    out = a.gather(-1, t)
    mm = len(x_shape) - 1
    return out.reshape(b, *((1,) * mm))


tmp_x_start = torch.ones((2, 3))
tmp_x_start

tmp_a = torch.Tensor([1, 2, 3, 4, 5])
tmp_a

extract(tmp_a, tmp_t, x_shape=tmp_x_start.shape)
