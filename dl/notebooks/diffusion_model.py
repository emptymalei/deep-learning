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
import torch

# +
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


class Diffusion(nn.Module):
    def __init__(self, batch_size: int, params: DiffusionParams):
        self.batch_size = batch_size
        self.params = params

    @cached_property
    def alpha_bar_by_step(self) -> torch.Tensor:
        return torch.tensor(self.params.alpha_bar_by_step, dtype=torch.float32)

    def likelihood(self):
        pass

    def _noise(self) -> torch.Tensor:
        return torch.randn(self.batch_size, 1)

    def _forward_process(self, initial: torch.Tensor, step: int) -> torch.Tensor:
        return (
            torch.sqrt(self.alpha_bar_by_step) * initial
            + torch.sqrt(1 - self.alpha_bar_by_step) * self._noise()
        )

    def forward(self):
        pass


# -

diffusion_params = DiffusionParams(5, 0.01)
diffusion_process = Diffusion(10, diffusion_params)


diffusion_initial_x = torch.sin(torch.ones(10, 1))
diffusion_process._forward_process(diffusion_initial_x, 0)


tmp_t = torch.Tensor([1]).long()
tmp_t


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
