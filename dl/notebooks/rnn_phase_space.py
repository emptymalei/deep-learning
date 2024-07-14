# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: light
#       format_version: '1.5'
#       jupytext_version: 1.15.2
#   kernelspec:
#     display_name: deep-learning
#     language: python
#     name: python3
# ---

# # RNN Dynamics During Inference
#
# We have the intuition that RNN inference is similar to a first order differential equation. Here we explore and echo on this idea using numerical simulations.

from functools import cached_property

import matplotlib.pyplot as plt

# +
import numpy as np
import numpy.typing as npt
import pandas as pd
import seaborn as sns

# -

# ## RNN


class RNNState:
    """
    Describes a RNN state and computes the history of the states
    based on inputs.

    :param w_h: $W_h$, the weight for $h$
    :param w_i: $W_i$, the weight for $x$
    :param b: $b$, the bias before applying activation
    :param h_init: the initial hidden state
    :param activation: the activation function to be applied.
    """

    def __init__(
        self,
        w_h: npt.ArrayLike,
        w_i: npt.ArrayLike,
        b: npt.ArrayLike,
        h_init: npt.ArrayLike,
        activation: callable = np.tanh,
    ):
        self.w_h = np.array(w_h)
        self.w_i = np.array(w_i)
        self.b = np.array(b)
        self.h_init = np.array(h_init)
        self.activation = activation

    def compute_new_state(
        self, h_current: npt.ArrayLike, z_i: npt.ArrayLike
    ) -> tuple[npt.ArrayLike, npt.ArrayLike]:
        """
        compute_new_state computes a new state

        :param h_current: the current hidden state
        :param z_i: external input
        :return: the new hidden state, and the difference between new and old
        """
        h_new = self.activation(
            np.dot(self.w_h, h_current) + np.dot(self.w_i, z_i) + self.b
        )
        h_delta = h_new - h_current

        return h_new, h_delta

    @cached_property
    def _metadata(self) -> dict:
        return {
            "experiment": (
                f"w_h={np.array2string(self.w_h)};w_i={np.array2string(self.w_i)};"
                f"b={np.array2string(self.b)};h_init={np.array2string(self.h_init)};{self.activation.__name__}"
            ),
            "w_h": np.array2string(self.w_h),
            "w_i": np.array2string(self.w_i),
            "b": np.array2string(self.b),
            "activation": self.activation.__name__,
            "initial_state": np.array2string(self.h_init),
        }

    def states(self, z: npt.ArrayLike) -> dict[str : npt.ArrayLike]:
        """
        states calculates the history of RNN states.

        We designed function to be easily readable by
        computing the values step by step.

        :param z: input values for the RNN
        :return: history of the RNN hidden states
        """
        h_t = [self.h_init]
        t_steps = [0]
        h_t_delta = [np.zeros_like(self.h_init)]
        for t, z_i in enumerate(z):
            h_new, h_delta = self.compute_new_state(h_current=h_t[-1], z_i=z_i)
            h_t.append(h_new)
            h_t_delta.append(h_delta)
            t_steps.append(t + 1)

        total_time_steps = len(t_steps)

        return {
            **{
                "t": np.array(t_steps),
                "h": np.array(h_t),
                "dh": np.array(h_t_delta),
            },
            **{k: [v] * total_time_steps for k, v in self._metadata.items()},
        }


# ## One Dimensional State

z_1 = np.linspace(0, 10, 101)
# z_1 = np.ones(10) * 0.5
# z_1 = np.random.rand(100)

# +
experiment_params = [
    {"w_h": 0.5, "w_i": 0.5, "b": 0, "h_init": 0.5},
    {"w_h": 1.5, "w_i": 1.5, "b": 0, "h_init": 0.5},
    {"w_h": 0.5, "w_i": 0.5, "b": 0, "h_init": 1.5},
    {"w_h": 1.5, "w_i": 1.5, "b": 0, "h_init": 1.5},
]

df_1_experiments = pd.DataFrame()
for p in experiment_params:
    df_1_experiments = pd.concat(
        [df_1_experiments, pd.DataFrame(RNNState(**p).states(z=z_1))]
    )

# +
_, ax = plt.subplots(figsize=(10, 6.18))

sns.lineplot(df_1_experiments, x="t", y="h", hue="experiment")
# -

# ### Two dimensional state

# z_2 = np.random.rand(100, 2) * 0.1
# z_2 = np.ones((100, 2))
# z_2 = np.zeros((1000, 2))
z_2 = (
    np.random.rand(100, 2) * 0.1 + np.stack([np.zeros(100), np.linspace(0, 99, 100)]).T
)

# +
experiment_2d_params = [
    {
        "w_h": np.array([[0.5, 0.5], [0.5, 0.5]]),
        "w_i": np.array([[0.5, 0.5], [0.5, 0.5]]),
        "b": np.array([0, 0]),
        "h_init": np.array([0.5, 0.5]),
    },
    {
        "w_h": np.array([[1.5, 0.5], [0.5, 1.5]]),
        "w_i": np.array([[1.5, 0.5], [0.5, 1.5]]),
        "b": np.array([0, 0]),
        "h_init": np.array([0.5, 0.5]),
    },
]

experiments_2d = []
for p in experiment_2d_params:
    experiments_2d.append(RNNState(**p).states(z=z_2))

# +
_, ax = plt.subplots(figsize=(10, 6.18))

experiments_2d_colors = sns.color_palette("husl", len(experiments_2d))

for idx, i in enumerate(experiments_2d):
    ax.plot(
        i["t"][2:],
        i["h"][2:, 1],
        marker=".",
        color=experiments_2d_colors[idx],
        label=i["experiment"][0],
    )
    ax.plot(
        i["t"][2:],
        i["h"][2:, 0],
        marker="x",
        color=experiments_2d_colors[idx],
        label=i["experiment"][0],
    )

plt.legend()
# -
