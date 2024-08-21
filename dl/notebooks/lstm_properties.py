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

# # LSTM
#
# In this notebook, we demonstrate a few special properties of LSTM.

import matplotlib.pyplot as plt

# +
import numpy as np
import seaborn as sns

sns.set_theme()
import pandas as pd

# -

# ## Cell State Phase Space


def cell_state(f, m, c_prev: int = 1):
    return np.tanh(c_prev * f + m)


f = np.linspace(-1, 1, 101)
m = np.linspace(-1, 1, 101)

c = []
for f_i in f:
    c_m = []
    for m_i in m:
        c_m.append(cell_state(f_i, m_i))
    c.append(c_m)

df = pd.DataFrame(c, columns=[f"{i:.2f}" for i in m], index=[f"{i:.2f}" for i in f])

_, ax = plt.subplots(figsize=(8.5, 8))
ax = sns.heatmap(df, ax=ax)
ax.set_title("LSTM Cell States")
ax.set_xlabel(r"i * h")


# +
df = pd.DataFrame()
c_init = 5

for f in [0.1, 0.9]:
    for i_g in [-0.5, 0.5, 0.9]:
        cell_states = [
            {
                "iter": 0,
                "c": c_init,
                "f": f,
                "i*g": i_g,
                "c_prev": np.nan,
                "tanh_c": np.tanh(c_init),
            }
        ]

        for _ in range(10):
            f_ = cell_states[-1]["f"]
            c_ = cell_states[-1]["c"]
            i_g_ = cell_states[-1]["i*g"]
            c_new = cell_state(f=f_, m=i_g_, c_prev=c_)
            cell_states.append(
                {
                    "iter": cell_states[-1]["iter"] + 1,
                    "c": c_new,
                    "f": f_,
                    "i*g": i_g_,
                    "c_prev": c_,
                    "tanh_c": np.tanh(c_new),
                }
            )

        df = pd.concat([df, pd.DataFrame(cell_states)])
# -

df.head()

# +
fig, ax = plt.subplots(figsize=(10, 6.18))

sns.lineplot(
    df,
    x="iter",
    y="c",
    hue="f",
    style="i*g",
    size="f",
    palette=sns.palettes.color_palette("colorblind")[:2],
    markers=True,
    ax=ax,
)

ax.set_xlabel("Iteration")
ax.set_ylabel("c")

# +
fig, ax = plt.subplots(figsize=(10, 6.18))

sns.lineplot(
    df,
    x="iter",
    y="tanh_c",
    hue="f",
    style="i*g",
    size="f",
    palette=sns.palettes.color_palette("colorblind")[:2],
    markers=True,
    ax=ax,
)

ax.set_xlabel("Iteration")
ax.set_ylabel("tanh(c)")

# +
fig, ax = plt.subplots(figsize=(10, 6.18))

sns.scatterplot(
    df,
    x="c_prev",
    y="c",
    hue="f",
    style="i*g",
    size="f",
    palette=sns.palettes.color_palette("colorblind")[:2],
    markers=True,
    ax=ax,
)
ax.set_xlabel("c_prev")
ax.set_ylabel("c")
