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

# # Comparing Time Series with Each Other
#
# Time series data involves a time dimension, and it is not that intuitive to see the difference between two time series. In this notebook, We will show you how to compare time series with each other.

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

# +
from dtaidistance import dtw
from dtaidistance import dtw_visualisation as dtwvis

sns.set_theme()

plt.rcParams.update(
    {
        "font.size": 18,  # General font size
        "axes.titlesize": 20,  # Title font size
        "axes.labelsize": 16,  # Axis label font size
        "xtick.labelsize": 14,  # X-axis tick label font size
        "ytick.labelsize": 14,  # Y-axis tick label font size
        "legend.fontsize": 14,  # Legend font size
        "figure.titlesize": 20,  # Figure title font size
    }
)
# -

# # DTW
# To illustrate how DTW can be used to compare time series, we will use the following datasets:

t = np.arange(0, 20, 0.1)
ts_original = np.sin(t)

# We apply different transformations to the original time series.

ts_shifted = np.roll(ts_original, 10)
ts_jitter = ts_original + np.random.normal(0, 0.1, len(ts_original))
ts_flipped = ts_original[::-1]
ts_shortened = ts_original[::2]
ts_raise_level = ts_original + 0.5
ts_outlier = ts_original + np.append(np.zeros(len(ts_original) - 1), [10])

df = pd.DataFrame(
    {
        "t": t,
        "original": ts_original,
        "shifted": ts_shifted,
        "jitter": ts_jitter,
        "flipped": ts_flipped,
        "shortened": np.pad(
            ts_shortened, (0, len(ts_original) - len(ts_shortened)), constant_values=0
        ),
        "raise_level": ts_raise_level,
        "outlier": ts_outlier,
    }
)

# +
_, ax = plt.subplots()

for s in df.columns[1:]:
    sns.lineplot(df, x="t", y=s, ax=ax, label=s)


# +
distances = {
    "series": df.columns[1:],
}

for s in df.columns[1:]:
    distances["dtw"] = distances.get("dtw", []) + [dtw.distance(df.original, df[s])]
    distances["euclidean"] = distances.get("euclidean", []) + [
        np.linalg.norm(df.original - df[s])
    ]


_, ax = plt.subplots(figsize=(10, 6.18 * 2), nrows=2)

pd.DataFrame(distances).set_index("series").plot.bar(ax=ax[0])

colors = sns.color_palette("husl", len(distances["series"]))
pd.DataFrame(distances).plot.scatter(x="dtw", y="euclidean", ax=ax[1], c=colors, s=100)

for i, txt in enumerate(distances["series"]):
    ax[1].annotate(txt, (distances["dtw"][i], distances["euclidean"][i]), fontsize=12)

ax[1].legend(distances["series"], loc="best")


# -


def dtw_map(s1, s2, window=None):
    if window is None:
        window = len(s1)
    d, paths = dtw.warping_paths(s1, s2, window=window, psi=2)
    best_path = dtw.best_path(paths)

    return dtwvis.plot_warpingpaths(s1, s2, paths, best_path)


dtw_map(df.original, df.jitter)

for s in df.columns[1:]:
    fig, ax = dtw_map(df.original, df[s])
    fig.suptitle(s, y=1.05)


# # Dimension Reduction
#
# We embed the original time series into a time-delayed embedding space, then reduce the dimensionality of the embedded time series for visualizations.


def time_delay_embed(df: pd.DataFrame, window_size: int) -> pd.DataFrame:
    """embed time series into a time delay embedding space

    Time column `t` is required in the input data frame.

    :param df: original time series data frame
    :param window_size: window size for the time delay embedding
    """
    dfs_embedded = []

    for i in df.rolling(window_size):
        i_t = i.t.iloc[0]
        dfs_embedded.append(
            pd.DataFrame(i.reset_index(drop=True))
            .drop(columns=["t"])
            .T.reset_index()
            .rename(columns={"index": "name"})
            .assign(t=i_t)
        )

    df_embedded = pd.concat(dfs_embedded[window_size - 1 :])

    return df_embedded


df_embedded_2 = time_delay_embed(df, window_size=2)

# +
_, ax = plt.subplots()

(
    df_embedded_2.loc[df_embedded_2.name == "original"].plot.line(
        x=0, y=1, ax=ax, legend=False
    )
)
(
    df_embedded_2.loc[df_embedded_2.name == "original"].plot.scatter(
        x=0, y=1, c="t", colormap="viridis", ax=ax
    )
)
# -

# We choose a higher window size to track longer time dependency and for the dimension reduction methods to function properly.

df_embedded = time_delay_embed(df, window_size=5)

# ## PCA

from sklearn.decomposition import PCA

pca = PCA(n_components=2)

df_embedded_pca = pd.concat(
    [
        pd.DataFrame(
            pca.fit_transform(
                df_embedded.loc[df_embedded.name == n].drop(columns=["name", "t"])
            ),
            columns=["pca_0", "pca_1"],
        ).assign(name=n)
        for n in df_embedded.name.unique()
    ]
)

sns.scatterplot(data=df_embedded_pca, x="pca_0", y="pca_1", hue="name")

# +
_, ax = plt.subplots()

sns.scatterplot(
    data=df_embedded_pca.loc[
        df_embedded_pca.name.isin(
            ["original", "jitter", "flipped", "raise_level", "shifted", "shortened"]
        )
    ],
    x="pca_0",
    y="pca_1",
    hue="name",
    style="name",
    ax=ax,
)

ax.legend(loc="lower center", bbox_to_anchor=(0.5, -0.4), ncol=3)
# -


# ## t-SNE

from sklearn.manifold import TSNE

t_sne = TSNE(n_components=2, learning_rate="auto", init="random", perplexity=3)

df_embedded.name.unique()

df_embedded_tsne = pd.concat(
    [
        pd.DataFrame(
            t_sne.fit_transform(
                df_embedded.loc[df_embedded.name == n].drop(columns=["name", "t"])
            ),
            columns=["tsne_0", "tsne_1"],
        ).assign(name=n)
        for n in df_embedded.name.unique()
    ]
)

df_embedded_tsne.loc[df_embedded_tsne.name == "original"]

sns.scatterplot(data=df_embedded_tsne, x="tsne_0", y="tsne_1", hue="name")

sns.scatterplot(
    data=df_embedded_tsne.loc[
        df_embedded_tsne.name.isin(["original", "jitter", "outlier"])
    ],
    x="tsne_0",
    y="tsne_1",
    hue="name",
)
