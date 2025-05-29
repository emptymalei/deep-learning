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

# # TabPFN as Forecaster

import pandas as pd
from ts_dl_utils.datasets.pendulum import Pendulum

pen = Pendulum(length=100)

df = pd.DataFrame(pen(300, 30, initial_angle=1, beta=0.00001))


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
            .T.reset_index(drop=True)
            # .rename(columns={"index": "name"})
            # .assign(t=i_t)
        )

    df_embedded = pd.concat(dfs_embedded[window_size - 1 :])

    return df_embedded


df = time_delay_embed(df, 11)
