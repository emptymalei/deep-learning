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

# # Time Series Data and Embeddings

import matplotlib.pyplot as plt
import numpy as np

# +
import pandas as pd
import plotly.express as px

# -


def plot_arrow_chart(
    dataframe: pd.DataFrame,
    x_col: str,
    y_col: str,
    ax: plt.Axes,
    color: str = "k",
    alpha: float = 0.7,
    marker: str = ".",
    linestyle: str = "-",
    arrow_head_width: int = 4000,
) -> plt.Axes:
    """
    Plot an arrow chart for the 'Total' and its lagged values
    within a specified date range.
    """

    x = dataframe[x_col].values
    y = dataframe[y_col].values

    ax.plot(x, y, marker=marker, linestyle=linestyle, color=color, alpha=alpha)

    step = max(1, len(x) // 100)
    for i in range(0, len(x) - 1, step):
        ax.arrow(
            x[i],
            y[i],
            x[i + 1] - x[i],
            y[i + 1] - y[i],
            shape="full",
            lw=0,
            length_includes_head=True,
            head_width=arrow_head_width,
            color=color,
            alpha=alpha,
        )

    return ax


# ## Pendulum

from ts_dl_utils.datasets.pendulum import Pendulum

# +
pen = Pendulum(length=20)

df_pen = pd.DataFrame(pen(3, 100, initial_angle=1, beta=0.01))

df_pen["theta_1"] = df_pen["theta"].shift()
df_pen["theta_diff"] = df_pen["theta"].diff()

df_pen

# +
fig = plt.figure(figsize=(10, 8), layout="constrained")
spec = fig.add_gridspec(2, 2)

ax0 = fig.add_subplot(spec[0, :])
ax10 = fig.add_subplot(spec[1, 0])
ax11 = fig.add_subplot(spec[1, 1])

ax0.plot(
    df_pen.t,
    df_pen.theta,
    marker=".",
    linestyle="-",
    color="k",
)
# Make x-ticks readable
ax0.xaxis.set_major_locator(plt.MaxNLocator(8))
# fig.autofmt_xdate(rotation=30)
ax0.set_title("Swing Angle")


ax10 = plot_arrow_chart(
    df_pen, x_col="theta", y_col="theta_1", ax=ax10, arrow_head_width=0.00001
)
ax10.set_xlabel("Swing Angle")
ax10.set_ylabel("Swing Angle 0.05 seconds ago")
ax10.set_title("Swing Angle and Angle 0.05 seconds ago")


ax11 = plot_arrow_chart(
    df_pen, x_col="theta", y_col="theta_diff", ax=ax11, arrow_head_width=0.00001
)

ax11.set_xlabel("Swing Angle")
ax11.set_ylabel("Swing Angle Change Rate")
ax11.set_title("Phase Portrait")

plt.tight_layout()
# -

# ## Covid

df_ecdc_covid = pd.read_csv(
    "https://gist.githubusercontent.com/emptymalei/"
    "90869e811b4aa118a7d28a5944587a64/raw"
    "/1534670c8a3859ab3a6ae8e9ead6795248a3e664"
    "/ecdc%2520covid%252019%2520data"
)

px.line(df_ecdc_covid, x="datetime", y="Total")

df_ecdc_covid


# +
df_ecdc_covid_arrow_chart = df_ecdc_covid.loc[
    pd.to_datetime(df_ecdc_covid.datetime).between("2020-08-01", "2020-12-01")
].copy()

df_ecdc_covid_arrow_chart["Total_1"] = df_ecdc_covid_arrow_chart["Total"].shift()
df_ecdc_covid_arrow_chart["Total_diff"] = df_ecdc_covid_arrow_chart["Total"].diff()

# +
fig = plt.figure(figsize=(10, 8), layout="constrained")
spec = fig.add_gridspec(2, 2)

ax0 = fig.add_subplot(spec[0, :])
ax10 = fig.add_subplot(spec[1, 0])
ax11 = fig.add_subplot(spec[1, 1])

ax0.plot(
    df_ecdc_covid_arrow_chart.datetime,
    df_ecdc_covid_arrow_chart.Total,
    marker=".",
    linestyle="-",
    color="k",
)
# Make x-ticks readable
ax0.xaxis.set_major_locator(plt.MaxNLocator(8))
# fig.autofmt_xdate(rotation=30)
ax0.set_title("Covid Cases in EU Over Time")


ax10 = plot_arrow_chart(
    df_ecdc_covid_arrow_chart, x_col="Total", y_col="Total_1", ax=ax10
)
ax10.set_xlabel("Total Cases")
ax10.set_ylabel("Total Cases Lagged by 1 Day")
ax10.set_title("Covid Cases and Lagged Values")


ax11 = plot_arrow_chart(
    df_ecdc_covid_arrow_chart, x_col="Total", y_col="Total_diff", ax=ax11
)

ax11.set_xlabel("Total Cases")
ax11.set_ylabel("Total Cases Change")
ax11.set_title("Covid Cases in EU Phase Portrait")
ax11.set_ylim(-100_000, 100_000)

plt.tight_layout()
# -

# ## Walmart

df_walmart = pd.read_csv(
    "https://raw.githubusercontent.com/datumorphism/"
    "dataset-m5-simplified/refs/heads/main/dataset/"
    "m5_store_sales.csv"
)

df_walmart

px.line(df_walmart, x="date", y="CA")

df_walmart_total = df_walmart[["date", "CA", "TX", "WI"]].copy()

# +
df_walmart_total["total"] = (
    df_walmart_total.CA + df_walmart_total.TX + df_walmart_total.WI
)

df_walmart_total["datetime"] = pd.to_datetime(df_walmart_total.date, format="%Y-%m-%d")
df_walmart_total["timestamp"] = df_walmart_total.datetime.astype(int) // 10**9
# -

df_walmart_total["total_1"] = df_walmart_total.total.shift()
df_walmart_total["total_diff"] = df_walmart_total.total.diff()

px.scatter(
    df_walmart_total.loc[pd.to_datetime(df_walmart_total.date).dt.year == 2016],
    x="total",
    y="total_1",
    color="timestamp",
)

df_walmart_arrow_chart = df_walmart_total.loc[
    pd.to_datetime(df_walmart_total.date).between("2016-01-01", "2016-03-01")
].copy()


# +
fig = plt.figure(figsize=(10, 8), layout="constrained")
spec = fig.add_gridspec(2, 2)

ax0 = fig.add_subplot(spec[0, :])
ax10 = fig.add_subplot(spec[1, 0])
ax11 = fig.add_subplot(spec[1, 1])

ax0.plot(
    df_walmart_arrow_chart.datetime,
    df_walmart_arrow_chart.total,
    marker=".",
    linestyle="-",
    color="k",
)
# Make x-ticks readable
ax0.xaxis.set_major_locator(plt.MaxNLocator(8))
# fig.autofmt_xdate(rotation=30)
ax0.set_title("Walmart Sales Over Time")

ax10 = plot_arrow_chart(
    df_walmart_arrow_chart,
    x_col="total",
    y_col="total_1",
    ax=ax10,
    arrow_head_width=500,
)
ax10.set_xlabel("Total Sales")
ax10.set_ylabel("Total Sales Lagged by 1 Day")
ax10.set_title("Walmart Sales and Lagged Sales")

ax11 = plot_arrow_chart(
    df_walmart_arrow_chart,
    x_col="total",
    y_col="total_diff",
    ax=ax11,
    arrow_head_width=500,
)
ax11.set_xlabel("Total Sales")
ax11.set_ylabel("Total Sales Change")
ax11.set_title("Walmart Sales Phase Portrait")


plt.tight_layout()
# -

#


# ## Electricity Data

import io
import zipfile

import pandas as pd

# +
import requests

# Download from remote URL
data_uri = "https://archive.ics.uci.edu/ml/machine-learning-databases/00321/LD2011_2014.txt.zip"

r = requests.get(data_uri)

z = zipfile.ZipFile(io.BytesIO(r.content))
z.extractall("tmp/data/uci_electricity/")

# -

# Load as pandas dataframe
df_electricity = (
    pd.read_csv("tmp/data/uci_electricity/LD2011_2014.txt", delimiter=";", decimal=",")
    .rename(columns={"Unnamed: 0": "date"})
    .set_index("date")
)
df_electricity.index = pd.to_datetime(df_electricity.index)

df_electricity

df_electricity.loc[
    (df_electricity.index >= "2012-01-01") & (df_electricity.index < "2012-02-01")
][["MT_001"]].plot()
