# Forecast Reconciliation

This is a notebook for the section [Hierarchical Time Series Reconciliation](<https://dl.leima.is/time-series/timeseries-hierarchical.reconciliation/>).

import re

import matplotlib.pyplot as plt import numpy as np import pandas as pd import seaborn as sns

```
import sympy as sp
from darts import TimeSeries
from darts.utils.model_selection import train_test_split
from darts.utils.statistics import plot_pacf

sns.reset_orig()

plt.rcParams["figure.figsize"] = (10, 6.18)
print(plt.rcParams.get("figure.figsize"))
```

## Some MinT Matrics

This section shows a few examples of the MinT method. We use these examples to interpret how MinT works.

```
m_l = 3


m_w_diag_elements = tuple(sp.Symbol(f"W_{i}") for i in range(1, m_l + 1))

m_s_ident_diag = np.diag([1] * (m_l - 1)).tolist()

m_w_diag_elements, m_s_ident_diag
```

```
class MinTMatrices:
    def __init__(self, levels: int):
        self.levels = levels

    @property
    def s(self):
        s_ident_diag = np.diag([1] * (self.levels - 1)).tolist()
        return sp.Matrix(
            [
                [1] * (self.levels - 1),
            ]
            + s_ident_diag
        )

    @property
    def w_diag_elements(self):
        return tuple(sp.Symbol(f"W_{i}") for i in range(1, self.levels + 1))

    @property
    def w(self):
        return sp.Matrix(np.diag(self.w_diag_elements).tolist())

    @property
    def p_left(self):
        return sp.Inverse(sp.MatMul(sp.Transpose(self.s), sp.Inverse(self.w), self.s))

    @property
    def p_right(self):
        return sp.MatMul(sp.Transpose(self.s), sp.Inverse(self.w))

    @property
    def p(self):
        return sp.MatMul(self.p_left, self.p_right)

    @property
    def s_p(self):
        return sp.MatMul(self.s, self.p)

    @property
    def s_p_numerical(self):
        return sp.lambdify(self.w_diag_elements, self.s_p)

    def visualize_s_p(self, w_elements, ax):
        sns.heatmap(self.s_p_numerical(*w_elements), annot=True, cbar=False, ax=ax)
        ax.grid(False)
        ax.set(xticklabels=[], yticklabels=[])
        ax.tick_params(bottom=False, left=False)
        ax.set_title(f"$W_{{diag}} = {w_elements}$")
        return ax
```

```
mtm_3 = MinTMatrices(levels=3)
```

```
print(
    f"s: {sp.latex(mtm_3.s)}\n"
    f"p: {sp.latex(mtm_3.p.as_explicit())}\n"
    f"s_p: {sp.latex(mtm_3.s_p.as_explicit())}\n"
)
```

```
mtm_3.s
```

```
mtm_3.p
```

```
mtm_3.s_p.as_explicit()
```

```
mtm_3.w_diag_elements
```

```
mtm_3.s_p_numerical(1, 2, 3)
```

```
w_elements = [(1, 1, 1), (2, 1, 1)]

fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(4 * 2, 4))

for idx, w in enumerate(w_elements):
    mtm_3.visualize_s_p(w, axes[idx])
fig.show()
```

```
mtm_4 = MinTMatrices(levels=4)

print(
    f"s: {sp.latex(mtm_4.s)}\n"
    f"p: {sp.latex(mtm_4.p.as_explicit())}\n"
    f"s_p: {sp.latex(mtm_4.s_p.as_explicit())}\n"
)

w_elements = [(1, 1, 1, 1), (3, 1, 1, 1)]

fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(4 * 2, 4))

for idx, w in enumerate(w_elements):
    mtm_4.visualize_s_p(w, axes[idx])
fig.show()
```

```
mtm_5 = MinTMatrices(levels=5)

print(
    f"s: {sp.latex(mtm_5.s)}\n"
    f"p: {sp.latex(mtm_5.p.as_explicit())}\n"
    f"s_p: {sp.latex(mtm_5.s_p.as_explicit())}\n"
)

w_elements = [(1, 1, 1, 1, 1), (4, 1, 1, 1, 1)]

fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(4 * 2, 4))

for idx, w in enumerate(w_elements):
    mtm_5.visualize_s_p(w, axes[idx])
fig.show()
```

## Load data

We load a small sample of the M5 dataset.

```
df = pd.read_csv(
    "https://github.com/datumorphism/dataset-m5-simplified/raw/main/dataset/m5_store_sales.csv",
    index_col="date",
)
```

```
df["Total"] = df[["CA", "TX", "WI"]].sum(axis="columns")
```

```
df.index = pd.to_datetime(df.index)
```

```
re_simple_col = re.compile(r"'(\w{2}_\d{1})'")

df.rename(
    columns={
        c: re_simple_col.findall(c)[0] for c in df.columns if re_simple_col.findall(c)
    },
    inplace=True,
)
```

```
df.head()
```

```
value_columns = df.columns.tolist()
value_columns
```

```
hierarchy = {
    "CA_1": ["CA"],
    "CA_2": ["CA"],
    "CA_3": ["CA"],
    "CA_4": ["CA"],
    "TX_1": ["TX"],
    "TX_2": ["TX"],
    "TX_3": ["TX"],
    "WI_1": ["WI"],
    "WI_2": ["WI"],
    "WI_3": ["WI"],
    "CA": ["Total"],
    "TX": ["Total"],
    "WI": ["Total"],
}
```

```
ts = TimeSeries.from_dataframe(
    df, value_cols=value_columns, freq="d", hierarchy=hierarchy
)
```

```
ts
```

## Visualize and Validate the Data

```
ts_sample = ts.drop_after(ts.time_index[20])
```

```
ts_sample[["CA", "CA_1", "CA_2", "CA_3", "CA_4"]].plot()
```

```
ts_sample["CA"].plot(label="CA")
(ts_sample["CA_1"] + ts_sample["CA_2"] + ts_sample["CA_3"] + ts_sample["CA_4"]).plot(
    label="CA_1 + CA_2 + CA_3 + CA_4", linestyle="--", color="r"
)
```

## Forecasts

We split the dataset into two time series, `ts_train` and `ts_test`. We will hold out `ts_test` from training.

```
ts.time_index
```

```
ts_train, ts_test = ts.split_after(ts.time_index[1863])

ts_train["Total"].plot(label="Train")
ts_test["Total"].plot(label="Test")
```

We check the partial autocorrelation function to choose some parameters for our models.

```
plot_pacf(ts_train["Total"])
```

```
from darts.models import LightGBMModel
```

```
model_params = {"lags": 14, "linear_tree": True, "output_chunk_length": 10}
model = LightGBMModel(**model_params)
```

```
model.fit(ts_train)
```

```
model.save("lightgbm.pkl")
```

```
ts_pred = model.predict(n=len(ts_test))
```

We check the performance visually for CA. The patterns looks similar but the scales are a bit off.

```
ca_columns = ["CA", "CA_1", "CA_2", "CA_3", "CA_4"]
ts_test[ca_columns].plot()
ts_pred[ca_columns].plot(linestyle="--")
```

```
vis_columns = ["CA_4"]
ts_test[vis_columns].plot()
ts_pred[vis_columns].plot(linestyle="--")
```

The forecasts are not coherent.

```
ts_pred["Total"].plot(label="CA")
(ts_pred["CA"] + ts_pred["TX"] + ts_pred["WI"]).plot(
    label="CA + TX + WI", linestyle="--", color="r"
)
```

```
ts_pred["CA"].plot(label="CA")
(ts_pred["CA_1"] + ts_pred["CA_2"] + ts_pred["CA_3"] + ts_pred["CA_4"]).plot(
    label="CA_1 + CA_2 + CA_3 + CA_4", linestyle="--", color="r"
)
```

## Reconciliation

```
from darts.dataprocessing.transformers import MinTReconciliator
```

```
reconciliator = MinTReconciliator(method="wls_val")
```

```
reconciliator.fit(ts_train)
ts_pred_recon = reconciliator.transform(ts_pred)
```

```
ts_pred_recon["Total"].plot(label="CA")
(ts_pred_recon["CA"] + ts_pred_recon["TX"] + ts_pred_recon["WI"]).plot(
    label="CA + TX + WI", linestyle="--", color="r"
)
```

```
ts_pred_recon["CA"].plot(label="CA")
(
    ts_pred_recon["CA_1"]
    + ts_pred_recon["CA_2"]
    + ts_pred_recon["CA_3"]
    + ts_pred_recon["CA_4"]
).plot(label="CA_1 + CA_2 + CA_3 + CA_4", linestyle="--", color="r")
```

```
_, ax = plt.subplots(figsize=(10, 6.18))
ca_columns = ["CA", "CA_1", "CA_2", "CA_3", "CA_4"]
ts_test[ca_columns].plot(ax=ax)
ts_pred_recon[ca_columns].plot(linestyle="--", ax=ax)
```

### What Changed

```
ts_pred_recon_shift = ts_pred_recon - ts_pred
```

```
_, ax = plt.subplots(figsize=(10, 6.18))

ts_pred_recon_shift[["Total", "CA", "WI", "TX"]].plot(ax=ax)
```

```
_, ax = plt.subplots(figsize=(10, 6.18))
ts_pred_recon_shift[ca_columns + ["Total"]].plot(ax=ax)
```

To see how the predictions are shifted during reconciliation, we plot out the changes from reconciliation as box plots.

```
ts_pred_recon_shift[ca_columns + ["Total"]].to_dataframe().plot.box()
```

```
ts_pred_recon_shift["CA"].to_dataframe().plot.box()
```

```
ts_pred_recon_shift[["Total", "CA", "TX", "WI"]].to_dataframe().plot.box(
    title="Box Plot for Reconciled - Original Prediction"
)
```

```
ts_pred_recon_shift[["Total", "CA", "TX", "WI"]].to_dataframe()
```

```
max(ts_pred.values().max(), ts_pred_recon.values().max())
```

```
chart_component = "Total"
chart_max = max(
    ts_pred[chart_component].values().max(),
    ts_pred_recon[chart_component].values().max(),
)
chart_min = min(
    ts_pred[chart_component].values().min(),
    ts_pred_recon[chart_component].values().min(),
)

fig, ax = plt.subplots(figsize=(10, 10))


ax.scatter(ts_pred[chart_component].values(), ts_pred_recon[chart_component].values())

ax.plot(np.linspace(chart_min, chart_max), np.linspace(chart_min, chart_max))
```

```
chart_component = "CA"
chart_max = max(
    ts_pred[chart_component].values().max(),
    ts_pred_recon[chart_component].values().max(),
)
chart_min = min(
    ts_pred[chart_component].values().min(),
    ts_pred_recon[chart_component].values().min(),
)

fig, ax = plt.subplots(figsize=(10, 10))

ax.scatter(ts_pred[chart_component].values(), ts_pred_recon[chart_component].values())

ax.plot(np.linspace(chart_min, chart_max), np.linspace(chart_min, chart_max))
```

### Can Reconciliations Adjust Bias?

We create some artificial bias by shifting one of the series down and then perform reconciliations.

This assumes that the reconciliation already learned about the general patterns on different levels, since we only manually shift the predictions only. The training is not touched.

```
reconciliator_pred_bias = MinTReconciliator(method="wls_val")
```

```
df_pred_biased = ts_pred.to_dataframe().copy()
df_pred_biased["CA_1"] = df_pred_biased["CA_1"] * 0.5
ts_pred_biased = TimeSeries.from_dataframe(df_pred_biased, hierarchy=ts_pred.hierarchy)

ts_pred["CA_1"].plot(label="Original Prediction for CA_1")
ts_pred_biased["CA_1"].plot(label="Manually Shifted Prediction for CA_1")
```

```
reconciliator_pred_bias.fit(ts_pred_biased)
ts_pred_biased_recon = reconciliator_pred_bias.transform(ts_pred_biased)
```

```
ts_pred["CA_1"].plot(label="Original Prediction for CA_1")
ts_pred_biased["CA_1"].plot(label="Manually Shifted Prediction for CA_1")
ts_pred_biased_recon["CA_1"].plot(label="Reconciled Shifted Prediction for CA_1")
```

```
ts_pred_biased_recon["CA"].plot(label="CA")
(
    ts_pred_biased_recon["CA_1"]
    + ts_pred_biased_recon["CA_2"]
    + ts_pred_biased_recon["CA_3"]
    + ts_pred_biased_recon["CA_4"]
).plot(label="CA_1 + CA_2 + CA_3 + CA_4", linestyle="--", color="r")
```

```
_, ax = plt.subplots(figsize=(10, 6.18))
ca_columns = ["CA", "CA_1", "CA_2", "CA_3", "CA_4"]
ts_test[ca_columns].plot(ax=ax)
ts_pred_biased_recon[ca_columns].plot(linestyle="--", ax=ax)
```

```
reconciliator_mint_cov = MinTReconciliator(method="mint_cov")

reconciliator_mint_cov.fit(ts_pred - ts_test)

ts_test[ca_columns].plot()
reconciliator_mint_cov.transform(ts_pred)[ca_columns].plot(linestyle="--")
```
