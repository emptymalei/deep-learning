# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py:light
#     text_representation:
#       extension: .py
#       format_name: light
#       format_version: '1.5'
#       jupytext_version: 1.14.5
#   kernelspec:
#     display_name: Python 3
#     language: python
#     name: python3
# ---

# + [markdown] id="0gswx1yK7dyv"
# # Forecast Reconciliation
#
# This is a notebook for the section [Hierarchical Time Series Reconciliation](https://dl.leima.is/time-series/timeseries-hierarchical.reconciliation/).

# + colab={"base_uri": "https://localhost:8080/"} id="Dv8Ua7IlbgcP" outputId="e1d62a51-78c6-4bfe-e7bf-717771baaf9c"
import sympy as sp
import numpy as np
import pandas as pd
from darts import TimeSeries
from darts.utils.model_selection import train_test_split
from darts.utils.statistics import plot_pacf

import re

import matplotlib.pyplot as plt
import seaborn as sns

sns.reset_orig()

plt.rcParams["figure.figsize"] = (10, 6.18)
print(plt.rcParams.get("figure.figsize"))

# + [markdown] id="LBgAdOhRtqCP"
# ## Some MinT Matrics
#
# This section shows a few examples of the MinT method. We use these examples to interpret how MinT works.
#

# + colab={"base_uri": "https://localhost:8080/"} id="Vawe4deuHWl4" outputId="153c5f7b-c527-4d54-811d-722fe373115e"
m_l = 3


m_w_diag_elements = tuple(sp.Symbol(f"W_{i}") for i in range(1, m_l + 1))

m_s_ident_diag = np.diag([1] * (m_l - 1)).tolist()

m_w_diag_elements, m_s_ident_diag


# + id="9K22Nsl4PcjE"
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


# + colab={"base_uri": "https://localhost:8080/"} id="CbBICSdpSDco" outputId="517ade29-459f-4c06-b6b8-0a6ea8096eea"
mtm_3 = MinTMatrices(levels=3)

# + colab={"base_uri": "https://localhost:8080/"} id="knQo9b6jIYBt" outputId="500be86a-cfe9-4c3e-8ccc-c366c49d0bc3"
print(
    f"s: {sp.latex(mtm_3.s)}\n"
    f"p: {sp.latex(mtm_3.p.as_explicit())}\n"
    f"s_p: {sp.latex(mtm_3.s_p.as_explicit())}\n"
)

# + colab={"base_uri": "https://localhost:8080/", "height": 78} id="ELydWTiOSDZa" outputId="928b77d2-e036-456c-9ebd-dc69c3d032ac"
mtm_3.s

# + colab={"base_uri": "https://localhost:8080/", "height": 88} id="DAutkpt9C6Uy" outputId="eca64d84-2a0c-4857-c6b5-ce56d4833d93"
mtm_3.p

# + colab={"base_uri": "https://localhost:8080/", "height": 105} id="UbjQzTIpSC7o" outputId="ec499f24-bc9b-4dbd-90c4-071d530f6b21"
mtm_3.s_p.as_explicit()

# + colab={"base_uri": "https://localhost:8080/"} id="xQmxUk_tTOMs" outputId="36dc7a9b-8615-4d78-e4a5-c7facc017c9e"
mtm_3.w_diag_elements

# + colab={"base_uri": "https://localhost:8080/"} id="6I2OdhSbTFDT" outputId="88f315c9-3690-462c-84a5-7e7a5cdf35b2"
mtm_3.s_p_numerical(1, 2, 3)

# + colab={"base_uri": "https://localhost:8080/", "height": 369} id="MQ7Vl7TDSeL9" outputId="9a22ae9a-61bb-4bfc-96b7-4cc0f3e3675c"
w_elements = [(1, 1, 1), (2, 1, 1)]

fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(4 * 2, 4))

for idx, w in enumerate(w_elements):
    mtm_3.visualize_s_p(w, axes[idx])
fig.show()

# + colab={"base_uri": "https://localhost:8080/", "height": 460} id="6TRf6m5eT5v0" outputId="c2d33c1e-f1b0-46fb-8a44-62ba8019f887"
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

# + colab={"base_uri": "https://localhost:8080/", "height": 460} id="kqBylSVST_iP" outputId="3b52dddc-562f-4ffd-c12f-f91714a27a47"
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

# + id="nmCgX4OWAud0"


# + [markdown] id="C416Q_Zn7sJD"
# ## Load data
#
# We load a small sample of the M5 dataset.

# + id="LgEv3C1JbYjb"
df = pd.read_csv(
    "https://github.com/datumorphism/dataset-m5-simplified/raw/main/dataset/m5_store_sales.csv",
    index_col="date",
)

# + id="U_fbko_bfY_p"
df["Total"] = df[["CA", "TX", "WI"]].sum(axis="columns")

# + id="RX9PvvsUf_YI"
df.index = pd.to_datetime(df.index)

# + id="IVP5GyJOg5gR"
re_simple_col = re.compile(r"'(\w{2}_\d{1})'")

df.rename(
    columns={
        c: re_simple_col.findall(c)[0] for c in df.columns if re_simple_col.findall(c)
    },
    inplace=True,
)

# + colab={"base_uri": "https://localhost:8080/", "height": 238} id="CxFNDVATbi9Z" outputId="cc440175-d741-4500-970f-002216155401"
df.head()

# + colab={"base_uri": "https://localhost:8080/"} id="XCcHkKJYeG4M" outputId="aefe51f9-30f0-4c72-dae5-6502311b0f8a"
value_columns = df.columns.tolist()
value_columns

# + id="CEeUXDn-ebOp"
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

# + id="8zSeA2XQblTL"
ts = TimeSeries.from_dataframe(
    df, value_cols=value_columns, freq="d", hierarchy=hierarchy
)

# + colab={"base_uri": "https://localhost:8080/", "height": 959} id="DXisSRHJgQ3L" outputId="c8bb7f79-6e68-4d55-9ab3-e6102d887a96"
ts

# + [markdown] id="WvZHcU7LnKIO"
# ## Visualize and Validate the Data

# + id="Sz_beaxTixuT"
ts_sample = ts.drop_after(ts.time_index[20])

# + colab={"base_uri": "https://localhost:8080/", "height": 587} id="yODotQpFgYfV" outputId="bb9665b5-da86-4d41-f896-33c137108dff"
ts_sample[["CA", "CA_1", "CA_2", "CA_3", "CA_4"]].plot()

# + colab={"base_uri": "https://localhost:8080/", "height": 587} id="Ncbl-Ob3ghKJ" outputId="526c46b8-fa7f-4d72-e0a5-e1234994155a"
ts_sample["CA"].plot(label="CA")
(ts_sample["CA_1"] + ts_sample["CA_2"] + ts_sample["CA_3"] + ts_sample["CA_4"]).plot(
    label="CA_1 + CA_2 + CA_3 + CA_4", linestyle="--", color="r"
)

# + [markdown] id="Ql4OudBkpmzb"
# ## Forecasts
#
# We split the dataset into two time series, `ts_train` and `ts_test`. We will hold out `ts_test` from training.

# + colab={"base_uri": "https://localhost:8080/"} id="AN-z3afoqDHw" outputId="c54ee3a8-7284-4fd6-8ac8-b8a1494ea07c"
ts.time_index

# + colab={"base_uri": "https://localhost:8080/", "height": 572} id="LxG_m9SBihFS" outputId="79aca2d2-1033-4483-ecac-31648b1ca06f"
ts_train, ts_test = ts.split_after(ts.time_index[1863])

ts_train["Total"].plot(label="Train")
ts_test["Total"].plot(label="Test")

# + [markdown] id="3pLwEgJO9QPO"
# We check the partial autocorrelation function to choose some parameters for our models.

# + colab={"base_uri": "https://localhost:8080/", "height": 444} id="HHNtFTQjrxd-" outputId="876ffec5-8f88-47ac-a43c-59d617cf7574"
plot_pacf(ts_train["Total"])

# + id="UPnPyAZPqHqs"
from darts.models import LightGBMModel

# + id="g1ND5Td7qiLu"
model_params = {"lags": 14, "linear_tree": True, "output_chunk_length": 10}
model = LightGBMModel(**model_params)

# + colab={"base_uri": "https://localhost:8080/"} id="MUB4EkKxq-4R" outputId="54ce4c7b-afe1-49b0-d7b5-5426094bc600"
model.fit(ts_train)

# + id="jyLNtuDK_0cQ"
model.save("lightgbm.pkl")

# + id="svsFNqyTrBQV"
ts_pred = model.predict(n=len(ts_test))

# + [markdown] id="lJFYh0OyuDtT"
# We check the performance visually for CA. The patterns looks similar but the scales are a bit off.

# + colab={"base_uri": "https://localhost:8080/", "height": 587} id="pBq9Khb-rUDq" outputId="7977c345-a5cc-45c0-f853-8d1ccd9eed00"
ca_columns = ["CA", "CA_1", "CA_2", "CA_3", "CA_4"]
ts_test[ca_columns].plot()
ts_pred[ca_columns].plot(linestyle="--")

# + colab={"base_uri": "https://localhost:8080/", "height": 587} id="s7OlaDrdrYwF" outputId="cac337b3-4d85-46e8-ceb9-30e516e5566a"
vis_columns = ["CA_4"]
ts_test[vis_columns].plot()
ts_pred[vis_columns].plot(linestyle="--")

# + [markdown] id="wlcPf6dNuGot"
# The forecasts are not coherent.

# + colab={"base_uri": "https://localhost:8080/", "height": 587} id="B8-NK55rt4ti" outputId="b2d1a97c-b1a9-4276-ce97-ed3614f74ba4"
ts_pred["Total"].plot(label="CA")
(ts_pred["CA"] + ts_pred["TX"] + ts_pred["WI"]).plot(
    label="CA + TX + WI", linestyle="--", color="r"
)

# + colab={"base_uri": "https://localhost:8080/", "height": 587} id="8G3_Wwbusgc0" outputId="e081adbf-13ac-41ee-fcd1-03eb1741c3d8"
ts_pred["CA"].plot(label="CA")
(ts_pred["CA_1"] + ts_pred["CA_2"] + ts_pred["CA_3"] + ts_pred["CA_4"]).plot(
    label="CA_1 + CA_2 + CA_3 + CA_4", linestyle="--", color="r"
)

# + [markdown] id="IZLhFftBuK_b"
# ## Reconciliation

# + id="W4Bx_UILt1kY"
from darts.dataprocessing.transformers import MinTReconciliator

# + id="9XrhIBIYuwEo"
reconciliator = MinTReconciliator(method="wls_val")

# + id="Twuq3MNuu2QE"
reconciliator.fit(ts_train)
ts_pred_recon = reconciliator.transform(ts_pred)

# + colab={"base_uri": "https://localhost:8080/", "height": 589} id="0c5ME7tQu_M7" outputId="393b199c-feeb-45e4-8c03-8e406c3f9f71"
ts_pred_recon["Total"].plot(label="CA")
(ts_pred_recon["CA"] + ts_pred_recon["TX"] + ts_pred_recon["WI"]).plot(
    label="CA + TX + WI", linestyle="--", color="r"
)

# + colab={"base_uri": "https://localhost:8080/", "height": 587} id="OmAzytxsu_8Z" outputId="f79657a6-a083-4f2b-ab53-45479a9d815c"
ts_pred_recon["CA"].plot(label="CA")
(
    ts_pred_recon["CA_1"]
    + ts_pred_recon["CA_2"]
    + ts_pred_recon["CA_3"]
    + ts_pred_recon["CA_4"]
).plot(label="CA_1 + CA_2 + CA_3 + CA_4", linestyle="--", color="r")

# + colab={"base_uri": "https://localhost:8080/", "height": 587} id="m42Tlou0yLRy" outputId="86f26605-68b5-4245-e8f7-029a63c76805"
_, ax = plt.subplots(figsize=(10, 6.18))
ca_columns = ["CA", "CA_1", "CA_2", "CA_3", "CA_4"]
ts_test[ca_columns].plot(ax=ax)
ts_pred_recon[ca_columns].plot(linestyle="--", ax=ax)

# + [markdown] id="2xDnmS4QvS9i"
# ### What Changed

# + id="EHsuTnhkvIN2"
ts_pred_recon_shift = ts_pred_recon - ts_pred

# + id="uHEf4k-yvsfh" colab={"base_uri": "https://localhost:8080/", "height": 587} outputId="d29b8c2a-27a4-47b5-a813-ce1d6f9145c4"
_, ax = plt.subplots(figsize=(10, 6.18))

ts_pred_recon_shift[["Total", "CA", "WI", "TX"]].plot(ax=ax)

# + colab={"base_uri": "https://localhost:8080/", "height": 587} id="PupGNcWXvWz3" outputId="59e24846-8af0-483f-8b9a-b0d163b8faac"
_, ax = plt.subplots(figsize=(10, 6.18))
ts_pred_recon_shift[ca_columns + ["Total"]].plot(ax=ax)

# + [markdown] id="wa_t7wPJCFus"
# To see how the predictions are shifted during reconciliation, we plot out the changes from reconciliation as box plots.

# + id="9qZu3nPTCgPc" colab={"base_uri": "https://localhost:8080/", "height": 553} outputId="390f2562-c7f3-4622-ec6f-b91a779f4a89"
ts_pred_recon_shift[ca_columns + ["Total"]].pd_dataframe().plot.box()

# + id="y5J4t0BPvbiT" colab={"base_uri": "https://localhost:8080/", "height": 553} outputId="4fe335a3-2e70-4837-e943-0151298faf1e"
ts_pred_recon_shift["CA"].pd_dataframe().plot.box()

# + id="RNQ9p8NaysI0" colab={"base_uri": "https://localhost:8080/", "height": 557} outputId="1c0a855b-01d5-4660-e22d-0a77c946bcab"
ts_pred_recon_shift[["Total", "CA", "TX", "WI"]].pd_dataframe().plot.box(
    title="Box Plot for Reconciled - Original Prediction"
)

# + id="X85NA0qd6z6d" colab={"base_uri": "https://localhost:8080/", "height": 1000} outputId="1067bbe8-58fa-4ad4-ca17-acac7482d192"
ts_pred_recon_shift[["Total", "CA", "TX", "WI"]].pd_dataframe()

# + id="X5Lljug99WKu" colab={"base_uri": "https://localhost:8080/"} outputId="63135bfd-bb69-40f4-9dc6-f5d712c993c9"
max(ts_pred.values().max(), ts_pred_recon.values().max())

# + id="IVtJFCEi8G0O" colab={"base_uri": "https://localhost:8080/", "height": 847} outputId="5ae53acd-e368-4397-e8db-8b241f601ed0"
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

# + id="qsCZG5lm9LX1" colab={"base_uri": "https://localhost:8080/", "height": 847} outputId="31716456-ebe8-456e-9d60-604592cb54a4"
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

# + [markdown] id="ChAEMN94PBqQ"
# ### Can Reconciliations Adjust Bias?
#
# We create some artificial bias by shifting one of the series down and then perform reconciliations.
#
# This assumes that the reconciliation already learned about the general patterns on different levels, since we only manually shift the predictions only. The training is not touched.

# + id="sAW-scTPBaTC"
reconciliator_pred_bias = MinTReconciliator(method="wls_val")

# + id="V3nuY0EyQJIk" colab={"base_uri": "https://localhost:8080/", "height": 587} outputId="fb633541-26a6-4793-fd22-54c8d6094bcc"
df_pred_biased = ts_pred.pd_dataframe().copy()
df_pred_biased["CA_1"] = df_pred_biased["CA_1"] * 0.5
ts_pred_biased = TimeSeries.from_dataframe(df_pred_biased, hierarchy=ts_pred.hierarchy)

ts_pred["CA_1"].plot(label="Original Prediction for CA_1")
ts_pred_biased["CA_1"].plot(label="Manually Shifted Prediction for CA_1")

# + id="psBNi4GoQefp"
reconciliator_pred_bias.fit(ts_pred_biased)
ts_pred_biased_recon = reconciliator_pred_bias.transform(ts_pred_biased)

# + id="675Kh3r-SvWx" colab={"base_uri": "https://localhost:8080/", "height": 587} outputId="964e3387-5bc6-4617-bc61-ca5327feacca"
ts_pred["CA_1"].plot(label="Original Prediction for CA_1")
ts_pred_biased["CA_1"].plot(label="Manually Shifted Prediction for CA_1")
ts_pred_biased_recon["CA_1"].plot(label="Reconciled Shifted Prediction for CA_1")

# + id="vzTvw6SrRXr9" colab={"base_uri": "https://localhost:8080/", "height": 587} outputId="1b6c233e-6eb8-4ed3-a610-ddec71724f1a"
ts_pred_biased_recon["CA"].plot(label="CA")
(
    ts_pred_biased_recon["CA_1"]
    + ts_pred_biased_recon["CA_2"]
    + ts_pred_biased_recon["CA_3"]
    + ts_pred_biased_recon["CA_4"]
).plot(label="CA_1 + CA_2 + CA_3 + CA_4", linestyle="--", color="r")

# + id="QbABL83tSh6Q" colab={"base_uri": "https://localhost:8080/", "height": 587} outputId="b5bc823e-da0a-4dad-97cb-ddf528e1d318"
_, ax = plt.subplots(figsize=(10, 6.18))
ca_columns = ["CA", "CA_1", "CA_2", "CA_3", "CA_4"]
ts_test[ca_columns].plot(ax=ax)
ts_pred_biased_recon[ca_columns].plot(linestyle="--", ax=ax)

# + id="j-GFUBWhSnPE" colab={"base_uri": "https://localhost:8080/", "height": 587} outputId="70847105-0006-4feb-c0e3-6728c515400e"
reconciliator_mint_cov = MinTReconciliator(method="mint_cov")

reconciliator_mint_cov.fit(ts_pred - ts_test)

ts_test[ca_columns].plot()
reconciliator_mint_cov.transform(ts_pred)[ca_columns].plot(linestyle="--")

# + id="uTOQnpE9Vdt7"
