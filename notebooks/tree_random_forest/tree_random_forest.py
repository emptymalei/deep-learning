# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py:light
#     text_representation:
#       extension: .py
#       format_name: light
#       format_version: '1.5'
#       jupytext_version: 1.15.2
#   kernelspec:
#     display_name: deep-learning
#     language: python
#     name: deep-learning
# ---

# # Random Forest Playground

# Outline
#
# 1. Generate data of specific functions
# 2. Fit the functions using ensemble methods
# 3. Analyze the trees

import matplotlib as mpl
import matplotlib.pyplot as plt
import pandas as pd
import sklearn.tree as _tree
from sklearn.ensemble import RandomForestRegressor
from sklearn.feature_selection import (
    SelectFromModel,
    SelectKBest,
    chi2,
    mutual_info_regression,
)
from sklearn.model_selection import (
    GridSearchCV,
    RandomizedSearchCV,
    cross_val_score,
    learning_curve,
    train_test_split,
    validation_curve,
)
from sklearn.pipeline import Pipeline
from sklearn.tree import DecisionTreeRegressor

mpl.rcParams["axes.unicode_minus"] = False
from random import random

import numpy as np
import seaborn as sns
from joblib import dump, load

# ## Model

# ### Components

# +
# Number of trees in random forest
n_estimators = [int(x) for x in np.linspace(1000, 4000, 11)]
# Number of features to consider at every split
max_features = ["auto", "sqrt"]
# Maximum number of levels in tree
max_depth = [int(x) for x in range(10, 30, 2)]
max_depth.append(None)
# Minimum number of samples required to split a node
min_samples_split = [0.001, 0.01, 0.02, 0.05, 0.1, 0.2]
# Minimum number of samples required at each leaf node
min_samples_leaf = [10, 20, 30, 40, 50]
# Method of selecting samples for training each tree
bootstrap = [True, False]


rf_random_grid = {
    "rf__n_estimators": n_estimators,
    #     "rf__max_features": max_features,
    "rf__max_depth": max_depth,
    "rf__min_samples_split": min_samples_split,
    "rf__min_samples_leaf": min_samples_leaf,
    "rf__bootstrap": bootstrap,
}

rf = RandomForestRegressor(random_state=42, oob_score=True)

##########

pipeline_steps = [
    ("rf", rf),
]

pipeline = Pipeline(pipeline_steps)


# +
def pred_true_comparison_plot(dataframe, ax, pred_sample=100):
    sns.scatterplot(
        dataframe, x="x", y="y", ax=ax, label="y", marker=".", ec="face", s=5
    )

    sns.scatterplot(
        dataframe.sample(100),
        x="x",
        y="y_pred",
        ax=ax,
        label="y_pred",
        marker="+",
        s=100,
        linewidth=2,
    )

    return ax


def predictions_each_estimators(x, rf_model):
    preds = []
    for i in x:
        i_preds = []
        for est in rf_model.best_estimator_["rf"].estimators_:
            i_preds.append(est.predict([[i]]).tolist())
        i_preds = sum(i_preds, [])
        preds.append(i_preds)

    return {"x": pd.DataFrame(x, columns=["x"]), "preds": pd.DataFrame(preds)}


# -

# ### Data without Noise

# +
X_sin = [6 * random() for i in range(10000)]
y_sin = np.sin(X_sin)

X_sin_test = [6 * random() for i in range(10000)]
y_sin_test = np.sin(X_sin_test)
# -


df_sin = pd.DataFrame(
    {
        "x": X_sin,
        "y": y_sin,
    }
)

model = RandomizedSearchCV(
    pipeline, cv=10, param_distributions=rf_random_grid, verbose=3, n_jobs=-1
)

model.fit(df_sin[["x"]], df_sin["y"].values.ravel())


sin_score = model.score(df_sin[["x"]], df_sin["y"].values.ravel())

model.best_params_

# +
# dump(model, "reports/rf_sin.joblib")
# -

fig, ax = plt.subplots(figsize=(10 * 10, 4 * 10))
_tree.plot_tree(model.best_estimator_["rf"].estimators_[0], fontsize=7)


# Plot out the result

df_sin["y_pred"] = model.predict(df_sin[["x"]])

df_sin

# +
fig, ax = plt.subplots(figsize=(10, 6.18))

pred_true_comparison_plot(df_sin, ax)
ax.set_title(f"Random Forest on Sin Data; $R^2$ Score: {sin_score:0.2f}")
plt.legend()
# -

# Plot out the boxplots of each data point

est_sample_skip = 100

sin_est_pred = predictions_each_estimators(sorted(X_sin_test)[::est_sample_skip], model)

# +
df_sin_est_quantiles = pd.merge(
    sin_est_pred["x"],
    sin_est_pred["preds"].quantile(q=[0.75, 0.25], axis=1).T,
    how="left",
    left_index=True,
    right_index=True,
)

df_sin_est_quantiles["boxsize"] = (
    df_sin_est_quantiles[0.75] - df_sin_est_quantiles[0.25]
)

# +
fig, ax = plt.subplots(figsize=(10, 1.5 * 6.18))
fig_skip = 5

ax.violinplot(
    sin_est_pred["preds"].values.tolist()[::fig_skip],
    positions=sin_est_pred["x"].x.tolist()[::fig_skip],
)

sns.lineplot(df_sin, x="x", y="y", ax=ax, label="y")

plt.xticks([])
# ax.yaxis.set_major_locator(mpl.ticker.FixedLocator(range(10)))
# ax.set_xticklabels([f"{i:0.2f}" for i in sin_est_pred["x"].x])

ax.set_title(
    "Violin Plot for All Predictions of Each Tree in a Random Forest on Some Sin Data Points"
)

# +
fig, ax = plt.subplots(figsize=(10, 2 * 6.18))
fig_skip = 5

ax.boxplot(
    sin_est_pred["preds"].values.tolist()[::fig_skip],
    positions=sin_est_pred["x"].x.tolist()[::fig_skip],
)
sns.lineplot(df_sin, x="x", y="y", ax=ax, label="y")

ax.set_xticklabels([f"{i:0.2f}" for i in sin_est_pred["x"].x[::fig_skip]])

ax.set_title("Box Plot for Tree Predictions on Random Forest on Sin Data")
# -


df_sin_est_quantiles

# +
fig, ax = plt.subplots(figsize=(10, 6.18))

sns.barplot(df_sin_est_quantiles, x="x", y="boxsize")

ax.set_xticklabels([f"{i:0.2f}" for i in sin_est_pred["x"].x])

# +
fig, ax = plt.subplots(figsize=(10, 6.18))

sns.histplot(df_sin_est_quantiles.boxsize, ax=ax)
ax.set_yscale("log")
ax.set_xscale("log")
# -


# ### Data with Noise

# +
X_sin_noise = np.array([6 * random() for i in range(10000)])
y_sin_noise = np.array([i + 0.1 * (random() - 0.5) for i in np.sin(X_sin_noise)])


df_sin_noise = pd.DataFrame({"x": X_sin_noise, "y": y_sin_noise})

# +
fig, ax = plt.subplots(figsize=(10, 6.18))

sns.lineplot(df_sin_noise, x="x", y="y", ax=ax, label="y")

# -

model_noise = RandomizedSearchCV(
    pipeline, cv=10, param_distributions=rf_random_grid, verbose=3, n_jobs=-1
)

model_noise.fit(df_sin_noise[["x"]], df_sin_noise["y"].values.ravel())


sin_noise_score = model_noise.score(df_sin_noise[["x"]], df_sin_noise["y"])

model_noise.best_params_

# +
# dump(model_noise, "reports/rf_sin_noise.joblib")
# -


fig, ax = plt.subplots(figsize=(9 * 10, 4 * 10))
_tree.plot_tree(model_noise.best_estimator_["rf"].estimators_[0], fontsize=7)

df_sin_noise["y_pred"] = model_noise.predict(df_sin_noise[["x"]])

# +
fig, ax = plt.subplots(figsize=(10, 6.18))

pred_true_comparison_plot(df_sin_noise, ax)

ax.set_title(
    f"Random Forest on Sin Data with Noise; Test $R^2$ Score: {sin_noise_score:0.2f}"
)

plt.legend()
# -


sin_noise_est_pred = predictions_each_estimators(
    sorted(X_sin_noise)[::100], model_noise
)

# +
df_sin_noise_est_quantiles = pd.merge(
    sin_noise_est_pred["x"],
    sin_noise_est_pred["preds"].quantile(q=[0.75, 0.25], axis=1).T,
    how="left",
    left_index=True,
    right_index=True,
)

df_sin_noise_est_quantiles["boxsize"] = (
    df_sin_noise_est_quantiles[0.75] - df_sin_noise_est_quantiles[0.25]
)

# +
fig, ax = plt.subplots(figsize=(10, 1.5 * 6.18))
fig_skip = 5

ax.violinplot(
    sin_noise_est_pred["preds"].values.tolist()[::fig_skip],
    positions=sin_noise_est_pred["x"].x.tolist()[::fig_skip],
)

sns.scatterplot(
    df_sin_noise, x="x", y="y", ax=ax, label="y", marker=".", ec="face", s=1
)

plt.xticks([])

ax.set_title(
    "Violin Plot for All Predictions of Each Tree in a Random Forest on Some Sin Data Points"
)

# +
fig, ax = plt.subplots(figsize=(10, 2 * 6.18))
fig_skip = 5

ax.boxplot(
    sin_noise_est_pred["preds"].values.tolist()[::fig_skip],
    positions=sin_noise_est_pred["x"].x.tolist()[::fig_skip],
)
sns.scatterplot(
    df_sin_noise, x="x", y="y", ax=ax, label="y", marker=".", ec="face", s=1
)


ax.set_xticklabels([f"{i:0.2f}" for i in sin_noise_est_pred["x"].x[::fig_skip]])

ax.set_title("Box Plot for Tree Predictions on Random Forest on Sin Data")

# +
df_sin_noise_est_quantiles["model"] = "with_noise"
df_sin_est_quantiles["model"] = "no_noise"

df_quantiles = pd.concat(
    [
        df_sin_est_quantiles[["model", "boxsize"]],
        df_sin_noise_est_quantiles[["model", "boxsize"]],
    ]
)

# +
fig, ax = plt.subplots(figsize=(10, 6.18))

sns.boxplot(df_quantiles, x="boxsize", y="model", ax=ax)

# +
fig, ax = plt.subplots(figsize=(10, 6.18))

sns.histplot(
    df_sin_noise_est_quantiles.boxsize,
    #     bins=20,
    ax=ax,
    kde=True,
    label="with Noise",
    stat="probability",
    binwidth=0.002,
    binrange=(0, 0.07),
)

sns.histplot(
    df_sin_est_quantiles.boxsize,
    #     bins=20,
    ax=ax,
    kde=True,
    label="without Noise",
    stat="probability",
    binwidth=0.002,
    binrange=(0, 0.07),
)

plt.legend()

# +
fig, ax = plt.subplots(figsize=(10, 6.18))

sns.kdeplot(
    df_sin_noise_est_quantiles.boxsize,
    #     bins=20,
    ax=ax,
    #     hist=False,
    label="with Noise",
)

sns.kdeplot(
    df_sin_est_quantiles.boxsize,
    #     bins=20,
    ax=ax,
    #     hist=True,
    label="without Noise",
)
plt.legend()
# -


# ## Generalization Error

# $$
# P_{err} \leq \frac{\bar \rho (1-s^2) }{s^2}.
# $$


def generalization_error(rhob, s):
    res = rhob * (1 - s**2) / s**2

    if res > 1:
        res = 1

    return res


generalization_error(0.2, 0.8)

# +
pe_data = [
    [generalization_error(rhob, s) for s in np.linspace(0.01, 0.1, 10)]
    for rhob in np.linspace(0.01, 0.1, 10)
]

pe_data_s_label = [s for s in np.linspace(0.01, 0.1, 10)]

pe_data_rhob_label = [rhob for rhob in np.linspace(0.01, 0.1, 10)]

fig, ax = plt.subplots(figsize=(10, 10))

sns.heatmap(pe_data, center=0, ax=ax)

ax.set_xlabel("s")
ax.set_ylabel("correlation")
ax.set_xticklabels([f"{i:0.2f}" for i in pe_data_s_label])
ax.set_yticklabels([f"{i:0.2f}" for i in pe_data_rhob_label])

# +
temp_space = np.linspace(0.1, 1, 91)

pe_data = [[generalization_error(rhob, s) for s in temp_space] for rhob in temp_space]

pe_data_s_label = [s for s in temp_space]

pe_data_rhob_label = [rhob for rhob in temp_space]

fig, ax = plt.subplots(figsize=(12, 10))

sns.heatmap(pe_data, center=0, ax=ax)

ax.set_xlabel("s")
ax.set_ylabel("correlation")

ax.set_xticklabels([f"{i:0.2f}" for i in (ax.get_xticks() + 0.1) / 100])
ax.set_yticklabels([f"{i:0.2f}" for i in (ax.get_yticks() + 0.1) / 100])

for label in ax.xaxis.get_ticklabels()[::2]:
    label.set_visible(False)

for label in ax.yaxis.get_ticklabels()[::2]:
    label.set_visible(False)


ax.set_title("Upper Limit of Generalization Error of Random Forest")
