# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py
#     text_representation:
#       extension: .py
#       format_name: light
#       format_version: '1.5'
#       jupytext_version: 1.14.5
#   kernelspec:
#     display_name: mini-code
#     language: python
#     name: mini-code
# ---

# ## Random Forest Playground

# Outline
#
# 1. Generate data of specific functions
# 2. Fit the functions using ensemble methods
# 3. Analyze the trees

from sklearn.ensemble import RandomForestRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.model_selection import (
    train_test_split,
    cross_val_score,
    learning_curve,
    validation_curve,
    GridSearchCV,
    RandomizedSearchCV,
)
from sklearn.feature_selection import (
    SelectKBest,
    SelectFromModel,
    chi2,
    mutual_info_regression,
)
from sklearn.pipeline import Pipeline
import sklearn.tree as _tree

import matplotlib.pyplot as plt
import matplotlib as mpl

mpl.rcParams["axes.unicode_minus"] = False
import seaborn as sns

import numpy as np
from random import random

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
    "rf__max_features": max_features,
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

# -

# ### Data without Noise

# +
X_sin = [[6 * random()] for i in range(10000)]
y_sin = np.sin(X_sin)

X_sin_test = [[6 * random()] for i in range(10000)]
y_sin_test = np.sin(X_sin_test)


# +

model = RandomizedSearchCV(
    pipeline, cv=10, param_distributions=rf_random_grid, verbose=3, n_jobs=-1
)
# -

model.fit(X_sin, y_sin)


model.score(X_sin, y_sin)

model.best_params_

dump(model, "reports/rf_sin.joblib")

# Plot out the result

# +
fig, ax = plt.subplots(figsize=(10, 6.18))

ax.plot(X_sin_test, model.predict(X_sin_test), ".")
ax.plot(
    [y for y, _ in sorted(zip(X_sin_test, y_sin_test))],
    [x for _, x in sorted(zip(X_sin_test, y_sin_test))],
    "k-",
)

ax.set_title(
    f"Random Forest on Sin Data; Test $R^2$ Score: {model.score(X_sin_test, y_sin_test):0.2f}"
)
# -

X_sin_est = sorted(X_sin_test)[::500]
y_sin_est_pred = []
for i in X_sin_est:
    i_y_sin_est_pred = []
    for est in model.best_estimator_["rf"].estimators_:
        i_y_sin_est_pred.append(est.predict([i]).tolist())
    i_y_sin_est_pred = sum(i_y_sin_est_pred, [])
    y_sin_est_pred.append(i_y_sin_est_pred)

y_sin_est_pred_boxsize = []
for i in y_sin_est_pred:
    y_sin_est_pred_boxsize.append(np.percentile(i, 75) - np.percentile(i, 25))

# +
fig, ax = plt.subplots(figsize=(10, 6.18))

ax.boxplot(y_sin_est_pred)

ax.set_xticklabels([f"{i:0.2f}" for i in sum(X_sin_est, [])])


ax.set_title("Box Plot for Tree Predictions on Random Forest on Sin Data")


# +
fig, ax = plt.subplots(figsize=(10, 6.18))

ax.plot(X_sin_est, y_sin_est_pred_boxsize)
# -

fig, ax = plt.subplots(figsize=(10, 6.18))
sns.distplot(y_sin_est_pred_boxsize, bins=20, ax=ax)


# ### Data with Noise

# +

X_sin_noise = [[6 * random()] for i in range(10000)]
y_sin_noise = [i * (1 + 0.1 * (random() - 0.5)) for i in np.sin(X_sin)]

X_sin_noise_test = [[6 * random()] for i in range(10000)]
y_sin_noise_test = [i * (1 + 0.1 * (random() - 0.5)) for i in np.sin(X_sin_noise_test)]
# -

model_noise = RandomizedSearchCV(
    pipeline, cv=10, param_distributions=rf_random_grid, verbose=3, n_jobs=-1
)

model_noise.fit(X_sin_noise, y_sin_noise)


model_noise.score(X_sin, y_sin)

model_noise.best_params_

dump(model_noise, "reports/rf_sin_noise.joblib")


fig, ax = plt.subplots(figsize=(2 * 10, 2 * 6.18))
_tree.plot_tree(model_noise.best_estimator_["rf"].estimators_[0])

# +
fig, ax = plt.subplots(figsize=(10, 6.18))


ax.plot(X_sin_noise_test, model.predict(X_sin_noise_test), "r.", alpha=0.1)

ax.plot(np.linspace(0, 6, 100), np.sin(np.linspace(0, 6, 100)), "k-")

ax.set_title(
    f"Random Forest on Sin Data with Noise; Test $R^2$ Score: {model.score(X_sin_noise_test, y_sin_noise_test):0.2f}"
)
# -


# +
X_sin_noise_est = sorted(X_sin_noise_test[::100])
y_sin_noise_est_pred = []

for i in X_sin_noise_est:
    i_y_sin_noise_est_pred = []
    for est in model.best_estimator_["rf"].estimators_:
        i_y_sin_noise_est_pred.append(est.predict([i]).tolist())
    i_y_sin_noise_est_pred = sum(i_y_sin_noise_est_pred, [])
    y_sin_noise_est_pred.append(i_y_sin_noise_est_pred)
# -

y_sin_noise_est_pred_boxsize = []
for i in y_sin_noise_est_pred:
    y_sin_noise_est_pred_boxsize.append(np.percentile(i, 75) - np.percentile(i, 25))

# +
fig, ax = plt.subplots(figsize=(10, 6.18))

ax.boxplot(y_sin_noise_est_pred)

# +
fig, ax = plt.subplots(figsize=(10, 6.18))

ax.plot(y_sin_noise_est_pred_boxsize)

# +
fig, ax = plt.subplots(figsize=(10, 6.18))

sns.distplot(y_sin_noise_est_pred_boxsize, bins=20, ax=ax)

# +
fig, ax = plt.subplots(figsize=(10, 6.18))

sns.distplot(
    y_sin_noise_est_pred_boxsize, bins=20, ax=ax, hist=False, label="with Noise"
)

sns.distplot(y_sin_est_pred_boxsize, bins=20, ax=ax, hist=False, label="without Noise")

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
# -
