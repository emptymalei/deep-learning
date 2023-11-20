# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py:percent
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.15.2
#   kernelspec:
#     display_name: deep-learning
#     language: python
#     name: deep-learning
# ---

# %% [markdown]
# # Tree Basics

# %%
import json

from typing import List, Literal, Union

import pandas as pd
import numpy as np
from sklearn import tree

import matplotlib.pyplot as plt
import seaborn as sns

sns.set()

# %% [markdown]
# ## Generate Data
#
# We generate some artificial dataset about whether to go to the office or work from home.
#
# We will use three features, `["health", "weather", "holiday"]`. And people are only going to the office, iff
#
# - health=1: not sick,
# - weather=1: good weather,
# - holiday=0: not holiday.
#
# We use `1` to indicate that we go to the office.


# %%
class WFHData:
    """
    Generate a dataset about wheter to go to the office.

    Go to the office, if and only if
    - I am healthy,
    - the weather is good,
    - not holiday.

    Represented in the feature values, this condition is `[1,1,0]`.
    However, we also randomize the target value based on `randomize_prob`:

    - `randomize_prob=0`: keep perfect data, no randomization
    - `randomize_prob=1`: use the wrong target value. The rules are inverted.


    ```python
    wfh = WFHData(length=10)
    ```

    :param length: the number of data points to generate.
    :param randomize_prob: the probability of randomizing the target values.
        `0` indicates that we keep the perfect target value based on rules.
    :param seed: random generator seed.
    """

    def __init__(self, length: int, randomize_prob: int = 0, seed: int = 42):
        self.randomize_prob = randomize_prob
        self.length = length
        self.rng = np.random.default_rng(seed)
        self.x = self._generate_feature_values()
        self.y = self._generate_target_values()

    @property
    def feature_names(self) -> List[str]:
        return ["health", "weather", "holiday"]

    @property
    def target_names(self) -> List[str]:
        return ["go_to_office"]

    @property
    def feature_dataframe(self) -> pd.DataFrame:
        return pd.DataFrame(self.x, columns=self.feature_names)

    @property
    def target_dataframe(self) -> pd.DataFrame:
        return pd.DataFrame(self.y, columns=self.target_names)

    def _generate_feature_values(self) -> List[List[Literal[0, 1]]]:
        """Generate the values for the three features

        The values can only be either `0` or `1`.
        """

        return self.rng.choice([0, 1], (self.length, len(self.feature_names))).tolist()

    def _perfect_target(self) -> List[Literal[0, 1]]:
        """Create target value based on rules:

        Go to the office, if and only if
        - I am healthy,
        - the weather is good,
        - not holiday.

        Represented in the feature values, this condition is `[1,1,0]`.
        """
        target = []
        for i in self.x:
            if i == [1, 1, 0]:
                target.append(1)
            else:
                target.append(0)

        return target

    @staticmethod
    def _randomize_target(y, rng, probability: float) -> Literal[0, 1]:
        """Randomly choose from the current value `y` and its alternative.
        For example, if current value of `y=0`, its alternative is `1`.
        We will randomly choose from `0` and `1` based on the specified probability.

        If `probability=0`, we return the current value, i.e., `0`.
        If `probability=0`, we return the alternative value, i.e., `1`.
        Otherwise, it is randomly selected based on the probability.
        """
        alternative_y = 1 if y == 0 else 0

        return rng.choice(
            [y, alternative_y], 1, p=(1 - probability, probability)
        ).item()

    def _generate_target_values(self) -> List[Literal[0, 1]]:
        """Generate the target values"""
        y = self._perfect_target()
        y = [self._randomize_target(i, self.rng, self.randomize_prob) for i in y]

        return y


# %%
wfh_demo = WFHData(length=10)

# %%
wfh_demo.feature_dataframe

# %%
wfh_demo.target_dataframe

# %% [markdown]
# ## Decision Tree on Perfect Data

# %%
wfh_pure = WFHData(length=100, randomize_prob=0)

# %%
clf_pure = tree.DecisionTreeClassifier()
clf_pure.fit(wfh_pure.feature_dataframe, wfh_pure.target_dataframe)

# %%
fig, ax = plt.subplots(figsize=(15, 15))
tree.plot_tree(clf_pure, feature_names=wfh_pure.feature_names, ax=ax)
ax.set_title("Tree Trained on Perfect Data")

# %% [markdown]
# ## Impure Data

# %%
wfh_impure = WFHData(length=100, randomize_prob=0.1)

# %%
clf_impure = tree.DecisionTreeClassifier(
    max_depth=20, min_samples_leaf=1, min_samples_split=0.0001
)
clf_impure.fit(wfh_impure.feature_dataframe, wfh_impure.target_dataframe)

# %%
fig, ax = plt.subplots(figsize=(15, 10))
tree.plot_tree(clf_impure, feature_names=wfh_impure.feature_names, ax=ax)
ax.set_title("Tree Trained on Imperfect Data")


# %% [markdown]
# ## Understand Gini Impurity

# %% [markdown]
# ### Gini Impurity for 2 possible classes


# %%
def gini_2(p1: float, p2: float) -> Union[None, float]:
    """Compute the Gini impurity for the two input values."""
    if p1 + p2 <= 1:
        return p1 * (1 - p1) + p2 * (1 - p2)
    else:
        return None


# %%
gini_2_test_p1 = np.linspace(0, 1, 1001)
gini_2_test_p2 = np.linspace(0, 1, 1001)

# %%
gini_2_test_impurity = [
    [gini_2(p1, p2) for p1 in gini_2_test_p1] for p2 in gini_2_test_p2
]

# %%
df_gini_2_test = pd.DataFrame(
    gini_2_test_impurity,
    index=[f"{i:0.2f}" for i in gini_2_test_p2],
    columns=[f"{i:0.2f}" for i in gini_2_test_p1],
)

# %%
fig, ax = plt.subplots(figsize=(12, 10))
sns.heatmap(df_gini_2_test.loc[::-1,], ax=ax)
ax.set_xlabel("$p_1$")
ax.set_ylabel("$p_2$")
ax.set_title("Gini Impurity for Data with 2 Possible Values")


# %% [markdown]
# ### Gini Impurity for 3 possible classes


# %%
def gini_3(p1: float, p2: float) -> Union[None, float]:
    """Computes the gini impurity for three potential classes"""
    if p1 + p2 <= 1:
        return p1 * (1 - p1) + p2 * (1 - p2) + (1 - p1 - p2) * (p1 + p2)
    else:
        return None


# %%
gini_3_test_p1 = np.linspace(0, 1, 1001)
gini_3_test_p2 = np.linspace(0, 1, 1001)
gini_3_test_impurity = [
    [gini_3(p1, p2) for p1 in gini_3_test_p1] for p2 in gini_3_test_p2
]

df_gini_3_test = pd.DataFrame(
    gini_3_test_impurity,
    index=[f"{i:0.2f}" for i in gini_3_test_p2],
    columns=[f"{i:0.2f}" for i in gini_3_test_p1],
)

# %%
fig, ax = plt.subplots(figsize=(12, 10))
sns.heatmap(df_gini_3_test.loc[::-1,], ax=ax)
ax.set_xlabel("$p_1$")
ax.set_ylabel("$p_2$")
ax.set_title("Gini Impurity for Data with 3 Possible Values")

# %%

# %%
