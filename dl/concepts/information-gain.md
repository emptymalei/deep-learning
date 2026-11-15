---
summary: >-
  Information gain is the change in entropy when a dataset is split, and tree-
  based methods use it to judge a split. The page works through a decision
  tree node: its entropy before the split, the entropy of the two groups after
  it, and the gain as their difference.
prerequisites:
  - concepts/entropy.md
related:
  - concepts/gini-impurity.md
  - trees/tree.basics.md
---

# Information Gain

Information gain is a frequently used metric in calculating the gain during a split in tree-based methods.

First of all, the [entropy](entropy.md) of a dataset is defined as

$$
S = - \sum_i p_i \log p_i - \sum_i (1-p_i)\log (1-p_i),
$$

where $p_i$ is the probability of a class.

The information gain is the change of entropy.

To illustrate this idea, we use decision tree as an example. In a decision tree algorithm, we would split a node. Before splitting, we assign a label $m$ to the node, the entropy is

$$
S_m = - p_m \log p_m - (1-p_m)\log (1-p_m).
$$

After the splitting, we have two groups that contributes to the entropy, group $L$ and group $R$ [@shalev-shwartz_ben-david_2014],

$$
S'_m = p_L (- p_m \log p_m - (1-p_m)\log (1-p_m)) + p_R (- p_m \log p_m - (1-p_m)\log (1-p_m)),
$$

where $p_L$ and $p_R$ are the probabilities of the two groups. Suppose we have 100 samples before splitting and 29 samples in the left group and 71 samples in the right group, we have $p_L = 29/100$ and $p_R = 71/100$.

The information gain is the difference between $S_m$ and $S'_m$,

$$
\mathrm{Gain} = S_m - S'_m.
$$


## Try It: The Gain of a Split

A node of 40 records is split into a left and a right group. Set how many records are class 1, how many go left, and how many of those are class 1. Entropies are in nats.

<div class="dl-widget" data-widget="split-gain" markdown>

An interactive diagram of a node of 40 records split into two groups, with sliders for the class mix and the split. It shows the entropy of the node, of both groups, and the information gain. It needs JavaScript. In short: the gain is the entropy of the node minus the size-weighted entropy of the two groups; it is largest when both groups are pure and 0 when both have the same mix as the node.

</div>
