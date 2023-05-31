# Information Gain

Information gain is a frequently used metric in calculating the gain during a split in tree-based methods.

First of all, the [entropy](../entropy) of a dataset is defined as

$$
S = - \sum_i p_i \log p_i - \sum_i (1-p_i)\log p_i,
$$

where $p_i$ is the probability of a class.

The information gain is the change of entropy.

To illustrate this idea, we use decision tree as an example. In a decision tree algorithm, we would split a node. Before splitting, we assign a label $m$ to the node, the entropy is

$$
S_m = - p_m \log p_m - (1-p_m)\log p_m.
$$

After the splitting, we have two groups that contributes to the entropy, group $L$ and group $R$ [@shalev-shwartz_ben-david_2014],

$$
S'_m = p_L (- p_m \log p_m - (1-p_m)\log p_m) + p_R (- p_m \log p_m - (1-p_m)\log p_m),
$$

where $p_L$ and $p_R$ are the probabilities of the two groups. Suppose we have 100 samples before splitting and 29 samples in the left group and 71 samples in the right group, we have $p_L = 29/100$ and $p_R = 71/100$.

The information gain is the difference between $S_m$ and $S'_m$,

$$
\mathrm{Gain} = S_m - S'_m.
$$
