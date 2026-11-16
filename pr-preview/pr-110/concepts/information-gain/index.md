# Information Gain

> [!NOTE]
>
> **In one minute**
>
> Information gain is the change in entropy when a dataset is split, and tree- based methods use it to judge a split. The page works through a decision tree node: its entropy before the split, the entropy of the two groups after it, and the gain as their difference.
>
> **Read first:** [Entropy](<https://dl.leima.is/concepts/entropy/index.md>)

Information gain is a frequently used metric in calculating the gain during a split in tree-based methods.

First of all, the [entropy](<https://dl.leima.is/concepts/entropy/index.md>) of a dataset is defined as

\\\[ S = - \\sum\_i p\_i \\log p\_i - \\sum\_i (1-p\_i)\\log (1-p\_i), \\\]

where \\(p\_i\\) is the probability of a class.

The information gain is the change of entropy.

To illustrate this idea, we use decision tree as an example. In a decision tree algorithm, we would split a node. Before splitting, we assign a label \\(m\\) to the node, the entropy is

\\\[ S\_m = - p\_m \\log p\_m - (1-p\_m)\\log (1-p\_m). \\\]

After the splitting, we have two groups that contributes to the entropy, group \\(L\\) and group \\(R\\) [^/concepts/information-gain/#bib-shalev-shwartz_ben-david_2014],

\\\[ S'\_m = p\_L (- p\_m \\log p\_m - (1-p\_m)\\log (1-p\_m)) + p\_R (- p\_m \\log p\_m - (1-p\_m)\\log (1-p\_m)), \\\]

where \\(p\_L\\) and \\(p\_R\\) are the probabilities of the two groups. Suppose we have 100 samples before splitting and 29 samples in the left group and 71 samples in the right group, we have \\(p\_L = 29/100\\) and \\(p\_R = 71/100\\).

The information gain is the difference between \\(S\_m\\) and \\(S'\_m\\),

\\\[ \\mathrm{Gain} = S\_m - S'\_m. \\\]

## Try It: The Gain of a Split

A node of 40 records is split into a left and a right group. Set how many records are class 1, how many go left, and how many of those are class 1. Entropies are in nats.

An interactive diagram of a node of 40 records split into two groups, with sliders for the class mix and the split. It shows the entropy of the node, of both groups, and the information gain. It needs JavaScript. In short: the gain is the entropy of the node minus the size-weighted entropy of the two groups; it is largest when both groups are pure and 0 when both have the same mix as the node.

---

**Related:** [Gini Impurity](<https://dl.leima.is/concepts/gini-impurity/index.md>) · [Tree-Based Models](<https://dl.leima.is/trees/tree.basics/index.md>)

[^/concepts/information-gain/#bib-shalev-shwartz_ben-david_2014]: Shalev-Shwartz S., Ben-David S.. Understanding Machine Learning: From Theory to Algorithms. *Cambridge University Press*. 2014. [https://doi.org/10.1017/CBO9781107298019](<https://doi.org/10.1017/CBO9781107298019>)
