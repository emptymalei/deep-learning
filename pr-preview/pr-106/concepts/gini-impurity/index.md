# Gini Impurity

> [!NOTE]
>
> **In one minute**
>
> Gini impurity measures how mixed the classes in a dataset are: it is 0 when every record has the same class and largest when the classes are evenly mixed. The page defines it, works through two small examples and shows its values for two and for three classes.

Suppose we have a dataset \\(\\{0,1\\}^{12}\\), which has 12 records and 2 possible classes of objects \\(\\{0,1\\}\\) in each record.

The first example we investigate is a pure 0 dataset.

| object |
| :---: |
| 0 |
| 0 |
| 0 |
| 0 |
| 0 |
| 0 |
| 0 |
| 0 |
| 0 |
| 0 |
| 0 |
| 0 |

For such an all-0 dataset, we would like to define its impurity as 0. Same with an all-1 dataset. For a dataset with 50% of 1 and 50% of 0, we would define its impurity as max due to the symmetries between 0 and 1.

## Definition

Given a dataset \\(\\{0,1,...,d\\}^n\\), the Gini impurity is calculated as

\\\[ G = \\sum\_{i \\in \\{0,1,...,d\\} } p(i)(1-p(i)), \\\]

where \\(p(i)\\) is the probability of a randomly picked record being class \\(i\\).

In the above example, we have two classes, \\(\\{0,1\\}\\). The probabilities are

\\\[ \\begin{align} p(0) =&amp; 1\\\\ p(1) =&amp; 0 \\end{align}. \\\]

The Gini impurity is

\\\[ G = p(0)(1-p(0)) + p(1)(1-p(1)) = 0+0 = 0. \\\]

## Examples

Suppose we have another dataset with 50% of the values being 1.

| object |
| :---: |
| 0 |
| 0 |
| 1 |
| 0 |
| 0 |
| 1 |
| 1 |
| 1 |
| 0 |
| 0 |
| 1 |
| 1 |

The Gini impurity is

\\\[ G = p(0)(1-p(0)) + p(1)(1-p(1)) = 0.5 \* 0.5+ 0.5\*0.5 = 0.5. \\\]

For data with two possible values \\(\\{0,1\\}\\), the maximum Gini impurity is 0.5. The following chart shows all the possible values of the Gini impurity for a two-value dataset.

The following heatmap shows the Gini impurity for data with two possible values. The color indicates the Gini impurity.

![Gini Two Values](<https://dl.leima.is/concepts/assets/gini-impurity/gini-2-values-heatmap.png>)

For data with three possible values, the Gini impurity is also visualized using the same chart given the condition that \\(p\_3 = 1 - p\_1 - p\_2\\).

The following chart shows the Gini impurity for data with three possible values. The color indicates the Gini impurity.

![Gini 3 Values](<https://dl.leima.is/concepts/assets/gini-impurity/gini-3-values-heatmap.png>)

## Try It: Gini Impurity of a Two-class Dataset

Change how many of the 20 records are class 1 and see the Gini impurity move along its curve.

An interactive chart of a two-class dataset of 20 records, with a slider for the number of records of class 1. It shows the Gini impurity and its curve over all mixes. It needs JavaScript. In short: the impurity is 0 for a pure dataset and largest, 0.5, for an even mix.

---

**Related:** [Information Gain](<https://dl.leima.is/concepts/information-gain/index.md>) · [Tree-Based Models](<https://dl.leima.is/trees/tree.basics/index.md>)
