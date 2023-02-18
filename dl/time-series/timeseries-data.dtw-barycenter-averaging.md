# DTW Barycenter Averaging

DTW Barycenter Averaging (**DBA**) constructs a series $\bar{\mathcal S}$ out of a set of series $\{\mathcal S^{(\alpha)}\}$ so that $\bar{\mathcal S}$ is the barycenter of $\{\mathcal S^{(\alpha)}\}$ measured by Dynamic Time Warping (**DTW**) distance [@Petitjean2011-sj].


## Dynamic Time Warping (**DTW**)

Given two sequences, $S^{(1)}$ and $S^{(2)}$, the Dynamic Time Warping (**DTW**) algorithm finds the best way to align two sequences. During this alignment process, we quantify the misalignment using a distance similar to the Levenshtein distance, where the distance between two series $S^{(1)}_{1:i}$ (with $i$ elements) and $S^{(2)}_{1:j}$ (with $j$ elements) is[@Petitjean2011-sj]

$$
\begin{align}
D(S^{(1)}_{1:i}, S^{(2)}_{1:j}) =& d(S^{(1)}_i, S^{(2)}_j)\\
& + \operatorname{min}\left[ D(S^{(1)}_{1:i-1}, S^{(2)}_{1:j-1}), D(S^{(1)}_{1:i}, S^{(2)}_{1:j-1}), D(S^{(1)}_{1:i-1}, S^{(2)}_{1:j}) \right],
\end{align}
$$

where $S^{(1)}_i$ is the $i$the element of the series $S^{(1)}$, $d(x,y)$ is a predetermined distance, e.g., Euclidean distance. This definition reveals the recursive nature of the DTW distance.


??? note "Notations in the Definition: $S_{1:i}$ and $S_{i}$"

    The notation $S_{1:i}$ stands for a series that contains the elements starting from the first to the $i$th in series $S$. For example, we have a series

    $$
    S^1 = [s^1_1, s^1_2, s^1_3, s^1_4, s^1_5, s^1_6].
    $$

    The notation $S^1_{1:4}$ represents

    $$
    S^1_{1:4} = [s^1_1, s^1_2, s^1_3, s^1_4].
    $$

    The notation $S_i$ indicates the $i$th element in $S$. For example,

    $$
    S^1_4 = s^1_4.
    $$

    If we map these two notations to Python,

    - $S_{1:i}$ is equivalent to `S[0:i]`, and
    - $S_i$ is equivalent to `S[i-1]`.

    Note that the indices in Python look strange. This is also the reason we choose to use subscripts not square brackets in our definition.



??? note "Levenshtein Distance"

    Given two words, e.g., $w^{a} = \mathrm{cats}$ and $w^{b} = \mathrm{katz}$. Suppose we can only use three operations: insertions, deletions and substitutions. The Levenshtein distance calculates the number of such operations needed to change from the first word $w^a$ to the second one $w^b$ by applying single-character edits. In this example, we need two replacements, i.e., `"c" -> "k"` and `"s" -> "z"`.

    The Levenshtein distance can be solved using recursive algorithms [^trekhleb].


## Barycenter Averaging Based on DTW Distance

Petitjean et al proposed a time series averaging algorithm based on DTW distance which is dubbed DTW Barycenter Averaging (**DBA**).


!!! note "DBA Implementation"

    https://github.com/fpetitjean/DBA



[^trekhleb]: trekhleb. javascript-algorithms/src/algorithms/string/levenshtein-distance at master · trekhleb/javascript-algorithms. In: GitHub [Internet]. [cited 27 Jul 2022]. Available: https://github.com/trekhleb/javascript-algorithms/tree/master/src/algorithms/string/levenshtein-distance
