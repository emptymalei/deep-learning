# DTW Barycenter Averaging

DTW Barycenter Averaging (**DBA**) constructs a series $\bar{\mathcal S}$ out of a set of series $\{\mathcal S_i\}$ so that $\bar{\mathcal S}$ is the barycenter of $\{\mathcal S_i\}$ measured by Dynamic Time Warping (**DTW**) distance.[@Petitjean2011-sj]


## Dynamic Time Warping (**DTW**)


!!! note "Levenshtein Distance"

    Given two words, e.g., $w^{a} = \mathrm{cats}$ and $w^{b} = \mathrm{katz}$. Suppose we can only use three operations: insertions, deletions and substitutions. The Levenshtein distance calculates the number of such operations needed to change from the first word $w^a$ to the second one $w^b$ by applying single-character edits. In this example, we need two replacements, i.e., `"c" -> "k"` and `"s" -> "z"`.

    The Levenshtein distance can be solved using recursive algorithms.[^trekhleb]



[^trekhleb]: trekhleb. javascript-algorithms/src/algorithms/string/levenshtein-distance at master · trekhleb/javascript-algorithms. In: GitHub [Internet]. [cited 27 Jul 2022]. Available: https://github.com/trekhleb/javascript-algorithms/tree/master/src/algorithms/string/levenshtein-distance
