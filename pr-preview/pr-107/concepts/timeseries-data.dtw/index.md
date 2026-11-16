# Dynamic Time Warping (**DTW**)

> [!NOTE]
>
> **In one minute**
>
> Dynamic time warping (DTW) finds the best way to align two sequences and measures their misalignment with a distance defined recursively, much like the Levenshtein distance between two words. It is useful for comparing series of different lengths, for example a forecast and an actual series.

Given two sequences, \\(S^{(1)}\\) and \\(S^{(2)}\\), the Dynamic Time Warping (**DTW**) algorithm finds the best way to align two sequences. During this alignment process, we quantify the misalignment using a distance similar to the Levenshtein distance, where the distance between two series \\(S^{(1)}\_{1:i}\\) (with \\(i\\) elements) and \\(S^{(2)}\_{1:j}\\) (with \\(j\\) elements) is[^/concepts/timeseries-data.dtw/#bib-Petitjean2011-sj]

\\\[ \\begin{align} D(S^{(1)}\_{1:i}, S^{(2)}\_{1:j}) =&amp; d(S^{(1)}\_i, S^{(2)}\_j)\\\\ &amp; + \\operatorname{min}\\left\[ D(S^{(1)}\_{1:i-1}, S^{(2)}\_{1:j-1}), D(S^{(1)}\_{1:i}, S^{(2)}\_{1:j-1}), D(S^{(1)}\_{1:i-1}, S^{(2)}\_{1:j}) \\right\], \\end{align} \\\]

where \\(S^{(1)}\_i\\) is the \\(i\\)th element of the series \\(S^{(1)}\\), \\(d(x,y)\\) is a predetermined distance, e.g., Euclidean distance. This definition reveals the recursive nature of the DTW distance.

> [!NOTE]
>
> **Notations in the Definition: \\(S\_{1:i}\\) and \\(S\_{i}\\)**
>
> The notation \\(S\_{1:i}\\) stands for a series that contains the elements starting from the first to the \\(i\\)th in series \\(S\\). For example, we have a series
>
> \\\[ S^1 = \[s^1\_1, s^1\_2, s^1\_3, s^1\_4, s^1\_5, s^1\_6\]. \\\]
>
> The notation \\(S^1\_{1:4}\\) represents
>
> \\\[ S^1\_{1:4} = \[s^1\_1, s^1\_2, s^1\_3, s^1\_4\]. \\\]
>
> The notation \\(S\_i\\) indicates the \\(i\\)th element in \\(S\\). For example,
>
> \\\[ S^1\_4 = s^1\_4. \\\]
>
> If we map these two notations to Python,
>
> - \\(S\_{1:i}\\) is equivalent to `S[0:i]`, and
> - \\(S\_i\\) is equivalent to `S[i-1]`.
>
> Note that the indices in Python look strange. This is also the reason we choose to use subscripts not square brackets in our definition.

> [!NOTE]
>
> **Levenshtein Distance**
>
> Given two words, e.g., \\(w^{a} = \\mathrm{cats}\\) and \\(w^{b} = \\mathrm{katz}\\). Suppose we can only use three operations: insertions, deletions and substitutions. The Levenshtein distance calculates the number of such operations needed to change from the first word \\(w^a\\) to the second one \\(w^b\\) by applying single-character edits. In this example, we need two replacements, i.e., `"c" -> "k"` and `"s" -> "z"`.
>
> The Levenshtein distance can be solved using recursive algorithms [^/concepts/timeseries-data.dtw/#trekhleb].

DTW is very useful when comparing series with different lengths. For example, most error metrics require the actual time series and predicted series to have the same length. In the case of different lengths, we can perform DTW when calculating these metrics[^/concepts/timeseries-data.dtw/#darts-dtw-metric].

## Try It: Aligning Two Series

The second series has the same shape as the first, shifted and stretched in time. The lines show which points DTW aligns, and the grid shows the accumulated cost \\(D\\) from the definition above with the cheapest path through it. The local distance is \\(d(x, y) = \\lvert x - y \\rvert\\).

An interactive chart of two series of 18 points, with sliders that shift and stretch the second one in time. It shows which points dynamic time warping aligns, the accumulated cost matrix with the cheapest path, and the DTW distance next to the point-by-point distance. It needs JavaScript. In short: for the same shape at a different pace, the DTW distance stays small while the point-by-point distance grows.

## Examples

The forecasting package darts provides a [demo of DTW](<https://unit8co.github.io/darts/examples/12-Dynamic-Time-Warping-example.html>).

---

**Related:** [DTW Barycenter Averaging](<https://dl.leima.is/concepts/timeseries-data.dtw-barycenter-averaging/index.md>) · [Time Series Forecasting Metrics](<https://dl.leima.is/time-series/timeseries-metrics.forecasting/index.md>)

[^/concepts/timeseries-data.dtw/#trekhleb]: trekhleb. javascript-algorithms/src/algorithms/string/levenshtein-distance at master · trekhleb/javascript-algorithms. In: GitHub \[Internet\]. \[cited 27 Jul 2022\]. Available: [https://github.com/trekhleb/javascript-algorithms/tree/master/src/algorithms/string/levenshtein-distance](<https://github.com/trekhleb/javascript-algorithms/tree/master/src/algorithms/string/levenshtein-distance >)

[^/concepts/timeseries-data.dtw/#darts-dtw-metric]: Unit8. Metrics — darts documentation. In: Darts \[Internet\]. \[cited 7 Mar 2023\]. Available: [https://unit8co.github.io/darts/generated\_api/darts.metrics.metrics.html?highlight=dtw\#darts.metrics.metrics.dtw\_metric](<https://unit8co.github.io/darts/generated_api/darts.metrics.metrics.html?highlight=dtw#darts.metrics.metrics.dtw_metric >)

[^/concepts/timeseries-data.dtw/#bib-Petitjean2011-sj]: Petitjean F., Ketterlin A., Gan\\c carski P.. A global averaging method for dynamic time warping, with applications to clustering. *Pattern recognition*. 2011. [https://www.sciencedirect.com/science/article/pii/S003132031000453X](<https://www.sciencedirect.com/science/article/pii/S003132031000453X>)
