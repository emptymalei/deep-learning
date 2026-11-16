# DTW Barycenter Averaging

> [!NOTE]
>
> **In one minute**
>
> DTW barycenter averaging (DBA), proposed by Petitjean et al., averages a set of time series: it constructs the series that is their barycenter under the dynamic time warping distance.
>
> **Read first:** [Dynamic Time Warping (**DTW**)](<https://dl.leima.is/concepts/timeseries-data.dtw/index.md>)

[DTW](<https://dl.leima.is/concepts/timeseries-data.dtw/index.md>) Barycenter Averaging (**DBA**) constructs a series \\(\\bar{\\mathcal S}\\) out of a set of series \\(\\{\\mathcal S^{(\\alpha)}\\}\\) so that \\(\\bar{\\mathcal S}\\) is the barycenter of \\(\\{\\mathcal S^{(\\alpha)}\\}\\) measured by Dynamic Time Warping (**DTW**) distance [^/concepts/timeseries-data.dtw-barycenter-averaging/#bib-Petitjean2011-sj].

## Barycenter Averaging Based on DTW Distance

Petitjean et al proposed a time series averaging algorithm based on DTW distance which is dubbed DTW Barycenter Averaging (**DBA**).

> [!NOTE]
>
> **DBA Implementation**
>
> [https://github.com/fpetitjean/DBA](<https://github.com/fpetitjean/DBA>)

---

**Related:** [Data Augmentation for Time Series](<https://dl.leima.is/time-series/timeseries-data.data-augmentation/index.md>)

[^/concepts/timeseries-data.dtw-barycenter-averaging/#trekhleb]: trekhleb. javascript-algorithms/src/algorithms/string/levenshtein-distance at master · trekhleb/javascript-algorithms. In: GitHub \[Internet\]. \[cited 27 Jul 2022\]. Available: [https://github.com/trekhleb/javascript-algorithms/tree/master/src/algorithms/string/levenshtein-distance](<https://github.com/trekhleb/javascript-algorithms/tree/master/src/algorithms/string/levenshtein-distance >)

[^/concepts/timeseries-data.dtw-barycenter-averaging/#bib-Petitjean2011-sj]: Petitjean F., Ketterlin A., Gan\\c carski P.. A global averaging method for dynamic time warping, with applications to clustering. *Pattern recognition*. 2011. [https://www.sciencedirect.com/science/article/pii/S003132031000453X](<https://www.sciencedirect.com/science/article/pii/S003132031000453X>)
