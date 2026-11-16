# Is the Difference Real?

> [!NOTE]
>
> **In one minute**
>
> One model scores 0.412 and another 0.405. Whether that gap means anything depends on how much the scores move when the experiment is repeated. The page covers run-to-run variance, the Diebold-Mariano test for two forecasts on one series, the Wilcoxon signed-ranks test across many datasets, and corrections for multiple comparisons.
>
> **Read first:** [Time Series Forecasting Evaluation](<https://dl.leima.is/time-series/timeseries-evaluation.forecasting/index.md>) · [Time Series Forecasting Metrics](<https://dl.leima.is/time-series/timeseries-metrics.forecasting/index.md>)

A results table says that model A has an error of 0.405 and model B has 0.412. A is in bold. The table does not say what would happen with another random seed, another test period or another set of series. If the answer is "B would be in bold about half of the time", the table has shown nothing.

This page is about the size of a difference relative to its noise. Which tool applies depends on what is being compared and what can be repeated.

```
flowchart TD
    Q["Two results differ.<br/>Is it more than noise?"] --> R{"What was repeated?"}
    R -->|"training runs<br/>(seeds, splits)"| V["Run-to-run variance"]
    R -->|"forecasts over time,<br/>one series"| D["Diebold-Mariano test"]
    R -->|"datasets or series"| W["Wilcoxon signed-ranks test"]
    V --> M{"More than<br/>two methods?"}
    D --> M
    W --> M
    M -->|yes| C["Correct for<br/>multiple comparisons"]
```

> [!NOTE]
>
> **How This Page Is Organized**
>
> As in [Evaluating Probabilistic Forecasts](<https://dl.leima.is/time-series/timeseries-evaluation.probabilistic/index.md>), each concept is treated in the same five steps: **the idea**, **the math**, **the picture**, **try it**, **watch out**.
>
> | Concept | Compares | The sample is |
> | --- | --- | --- |
> | Run-to-run variance | two training procedures | repeated runs |
> | Diebold-Mariano test | two sets of forecasts | time steps |
> | Wilcoxon signed-ranks test | two methods | datasets |
> | Multiple comparisons | many methods | many tests at once |

Scores are errors: lower is better.

## A Score Is a Random Variable

**The idea.** Training a deep learning model involves choices that nobody intends to matter: the random initial weights, the order of the batches, which observations ended up in the training and the test set. Change any of them and the test score changes. The reported score is one draw from a distribution, and two models can only be told apart when the gap between them is large compared with the width of those distributions.

**The math.** Bouthillier and coauthors measured these sources of variation on five deep learning tasks, randomizing one source at a time with 200 seeds each. Their findings[^/time-series/timeseries-evaluation.significance/#bib-Bouthillier2021-var]:

- The sampling of the data, measured with bootstrap resamples, was the largest source of variance.
- The weight initialization, the source most often varied in papers, was generally below half of the variance from data sampling, and about as large as the effect of the data order.
- For classification, the variance from data sampling matched a simple binomial model of a test set of finite size: a small test set by itself limits how small a difference can be detected.

Suppose a score has standard deviation \\(\\sigma\\) from run to run. The average of \\(k\\) runs has standard error \\(\\sigma / \\sqrt{k}\\). For two methods compared on \\(k\\) runs each, the paper gives the smallest difference of averages that a \\(z\\)-test can call at the 5% level,

\\\[ z\_{0.05} \\sqrt{\\frac{\\sigma\_A^2 + \\sigma\_B^2}{k}}, \\\]

and notes that a smaller observed difference could be due to noise alone[^/time-series/timeseries-evaluation.significance/#bib-Bouthillier2021-var]. With \\(k = 1\\), the usual single run, that threshold is \\(z\_{0.05} \\sqrt{2}\\, \\sigma\\).

Instead of comparing averages against a threshold, the authors propose to estimate how often one method beats the other,

\\\[ P(A \> B) = \\frac{1}{k} \\sum\_{i=1}^{k} \\mathbb 1 \\{ A \\text{ beats } B \\text{ on pair } i \\}, \\\]

over \\(k\\) pairs of runs, and to call A better only when this probability is both significantly above 0.5 and above a threshold \\(\\gamma\\) that says what counts as a meaningful improvement; they report that \\(\\gamma = 0.75\\) worked well in their case studies[^/time-series/timeseries-evaluation.significance/#bib-Bouthillier2021-var].

**The picture.** If the two score distributions are normal with the same \\(\\sigma\\) and means that differ by \\(\\Delta\\), two formulas follow directly: a single run of A beats a single run of B with probability \\(\\Phi\\left( \\Delta / (\\sigma \\sqrt 2) \\right)\\), and the method that is truly worse has the better \\(k\\)-run average with probability \\(\\Phi\\left( -\\Delta \\sqrt{k} / (\\sigma \\sqrt 2) \\right)\\). The widget draws both distributions, one experiment of \\(k\\) runs per model, and the distribution of the difference of the two averages.

**Try it.** Press Play to repeat the experiment and count how often B looks better although A is the better model. Then raise the number of runs.

An interactive chart of two models whose test errors vary from run to run. Sliders set the true advantage of model A, the run-to-run spread and the number of runs per model; repeated experiments are drawn as dots on the distribution of the difference of the two averages. It needs JavaScript. In short: with a true advantage of 0.05, a spread of 0.10 and 3 runs per model, the worse model has the better average in about 27% of experiments; with 30 runs, in about 3%.

**Watch out.**

- Varying only the seed of the weight initialization understates the variance, because it leaves the larger source, the data sampling, fixed. The paper's first recommendation is to randomize as many sources of variation as possible, including data splits and the hyperparameter search[^/time-series/timeseries-evaluation.significance/#bib-Bouthillier2021-var].
- A fixed held-out test set hides the uncertainty from data sampling. The authors recommend resampling, with several random splits, over a single fixed test set when small improvements have to be detected[^/time-series/timeseries-evaluation.significance/#bib-Bouthillier2021-var].
- A threshold on the difference of averages that ignores the variance, such as "better by at least 0.5 points", does not separate real effects from chance[^/time-series/timeseries-evaluation.significance/#bib-Bouthillier2021-var].
- With enough runs, any tiny difference becomes statistically significant. Significant and meaningful are two separate questions, which is why the proposal above has two conditions.

## Two Forecasts on One Series: The Diebold-Mariano Test

**The idea.** In forecasting, the natural repetition is time: two forecasters each produce a forecast for every time step of a test period. At each step one of them loses less than the other. If the two are equally good, these loss differences scatter around zero. The Diebold-Mariano test asks whether their average is further from zero than chance allows. The complication is that forecast errors at neighbouring time steps are correlated, so there is less independent information in the test period than its length suggests.

**The math.** Let \\(e\_{1t}\\) and \\(e\_{2t}\\) be the errors of the two forecasts at time \\(t\\), and \\(g\\) a loss function, for example \\(g(e) = e^2\\) or \\(g(e) = \\lvert e \\rvert\\). The loss differential is

\\\[ d\_t = g(e\_{1t}) - g(e\_{2t}), \\qquad \\bar d = \\frac{1}{T} \\sum\_{t=1}^{T} d\_t. \\\]

The null hypothesis is equal expected loss, \\(\\mathbb E\\, d\_t = 0\\). Diebold and Mariano show that, when \\(d\_t\\) is covariance stationary, the statistic[^/time-series/timeseries-evaluation.significance/#bib-Diebold1995-dm]

\\\[ \\operatorname{DM} = \\frac{\\bar d}{\\sqrt{\\widehat{V} / T}} \\\]

is asymptotically standard normal under the null, where \\(\\widehat V\\) estimates the long-run variance of \\(d\_t\\). With independent \\(d\_t\\), \\(\\widehat V\\) would be the sample variance \\(\\hat\\gamma\_d(0)\\). With serial correlation the autocovariances enter,

\\\[ \\widehat V = \\hat\\gamma\_d(0) + 2 \\sum\_{\\tau = 1}^{h - 1} \\hat\\gamma\_d(\\tau), \\qquad \\hat\\gamma\_d(\\tau) = \\frac{1}{T} \\sum\_{t = \\tau + 1}^{T} (d\_t - \\bar d)(d\_{t - \\tau} - \\bar d). \\\]

The number of autocovariances comes from the forecasting problem: the errors of optimal \\(h\\)-step-ahead forecasts are correlated up to lag \\(h - 1\\) and not beyond, and the authors take this as the benchmark[^/time-series/timeseries-evaluation.significance/#bib-Diebold1995-dm]. This is the uniform lag window of the paper, written out. The test rejects equal accuracy at the 5% level when \\(\\lvert \\operatorname{DM} \\rvert \> 1.96\\).

Diebold later summarized the test as "simply an asymptotic z-test of the hypothesis that the mean of an observed series (the loss differential) is zero", with a standard error that is robust to serial correlation[^/time-series/timeseries-evaluation.significance/#bib-Diebold2012-dm].

**The picture.** The widget plots a loss differential series for forecasts at horizon \\(h\\): a bar above zero is a time step where forecast 1 lost more. The thick line is \\(\\bar d\\). Longer horizons give longer runs of bars on the same side, which is the serial correlation the standard error has to account for.

**Try it.** Keep the true mean difference at 0, set the horizon to 4 or more, and press "Repeat 2000 times". Both tests are run on 2000 fresh samples, and the bars show how often each one declares a difference that does not exist.

An interactive chart of a simulated loss differential between two forecasts over time, with its mean, the Diebold-Mariano statistic and a z statistic that ignores serial correlation. Sliders set the true mean difference, the forecast horizon and the number of forecasts. A button repeats the experiment 2000 times and shows how often each test rejects. It needs JavaScript. In short: with no true difference and a horizon of 4, the test that ignores serial correlation rejects in roughly a third of the samples and not in 5%, while the Diebold-Mariano test stays much closer to 5%.

**Watch out.**

- The test compares forecasts, not models. Its only assumption is on the loss differential, which makes it usable for forecasts from any source, including forecasts with no model behind them. Diebold warns that using it to select between models estimated on the same data is a different problem with different answers[^/time-series/timeseries-evaluation.significance/#bib-Diebold2012-dm].
- The assumption is that the loss differential is covariance stationary. Diebold suggests checking it the direct way: plot the loss differential and look at its autocorrelations[^/time-series/timeseries-evaluation.significance/#bib-Diebold2012-dm].
- With the uniform window, \\(\\widehat V\\) can come out negative. Diebold and Mariano treat that case as a rejection, and mention a Bartlett window as the alternative when a nonnegative estimate is required[^/time-series/timeseries-evaluation.significance/#bib-Diebold1995-dm].
- The normal distribution is a large-sample result. For short test periods Diebold and Mariano turn to exact tests on the same loss differentials: the sign test and the Wilcoxon signed-rank test[^/time-series/timeseries-evaluation.significance/#bib-Diebold1995-dm]. The second is the subject of the next section.

## Many Datasets: The Wilcoxon Signed-Ranks Test

**The idea.** A method is often evaluated on many datasets, or on many series. On each one, method 2 is better or worse than method 1 by some amount. Averaging these amounts is tempting and fragile: errors on different datasets are on different scales, and one dataset with a huge difference can decide the average. The Wilcoxon signed-ranks test keeps only the order of the differences. It asks whether the large differences tend to be on one side.

**The math.** Following Demšar, let \\(d\_i\\) be the difference between the scores of the two methods on the \\(i\\)-th of \\(N\\) datasets. Rank the differences by absolute value, with average ranks for ties, and sum the ranks on each side[^/time-series/timeseries-evaluation.significance/#bib-Demsar2006-sc]:

\\\[ R^+ = \\sum\_{d\_i \> 0} \\operatorname{rank}(d\_i) + \\frac{1}{2} \\sum\_{d\_i = 0} \\operatorname{rank}(d\_i), \\qquad R^- = \\sum\_{d\_i \< 0} \\operatorname{rank}(d\_i) + \\frac{1}{2} \\sum\_{d\_i = 0} \\operatorname{rank}(d\_i). \\\]

The test statistic is \\(T = \\min(R^+, R^-)\\). If the two methods are equivalent, the ranks fall on either side at random and the two sums are similar; a small \\(T\\) is evidence against that. For small \\(N\\) the critical values come from a table. For larger \\(N\\),

\\\[ z = \\frac{T - \\frac{1}{4} N (N + 1)}{\\sqrt{\\frac{1}{24} N (N + 1)(2N + 1)}} \\\]

is approximately standard normal, and the null hypothesis is rejected at the 5% level when \\(z \< -1.96\\)[^/time-series/timeseries-evaluation.significance/#bib-Demsar2006-sc].

Demšar recommends this test over the paired \\(t\\)-test for comparing two classifiers over multiple datasets, for three reasons: the \\(t\\)-test assumes that the differences are commensurable across datasets, it needs normally distributed differences unless there are about 30 datasets or more, and it is affected by outliers. The Wilcoxon test assumes commensurability only qualitatively: larger differences count more, but their magnitudes are ignored[^/time-series/timeseries-evaluation.significance/#bib-Demsar2006-sc].

**The picture.** The widget uses the worked example from Demšar's paper: the AUC of the C4.5 decision tree with and without tuning one of its parameters, on 14 datasets. The differences are sorted by absolute size, and the rank of each goes to the side of its sign. For these data \\(R^+ = 93\\) and \\(R^- = 12\\), so \\(T = 12\\); the paper gives 21 as the critical value for 14 datasets at the 5% level, so the difference is significant[^/time-series/timeseries-evaluation.significance/#bib-Demsar2006-sc].

**Try it.** Press Play to watch the ranks being handed out. Then multiply the largest difference by 10 and compare what happens to the two statistics. Click rows to flip signs and find out how many datasets have to change sides before the result is no longer significant.

An interactive chart of the differences between two classifiers on 14 datasets, from Demšar (2006), sorted by absolute size and labelled with their ranks. The sums of ranks for positive and negative differences, the Wilcoxon statistic and the paired t statistic are shown. A slider enlarges the largest difference and clicking a row flips its sign. It needs JavaScript. In short: the rank sums are 93 and 12, which is significant at the 5% level; enlarging the largest difference changes no rank and so does not change the Wilcoxon result, while the t statistic drops because the standard deviation grows.

**Watch out.**

- The test needs enough datasets. Bouthillier and coauthors point out that machine learning papers typically use as few as 3 to 5 datasets, where this test has very little power[^/time-series/timeseries-evaluation.significance/#bib-Bouthillier2021-var]. With 5 datasets, even five wins out of five cannot be significant at the 5% level in a two-sided test: there are \\(2^5 = 32\\) equally likely sign patterns under the null, so the most extreme result has probability \\(2/32 \\approx 0.06\\).
- Differences should not be rounded before ranking. Rounding creates ties, and ties cost power[^/time-series/timeseries-evaluation.significance/#bib-Demsar2006-sc].
- The datasets are treated as a sample of independent problems. Many series from one source, for example the stores of one retail chain, are less independent than their number suggests.
- Counting wins is weaker still: it ignores the size of the differences altogether. Demšar discusses it as the sign test and notes that it is weaker than the Wilcoxon test[^/time-series/timeseries-evaluation.significance/#bib-Demsar2006-sc].

## Many Comparisons at Once

**The idea.** A test at the 5% level raises a false alarm in one of twenty cases where there is nothing to find. Compare a new method against ten others, or seven methods against each other in all 21 pairs, and false alarms are no longer rare events. Some of the "significant" entries in a large results table are there by chance. The remedy is to make each single test stricter, so that the chance of any false alarm in the whole family of tests stays at 5%.

**The math.** The family-wise error is the probability of making at least one false rejection among all the comparisons[^/time-series/timeseries-evaluation.significance/#bib-Demsar2006-sc]. For \\(m\\) independent tests at level \\(\\alpha\\) where no real difference exists, it is

\\\[ 1 - (1 - \\alpha)^m, \\\]

which is 40% for \\(m = 10\\) and 64% for \\(m = 20\\) at \\(\\alpha = 0.05\\). Two corrections, both described by Demšar for comparing \\(k - 1\\) methods with a control[^/time-series/timeseries-evaluation.significance/#bib-Demsar2006-sc]:

- **Bonferroni**: compare every p-value with \\(\\alpha / m\\), where \\(m\\) is the number of comparisons.
- **Holm**: sort the p-values, \\(p\_1 \\le p\_2 \\le \\cdots \\le p\_m\\). Compare \\(p\_1\\) with \\(\\alpha / m\\); if it is below, reject and compare \\(p\_2\\) with \\(\\alpha / (m - 1)\\), and so on. Stop at the first p-value that is not below its threshold and retain all remaining hypotheses.

Holm's procedure rejects everything Bonferroni rejects and possibly more, and it makes no additional assumptions, so Demšar describes it as the more powerful of the two[^/time-series/timeseries-evaluation.significance/#bib-Demsar2006-sc].

When all methods are compared with each other over many datasets, Demšar recommends starting with the Friedman test, which works on the ranks of the methods within each dataset, followed by a post-hoc test. In the Nemenyi test, two of \\(k\\) methods compared on \\(N\\) datasets differ significantly when their average ranks differ by at least the critical difference

\\\[ \\operatorname{CD} = q\_\\alpha \\sqrt{\\frac{k (k + 1)}{6 N}}, \\\]

with \\(q\_\\alpha\\) from a table in the paper[^/time-series/timeseries-evaluation.significance/#bib-Demsar2006-sc]. The same paper shows that the tests have much more power when all methods are compared with one control than when all pairs are compared, and concludes that pairwise comparisons should not be made when the question is only whether a new method is better than the existing ones.

**The picture.** Each cell of the widget is one comparison with its p-value. A filled orange cell is a false discovery: declared significant although the two methods do not differ. Outlined blue cells are comparisons with a real difference, filled when the procedure detects them.

**Try it.** With no real differences, press Play and watch how often at least one cell lights up without a correction. Then add a few real differences and compare how many of them Bonferroni and Holm still find.

An interactive grid of p-values from many comparisons made at once. Sliders set the number of comparisons and how many of them have a real difference; buttons choose no correction, the Bonferroni correction or Holm's procedure. Bars show in what share of repeated experiments at least one false discovery occurs under each procedure. It needs JavaScript. In short: with 10 comparisons and no real differences, at least one comes out significant at the 5% level in about 40% of experiments without a correction, and in about 5% with either correction.

**Watch out.**

- Corrections cost power: with many comparisons, real differences are missed more often. Fewer, planned comparisons are better than all possible ones.
- The comparisons that were tried and not reported count as well. A hyperparameter search that evaluates a hundred configurations on the test set and reports the best is a multiple comparison without a correction.
- Bouthillier and coauthors note the limit of the approach for leaderboards: with hundreds of entries, corrections become so strict that the choice of a winner is necessarily somewhat arbitrary, and a different test set might have given a different ranking[^/time-series/timeseries-evaluation.significance/#bib-Bouthillier2021-var].

## In Practice

1. Decide what the unit of repetition is: runs, time steps or datasets. That decides the tool.
2. Report a measure of spread next to every score: the standard deviation over runs, or a confidence interval.
3. Randomize more than the weight initialization when the budget allows it: data splits first.
4. For two forecasters on one test period, look at the loss differential over time before testing it.
5. For many datasets, compare with ranks, and say how many datasets there were.
6. Count the comparisons that were made, including the ones left out of the table.
7. Report the size of the difference, not only whether it is significant, with a [baseline](<https://dl.leima.is/time-series/timeseries-evaluation.baselines/index.md>) for scale.

---

**Related:** [Baselines and Skill Scores](<https://dl.leima.is/time-series/timeseries-evaluation.baselines/index.md>) · [Evaluating Probabilistic Forecasts](<https://dl.leima.is/time-series/timeseries-evaluation.probabilistic/index.md>)

[^/time-series/timeseries-evaluation.significance/#bib-Bouthillier2021-var]: Bouthillier X., Delaunay P., Bronzi M. et al.. Accounting for Variance in Machine Learning Benchmarks. 2021. [http://arxiv.org/abs/2103.03098](<http://arxiv.org/abs/2103.03098>)

[^/time-series/timeseries-evaluation.significance/#bib-Diebold1995-dm]: Diebold F.X., Mariano R.S.. Comparing Predictive Accuracy. 1994. [https://www.nber.org/papers/t0169](<https://www.nber.org/papers/t0169>)

[^/time-series/timeseries-evaluation.significance/#bib-Diebold2012-dm]: Diebold F.X.. Comparing Predictive Accuracy, Twenty Years Later: A Personal Perspective on the Use and Abuse of Diebold-Mariano Tests. 2012. [http://www.nber.org/papers/w18391](<http://www.nber.org/papers/w18391>)

[^/time-series/timeseries-evaluation.significance/#bib-Demsar2006-sc]: Dem\\v sar J.. Statistical Comparisons of Classifiers over Multiple Data Sets. *Journal of Machine Learning Research*. 2006. [https://www.jmlr.org/papers/v7/demsar06a.html](<https://www.jmlr.org/papers/v7/demsar06a.html>)
