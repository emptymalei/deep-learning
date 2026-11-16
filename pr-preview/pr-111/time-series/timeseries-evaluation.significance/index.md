# Is the Difference Real?

> [!NOTE]
>
> **In one minute**
>
> One model scores 0.412 and another 0.405. Whether that gap means anything depends on how much the scores move when the experiment is repeated. Drawing on more than forty papers, the page covers run-to-run variance, repeated splits that overlap, the Diebold-Mariano test, the inflation of the best of many models, the Wilcoxon signed-ranks test, multiple comparisons, and the difference between a significant result and a large one.
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
> | Overlapping replicates | two learning algorithms | repeated splits of one dataset |
> | Diebold-Mariano test | two sets of forecasts | time steps |
> | The best of many | one winner against a benchmark | all the models that were tried |
> | Wilcoxon signed-ranks test | two methods | datasets |
> | Multiple comparisons | many methods | many tests at once |
> | Size and power | a claimed gain against its noise | the experiment itself, repeated in thought |

## Three Questions before Any Test

The literature on this topic grew up separately in econometrics, in machine learning and, since about 2017, in deep learning, where many of the older results were found again for random seeds. Read together, every method answers the same three questions. The arrangement below is this page's own.

1. **What was repeated?** A result is one draw, and a comparison needs repeated draws of something: time steps, forecast origins, test examples, train and test splits, training runs, datasets, or the models that were tried.
2. **How dependent are the repetitions?** Nearly every test on this page is an older test with its variance repaired for a dependence that had been ignored.
3. **What statement is wanted?** A decision to reject or not, an interval, a set of models that can not be told apart, or a probability that the difference is large enough to matter.

| What is compared | Repeated: time steps | Repeated: splits or runs | Repeated: datasets | Repeated: models tried |
| --- | --- | --- | --- | --- |
| two forecasts, as given | Diebold-Mariano[^/time-series/timeseries-evaluation.significance/#bib-Diebold1995-dm] |  | Wilcoxon signed-ranks[^/time-series/timeseries-evaluation.significance/#bib-Demsar2006-sc] |  |
| two forecasting methods, estimation included | conditional predictive ability[^/time-series/timeseries-evaluation.significance/#bib-Giacomini2003-cpa]; tests for nested models[^/time-series/timeseries-evaluation.significance/#bib-Clark1999-nested] |  |  |  |
| two learning algorithms |  | 5x2 cross-validation[^/time-series/timeseries-evaluation.significance/#bib-Dietterich1997-tests]; corrected resampled t-test[^/time-series/timeseries-evaluation.significance/#bib-Nadeau1999-inference] | Wilcoxon, sign test; Bayesian signed-rank test[^/time-series/timeseries-evaluation.significance/#bib-Benavoli2017-bayes] |  |
| many models against a benchmark |  |  | Friedman test with corrections[^/time-series/timeseries-evaluation.significance/#bib-Demsar2006-sc][^/time-series/timeseries-evaluation.significance/#bib-Garcia2008-pairwise] | reality check[^/time-series/timeseries-evaluation.significance/#bib-White2000-rc]; superior predictive ability[^/time-series/timeseries-evaluation.significance/#bib-Hansen2005-spa] |
| many models, no benchmark |  |  | pairwise tests with corrections[^/time-series/timeseries-evaluation.significance/#bib-Benavoli2016-ranks] | model confidence set[^/time-series/timeseries-evaluation.significance/#bib-Hansen2011-mcs]; false discovery rate[^/time-series/timeseries-evaluation.significance/#bib-Benjamini1995-fdr] |

Scores are errors: lower is better.

## A Score Is a Random Variable

**The idea.** Training a deep learning model involves choices that nobody intends to matter: the random initial weights, the order of the batches, which observations ended up in the training and the test set. Change any of them and the test score changes. The reported score is one draw from a distribution, and two models can only be told apart when the gap between them is large compared with the width of those distributions.

**The math.** Bouthillier and coauthors measured these sources of variation on five deep learning tasks, randomizing one source at a time with 200 seeds each. Their findings[^/time-series/timeseries-evaluation.significance/#bib-Bouthillier2021-var]:

- The sampling of the data, measured with bootstrap resamples, was the largest source of variance.
- The weight initialization, the source most often varied in papers, was generally below half of the variance from data sampling, and about as large as the effect of the data order.
- For classification, the variance from data sampling matched a simple binomial model of a test set of finite size: a small test set by itself limits how small a difference can be detected.

Other studies put numbers on single sources:

| Study | What was varied | Spread found |
| --- | --- | --- |
| Reimers and Gurevych[^/time-series/timeseries-evaluation.significance/#bib-Reimers2017-seeds] | the seed of one published sequence tagger, 86 runs | test F1 from 89.99% to 91.00%, against a published gap of 0.27 points between two systems |
| Picard[^/time-series/timeseries-evaluation.significance/#bib-Picard2021-seed] | 10,000 seeds of an image classifier | standard deviation 0.2 points, 1.82 points between the worst and the best seed |
| Henderson and coauthors[^/time-series/timeseries-evaluation.significance/#bib-Henderson2017-matters] | ten runs of one reinforcement learning setup, split into two groups of five | the two group averages differ significantly (\\(p = 0.0016\\)) |
| Melis, Dyer and Blunsom[^/time-series/timeseries-evaluation.significance/#bib-Melis2017-eval] | seeds and non-deterministic arithmetic of tuned language models | about 0.4 to 0.5 perplexity; they treat a gap of 1.0 as robust |

The amount of tuning is a further source. Melis and coauthors found that standard LSTMs, tuned with the same automatic procedure as newer architectures, outperformed them[^/time-series/timeseries-evaluation.significance/#bib-Melis2017-eval]. Dodge and coauthors show that which of two models is better can depend on how many hyperparameter settings were tried, and propose to report the expected best validation score as a function of that budget. For \\(n\\) random trials with scores \\(V\_1, \\cdots, V\_n\\) and \\(V\_n^\* = \\max\_i V\_i\\)[^/time-series/timeseries-evaluation.significance/#bib-Dodge2019-show],

\\\[ P(V\_n^\* \\le v) = P(V \\le v)^n , \\\]

so the expected maximum can be computed from the empirical distribution of the scores for every \\(n\\).

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
- Few runs are the norm. Agarwal and coauthors found that results on a common reinforcement learning benchmark were mostly computed from 3 or 5 runs, and that reported medians moved enough between sets of runs to reverse published orderings[^/time-series/timeseries-evaluation.significance/#bib-Agarwal2021-precipice].
- Tests on few runs raise false alarms. Colas, Sigaud and Oudeyer took two samples of five runs that both tests they tried declared different, and the two samples came from the same algorithm[^/time-series/timeseries-evaluation.significance/#bib-Colas2018-seeds].
- A fixed seed gives reproducibility and no information: there is no reason to prefer one seed over another, so the comparison should be between distributions of scores[^/time-series/timeseries-evaluation.significance/#bib-Reimers2017-seeds].
- With enough runs, any tiny difference becomes statistically significant. Significant and meaningful are two separate questions, which is why the proposal above has two conditions; see [Size and Power](<#significant-is-not-large-effect-size-and-power>) below.

## Overlapping Replicates: Why the Naive Standard Error Is Too Small

**The idea.** With one dataset and no budget for many independent experiments, the usual way to get repetitions is to split the data several times: ten random train and test splits, or ten folds. The ten results look like ten measurements. They are not ten independent measurements, because the training sets share most of their data, and the test sets are drawn from the same pool. A standard error computed as if they were independent is too small, and a t-test built on it finds differences that are not there.

**The math.** Nadeau and Bengio work with \\(J\\) random splits into \\(n\_1\\) training and \\(n\_2\\) test examples. If each split result has variance \\(\\sigma^2\\) and any two of them have correlation \\(\\rho\\), the variance of their mean is[^/time-series/timeseries-evaluation.significance/#bib-Nadeau1999-inference]

\\\[ \\operatorname{Var}(\\hat\\mu) = \\sigma^2 \\left( \\rho + \\frac{1 - \\rho}{J} \\right), \\\]

which does not go to zero as \\(J\\) grows: it levels off at \\(\\sigma^2 \\rho\\). The sample variance \\(S^2\\) of the split results underestimates \\(\\sigma^2\\) at the same time, and the two effects together call for the factor \\(\\frac{1}{J} + \\frac{\\rho}{1 - \\rho}\\) in place of \\(\\frac{1}{J}\\). Since \\(\\rho\\) is hard to estimate, they use the share of data held out as a stand-in, \\(\\rho\_0 = \\frac{n\_2}{n\_1 + n\_2}\\), which gives the corrected resampled t-test. For the differences \\(x\_j\\) between two algorithms over \\(n\\) runs[^/time-series/timeseries-evaluation.significance/#bib-Bouckaert2004-replic],

\\\[ t = \\frac{ \\frac{1}{n} \\sum\_{j=1}^{n} x\_j }{ \\sqrt{ \\left( \\frac{1}{n} + \\frac{n\_2}{n\_1} \\right) \\hat\\sigma^2 } }, \\\]

with \\(n - 1\\) degrees of freedom.

The same repair appears wherever replicates share something:

| Replicates | What they share | Finding |
| --- | --- | --- |
| cross-validation folds | most of the training data | the 10-fold paired t-test rejects up to twice as often as it should; a t-test on repeated random splits should never be used[^/time-series/timeseries-evaluation.significance/#bib-Dietterich1997-tests] |
| any K-fold cross-validation | the same dataset | no unbiased estimator of its variance exists[^/time-series/timeseries-evaluation.significance/#bib-Bengio2004-novar]; the usual intervals miss two to three times more often than stated[^/time-series/timeseries-evaluation.significance/#bib-Bates2021-cv] |
| loss differentials over time | serial correlation | the long-run variance in the Diebold-Mariano test of the next section |
| test questions about one passage | the passage | standard errors that account for the clusters can be several times the naive ones[^/time-series/timeseries-evaluation.significance/#bib-Miller2024-errorbars] |

**The picture.** The widget compares two algorithms of exactly equal quality on \\(J\\) random splits, thousands of times, and counts how often each test declares a difference. A sound test at the 5% level should do so 5% of the time.

**Try it.** Press Play to increase the number of splits. Then change the share of the data that is held out.

An interactive chart of the false alarm rate of two t-tests applied to the results of J random train and test splits, when the two algorithms compared are equally good. One line is the ordinary t-test, the other the corrected resampled t-test of Nadeau and Bengio. Sliders set the share of the data used for testing and the number of splits. It needs JavaScript. In short: the ordinary test declares a difference far more often than 5%, and more often the more splits are used; the corrected test stays near 5%.

**Watch out.**

- More splits do not add data. They make the ordinary test more confident about the same sample, which is why its false alarm rate grows with the number of splits.
- The correction uses a stand-in for an unknown correlation and can be off in either direction[^/time-series/timeseries-evaluation.significance/#bib-Nadeau1999-inference].
- Replicability is a criterion of its own. Bouckaert and Frank repeated each test ten times on the same data with different random partitions and counted how often the verdict was the same every time. In the worst case this happened on 17% of datasets for 5x2 cross-validation and on 77% for the corrected test on ten times ten-fold cross-validation[^/time-series/timeseries-evaluation.significance/#bib-Bouckaert2004-replic].
- Dietterich's alternative, five repetitions of two-fold cross-validation, keeps its error rate and was later found to have little power[^/time-series/timeseries-evaluation.significance/#bib-Dietterich1997-tests][^/time-series/timeseries-evaluation.significance/#bib-Nadeau1999-inference].
- These tests compare algorithms on training sets smaller than the full dataset, and a single held-out test set does not capture the variation from the choice of training set at all[^/time-series/timeseries-evaluation.significance/#bib-Dietterich1997-tests].
- For a time series, the repetitions are forecast origins. A rolling origin gives several test periods from one series, and they overlap in the same way[^/time-series/timeseries-evaluation.significance/#bib-Hewamalage2022-sc][^/time-series/timeseries-evaluation.significance/#bib-Cerqueira2019-vt].

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
- Nested models need other critical values. When one model contains the other, the two forecast errors coincide under the null hypothesis, the variance in the denominator degenerates, and the statistic compared with normal critical values rejects too rarely; Clark and McCracken derive the appropriate distributions[^/time-series/timeseries-evaluation.significance/#bib-Clark1999-nested].
- There are two different null hypotheses about models. One asks whether two models would be equally accurate with their parameters known exactly; the other asks whether two forecasting methods, with the estimation they involve, are equally accurate at the sample size at hand. For nested models the two can give opposite answers[^/time-series/timeseries-evaluation.significance/#bib-Clark2011-advances]. Giacomini and White test the second, and condition on the information available at the forecast origin: under their null, the forecaster can not predict which of the two methods will be more accurate next[^/time-series/timeseries-evaluation.significance/#bib-Giacomini2003-cpa].
- A small-sample modification of the statistic by Harvey, Leybourne and Newbold is treated as standard practice in the econometric literature[^/time-series/timeseries-evaluation.significance/#bib-Clark2011-advances][^/time-series/timeseries-evaluation.significance/#bib-Hansen2005-spa]. Its formula is not given here because that paper could not be opened for this page.
- The normal distribution is a large-sample result. For short test periods Diebold and Mariano turn to exact tests on the same loss differentials: the sign test and the Wilcoxon signed-rank test[^/time-series/timeseries-evaluation.significance/#bib-Diebold1995-dm]. The second is the subject of the next section.

## The Best of Many: Searching Inflates the Winner

**The idea.** Try twenty models that are all exactly as good as the benchmark. Each one's measured advantage is noise around zero, and the largest of twenty noisy numbers is reliably positive. Reported alone, the winner looks like an improvement. White calls this data snooping: a given set of data is used more than once for inference or model selection, so that a good result may simply be due to chance. For time series he calls it practically unavoidable, because there is only one history[^/time-series/timeseries-evaluation.significance/#bib-White2000-rc]. The same mechanism operates over hyperparameter settings, over random seeds, and over the research groups of a field.

**The math.** Let \\(f\_k\\) be the performance of model \\(k\\) relative to the benchmark, with mean \\(\\bar f\_k\\) over \\(n\\) test periods, for \\(k = 1, \\cdots, l\\). White tests the hypothesis that even the best model has no advantage[^/time-series/timeseries-evaluation.significance/#bib-White2000-rc],

\\\[ H\_0 : \\max\_{k = 1, \\cdots, l} \\mathbb E (f\_k) \\le 0, \\qquad \\bar V\_l = \\max\_{k = 1, \\cdots, l} \\sqrt{n} \\, \\bar f\_k , \\\]

and obtains the distribution of the statistic \\(\\bar V\_l\\) by resampling the test periods in blocks, with all \\(l\\) models kept together so that their correlation is preserved. The p-value of this reality check answers the right question: how often would the best of \\(l\\) useless models look this good?

Two refinements followed.

- Hansen's test for superior predictive ability divides each \\(\\bar f\_k\\) by its standard error and stops clearly inferior models from inflating the critical value. He shows that the reality check can be manipulated by adding poor and irrelevant models to the comparison[^/time-series/timeseries-evaluation.significance/#bib-Hansen2005-spa].
- The model confidence set of Hansen, Lunde and Nason needs no benchmark. It removes the worst model, tests whether the remaining ones are equally good, and repeats until the test no longer rejects. What is left contains the best model with a given confidence, like a confidence interval for a parameter. Uninformative data leave many models in the set, informative data few[^/time-series/timeseries-evaluation.significance/#bib-Hansen2011-mcs].

**The picture.** The widget draws the test statistic of the winner over thousands of repeated searches, each over models with no true advantage. The dashed curve is what a single model chosen in advance would show.

**Try it.** Press Play to increase the number of models tried from 1 to 200. Then raise the correlation between the models.

An interactive histogram of the test statistic of the best of many models that are all no better than the benchmark, over 3,000 repeated searches, against the standard normal curve for a single model. Sliders set the number of models tried and the correlation between them. It needs JavaScript. In short: with 20 independent models the winner passes a 5% test meant for one model in about 64% of the searches; with highly correlated models the inflation is much smaller.

**Watch out.**

- Correlated models cost less. In White's example the p-value erodes slowly over long stretches of the search, because the models' forecasts are highly correlated[^/time-series/timeseries-evaluation.significance/#bib-White2000-rc]. Twenty variants of one architecture are not twenty independent tries.
- The search includes what is not reported. Picard argues that the thousands of papers submitted on one benchmark each year amount to a scan of the seed landscape, and that trial-and-error improvement of a model is an implicit one[^/time-series/timeseries-evaluation.significance/#bib-Picard2021-seed]. Demšar makes the point for whole fields: among groups trying similar ideas of equal merit, the one that publishes is the one with the most luck in its datasets and samples[^/time-series/timeseries-evaluation.significance/#bib-Demsar2008-appropriate].
- The winner of a hyperparameter search is the same case: the expected best of \\(n\\) trials grows with \\(n\\), so two models are comparable only at the same budget[^/time-series/timeseries-evaluation.significance/#bib-Dodge2019-show].
- Leaderboards add the adaptive version, where each new submission is shaped by earlier scores; see [Leakage and Contamination](<https://dl.leima.is/time-series/timeseries-evaluation.leakage/index.md#reuse-the-test-set-wears-out>).
- The model confidence set says nothing about models that were not in the candidate set[^/time-series/timeseries-evaluation.significance/#bib-Hansen2011-mcs].

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
- The same data can be read without a null hypothesis. Benavoli and coauthors, among them the author of the paper this section follows, propose a Bayesian version of the signed-rank test with a region of practical equivalence: differences below 1% of accuracy count as no difference, and the analysis returns three probabilities, that one method is practically better, that the other is, and that they are equivalent[^/time-series/timeseries-evaluation.significance/#bib-Benavoli2017-bayes].
- Counting wins is weaker still: it ignores the size of the differences altogether. Demšar discusses it as the sign test and notes that it is weaker than the Wilcoxon test[^/time-series/timeseries-evaluation.significance/#bib-Demsar2006-sc].

## Many Comparisons at Once

**The idea.** A test at the 5% level raises a false alarm in one of twenty cases where there is nothing to find. Compare a new method against ten others, or seven methods against each other in all 21 pairs, and false alarms are no longer rare events. Some of the "significant" entries in a large results table are there by chance. The remedy is to make each single test stricter, so that the chance of any false alarm in the whole family of tests stays at 5%.

**The math.** The family-wise error is the probability of making at least one false rejection among all the comparisons[^/time-series/timeseries-evaluation.significance/#bib-Demsar2006-sc]. For \\(m\\) independent tests at level \\(\\alpha\\) where no real difference exists, it is

\\\[ 1 - (1 - \\alpha)^m, \\\]

which is 40% for \\(m = 10\\) and 64% for \\(m = 20\\) at \\(\\alpha = 0.05\\). Two corrections, both described by Demšar for comparing \\(k - 1\\) methods with a control[^/time-series/timeseries-evaluation.significance/#bib-Demsar2006-sc]:

- **Bonferroni**: compare every p-value with \\(\\alpha / m\\), where \\(m\\) is the number of comparisons.
- **Holm**: sort the p-values, \\(p\_1 \\le p\_2 \\le \\cdots \\le p\_m\\). Compare \\(p\_1\\) with \\(\\alpha / m\\); if it is below, reject and compare \\(p\_2\\) with \\(\\alpha / (m - 1)\\), and so on. Stop at the first p-value that is not below its threshold and retain all remaining hypotheses.

Holm's procedure rejects everything Bonferroni rejects and possibly more, and it makes no additional assumptions, so Demšar describes it as the more powerful of the two[^/time-series/timeseries-evaluation.significance/#bib-Demsar2006-sc].

A third procedure changes the target. Benjamini and Hochberg argue that with many comparisons the relevant error is the share of false discoveries among all discoveries, not the chance of a single false one. With \\(V\\) false and \\(S\\) true rejections, the false discovery rate is \\(\\mathbb E\\left( V / (V + S) \\right)\\). Their procedure sorts the p-values, finds the largest \\(i\\) with[^/time-series/timeseries-evaluation.significance/#bib-Benjamini1995-fdr]

\\\[ p\_{(i)} \\le \\frac{i}{m} \\, q^\*, \\\]

and rejects the hypotheses belonging to \\(p\_{(1)}, \\cdots, p\_{(i)}\\). For independent tests this keeps the false discovery rate at or below \\(q^\*\\). It rejects more than the family-wise procedures, and it is the natural choice for screening many candidates when a small share of false leads is acceptable.

When all methods are compared with each other over many datasets, Demšar recommends starting with the Friedman test, which works on the ranks of the methods within each dataset, followed by a post-hoc test. In the Nemenyi test, two of \\(k\\) methods compared on \\(N\\) datasets differ significantly when their average ranks differ by at least the critical difference

\\\[ \\operatorname{CD} = q\_\\alpha \\sqrt{\\frac{k (k + 1)}{6 N}}, \\\]

with \\(q\_\\alpha\\) from a table in the paper[^/time-series/timeseries-evaluation.significance/#bib-Demsar2006-sc]. The same paper shows that the tests have much more power when all methods are compared with one control than when all pairs are compared, and concludes that pairwise comparisons should not be made when the question is only whether a new method is better than the existing ones.

**The picture.** Each cell of the widget is one comparison with its p-value. A filled orange cell is a false discovery: declared significant although the two methods do not differ. Outlined blue cells are comparisons with a real difference, filled when the procedure detects them.

**Try it.** With no real differences, press Play and watch how often at least one cell lights up without a correction. Then add a few real differences and compare how many of them Bonferroni and Holm still find.

An interactive grid of p-values from many comparisons made at once. Sliders set the number of comparisons and how many of them have a real difference; buttons choose no correction, the Bonferroni correction or Holm's procedure. Bars show in what share of repeated experiments at least one false discovery occurs under each procedure. It needs JavaScript. In short: with 10 comparisons and no real differences, at least one comes out significant at the 5% level in about 40% of experiments without a correction, and in about 5% with either correction.

**Watch out.**

- Corrections cost power: with many comparisons, real differences are missed more often. Fewer, planned comparisons are better than all possible ones.
- The Nemenyi test and the diagrams built on it have been challenged twice. García and Herrera advise against it because it is very conservative and misses obvious differences, and recommend procedures that use the logical relations between pairwise hypotheses[^/time-series/timeseries-evaluation.significance/#bib-Garcia2008-pairwise]. Benavoli, Corani and Mangili show a deeper problem: since the ranks are computed over all methods, whether A differs significantly from B depends on which other methods are in the comparison. They recommend pairwise Wilcoxon or sign tests with a correction instead[^/time-series/timeseries-evaluation.significance/#bib-Benavoli2016-ranks].
- With very many series the critical difference becomes so small that negligible differences are significant[^/time-series/timeseries-evaluation.significance/#bib-Hewamalage2022-sc].
- "A is significantly better than the baseline, and B is not" does not show that A is better than B. Gelman and Stern's example: two estimates of 25 and 10, each with standard error 10. The first is clearly significant and the second is not, while their difference, 15 with a standard error of 14, is far from significant[^/time-series/timeseries-evaluation.significance/#bib-Gelman2006-difference].
- The comparisons that were tried and not reported count as well. A hyperparameter search that evaluates a hundred configurations on the test set and reports the best is a multiple comparison without a correction.
- Bouthillier and coauthors note the limit of the approach for leaderboards: with hundreds of entries, corrections become so strict that the choice of a winner is necessarily somewhat arbitrary, and a different test set might have given a different ranking[^/time-series/timeseries-evaluation.significance/#bib-Bouthillier2021-var].

## Significant Is Not Large: Effect Size and Power

**The idea.** A significance test answers one question: could noise alone have produced this difference? It does not say how large the difference is or whether it matters. With enough data any difference, however small, becomes significant. With too little data a real difference goes unnoticed, and there is a less obvious consequence: the experiments that do reach significance are the ones where noise happened to push the estimate up, so they overstate the effect. A field that publishes only significant results from small experiments publishes exaggerations.

**The math.** The power of a test is the probability that it rejects when a real effect of a given size exists. For an estimate \\(\\hat d\\) with standard error \\(s\\) and true value \\(d\\), tested two-sided at the 5% level,

\\\[ \\text{power} = P\\left( \\lvert \\hat d \\rvert \> 1.96 \\, s \\right) = 1 - \\Phi\\left( 1.96 - \\frac{d}{s} \\right) + \\Phi\\left( -1.96 - \\frac{d}{s} \\right) . \\\]

Gelman and Carlin add two quantities for the experiments that come out significant: the probability that the estimate has the wrong sign, and the exaggeration ratio, the expected size of a significant estimate divided by the true effect[^/time-series/timeseries-evaluation.significance/#bib-Gelman2014-typesm],

\\\[ \\text{exaggeration} = \\frac{ \\mathbb E \\left( \\lvert \\hat d \\rvert \\; \\middle\\vert \\; \\lvert \\hat d \\rvert \> 1.96 \\, s \\right) }{ d } . \\\]

At 80% power, when the true effect is 2.8 standard errors, the ratio is 1.12 and wrong signs practically do not occur. At low power both get worse quickly.

How much power do typical evaluations have?

- Card and coauthors estimated the smallest accuracy difference that the test sets of a language benchmark can detect with 80% power. For a test set of 1,725 examples it was 1.6 points, while the mean improvement reported on it was 0.6 points[^/time-series/timeseries-evaluation.significance/#bib-Card2020-power].
- Miller derives the sample size for comparing two language models on the same questions and arrives at about 1,000 questions to detect a difference of 3 points with 80% power in his example[^/time-series/timeseries-evaluation.significance/#bib-Miller2024-errorbars].
- Estimates of the needed number of runs from a small pilot are too optimistic, because standard deviations estimated from few runs are too small on average[^/time-series/timeseries-evaluation.significance/#bib-Colas2018-seeds].

**The picture.** The widget shows the distribution of the estimated difference over repeated experiments. Its unit is the standard error, so one slider covers every case. The shaded tails are the experiments that reach significance.

**Try it.** Press Play to shrink the true difference from 4 standard errors to almost none, and watch the average significant estimate stay far out while the truth moves towards zero.

An interactive chart of the sampling distribution of an estimated difference, in units of its standard error, with the regions shaded where the estimate is significant at the 5% level. A slider sets the true difference. The power, the average size of the significant estimates relative to the truth, and the share of significant estimates with the wrong sign are shown. It needs JavaScript. In short: with a true difference of 2.8 standard errors the power is 80% and significant estimates overstate the effect by a factor of about 1.1; with a true difference of 1 standard error the power is 17% and significant estimates are about 2.5 times too large.

**Watch out.**

- The statement of the American Statistical Association on p-values lists what a p-value is not: it is not the probability that the hypothesis is true, and it does not measure the size or the importance of an effect; any effect, however small, can produce a small p-value if the sample is large enough[^/time-series/timeseries-evaluation.significance/#bib-Wasserstein2016-asa].
- A significant result does not show that the experiment had high power[^/time-series/timeseries-evaluation.significance/#bib-Card2020-power].
- The effect size for a power calculation has to come from outside the experiment, not from its own noisy estimate[^/time-series/timeseries-evaluation.significance/#bib-Gelman2014-typesm].
- A non-significant difference is not evidence of equality. To claim that two methods are practically the same, an interval has to lie inside a stated region of equivalence[^/time-series/timeseries-evaluation.significance/#bib-Benavoli2017-bayes].
- The statement also asks researchers to disclose every hypothesis explored and every analysis run, since reporting only the ones that passed a threshold makes the reported p-values uninterpretable[^/time-series/timeseries-evaluation.significance/#bib-Wasserstein2016-asa].

## Beyond a Single Winner: Intervals, Sets and Curves

Several papers replace the question "who wins?" by a statement that shows how much of the ordering the data can support.

| Report | What it shows | Source |
| --- | --- | --- |
| interval for an aggregate score | resample runs within each task, recompute the aggregate, repeat | Agarwal and coauthors[^/time-series/timeseries-evaluation.significance/#bib-Agarwal2021-precipice] |
| interquartile mean | the mean of the middle half of all runs: less noisy than the median, less outlier-driven than the mean | Agarwal and coauthors[^/time-series/timeseries-evaluation.significance/#bib-Agarwal2021-precipice] |
| performance profile | for every threshold, the share of problems or runs on which a method is within that threshold | Dolan and Moré[^/time-series/timeseries-evaluation.significance/#bib-Dolan2002-profiles]; with uncertainty bands[^/time-series/timeseries-evaluation.significance/#bib-Agarwal2021-precipice] |
| probability of improvement | how often one method beats the other | Bouthillier and coauthors[^/time-series/timeseries-evaluation.significance/#bib-Bouthillier2021-var] |
| almost stochastic dominance | how far the whole distribution of one model's scores lies above the other's | Dror, Shlomov and Reichart[^/time-series/timeseries-evaluation.significance/#bib-Dror2019-dominance] |
| model confidence set | the models that can not be distinguished from the best | Hansen, Lunde and Nason[^/time-series/timeseries-evaluation.significance/#bib-Hansen2011-mcs] |
| expected best score by budget | performance as a function of the number of tuning trials | Dodge and coauthors[^/time-series/timeseries-evaluation.significance/#bib-Dodge2019-show] |

Dolan and Moré give the reasons that also apply to forecasting benchmarks: averages are dominated by a few hard problems, ranks lose the size of the differences, and medians hide what happens between the quartiles[^/time-series/timeseries-evaluation.significance/#bib-Dolan2002-profiles].

## Where the Literature Disagrees

| Question | One position | The other |
| --- | --- | --- |
| Friedman test with Nemenyi post-hoc? | recommended for many methods on many datasets[^/time-series/timeseries-evaluation.significance/#bib-Demsar2006-sc] | too conservative[^/time-series/timeseries-evaluation.significance/#bib-Garcia2008-pairwise]; the verdict on two methods depends on the others in the pool[^/time-series/timeseries-evaluation.significance/#bib-Benavoli2016-ranks] |
| Keep null hypothesis tests? | yes, used properly and with power reported[^/time-series/timeseries-evaluation.significance/#bib-Dror2018-hitchhiker][^/time-series/timeseries-evaluation.significance/#bib-Card2020-power][^/time-series/timeseries-evaluation.significance/#bib-Wasserstein2016-asa] | abandon them for Bayesian analysis with a region of equivalence[^/time-series/timeseries-evaluation.significance/#bib-Benavoli2017-bayes]; they promote lucky results[^/time-series/timeseries-evaluation.significance/#bib-Demsar2008-appropriate] |
| 5x2 cross-validation? | recommended: it keeps its error rate[^/time-series/timeseries-evaluation.significance/#bib-Dietterich1997-tests] | too little power and poor replicability; use the corrected resampled test[^/time-series/timeseries-evaluation.significance/#bib-Nadeau1999-inference][^/time-series/timeseries-evaluation.significance/#bib-Bouckaert2004-replic] |
| Bootstrap intervals from few runs? | yes, pooled over tasks[^/time-series/timeseries-evaluation.significance/#bib-Agarwal2021-precipice] | about twice the stated false alarm rate with five runs; prefer Welch's t-test[^/time-series/timeseries-evaluation.significance/#bib-Colas2018-seeds] |
| Which null hypothesis for forecasting models? | equal accuracy of the models at their true parameters | equal accuracy of the methods as estimated, given the information at the forecast origin[^/time-series/timeseries-evaluation.significance/#bib-Giacomini2003-cpa]; they answer different questions[^/time-series/timeseries-evaluation.significance/#bib-Clark2011-advances] |
| Does reusing a test set overfit it in practice? | it can, badly, in theory | little found in large audits; see [Leakage and Contamination](<https://dl.leima.is/time-series/timeseries-evaluation.leakage/index.md#reuse-the-test-set-wears-out>) |

The deep learning papers of 2017 to 2021 arrived, for random seeds, at what Dietterich and Nadeau and Bengio had established twenty years earlier for data splits: the replicates are fewer and less independent than they look.

## In Practice

1. Decide what the unit of repetition is: runs, time steps or datasets. That decides the tool.
2. Report a measure of spread next to every score: the standard deviation over runs, or a confidence interval.
3. Randomize more than the weight initialization when the budget allows it: data splits first.
4. For two forecasters on one test period, look at the loss differential over time before testing it.
5. For many datasets, compare with ranks, and say how many datasets there were.
6. Count the comparisons that were made, including the ones left out of the table.
7. Report the size of the difference, not only whether it is significant, with a [baseline](<https://dl.leima.is/time-series/timeseries-evaluation.baselines/index.md>) for scale.
8. Before trusting a standard error, ask what the repetitions share: training data, a test period, a seed, a common shock.
9. When the best of several models is reported, say how many were tried, and compare the winner with the best of that many, not with one.
10. Say what difference would matter, and how large a difference the experiment could have detected.

---

**Related:** [Baselines and Skill Scores](<https://dl.leima.is/time-series/timeseries-evaluation.baselines/index.md>) · [Evaluating Probabilistic Forecasts](<https://dl.leima.is/time-series/timeseries-evaluation.probabilistic/index.md>)

[^/time-series/timeseries-evaluation.significance/#bib-Diebold1995-dm]: Diebold F.X., Mariano R.S.. Comparing Predictive Accuracy. 1994. [https://www.nber.org/papers/t0169](<https://www.nber.org/papers/t0169>)

[^/time-series/timeseries-evaluation.significance/#bib-Demsar2006-sc]: Dem\\v sar J.. Statistical Comparisons of Classifiers over Multiple Data Sets. *Journal of Machine Learning Research*. 2006. [https://www.jmlr.org/papers/v7/demsar06a.html](<https://www.jmlr.org/papers/v7/demsar06a.html>)

[^/time-series/timeseries-evaluation.significance/#bib-Giacomini2003-cpa]: Giacomini R., White H.. Tests of conditional predictive ability. 2003.

[^/time-series/timeseries-evaluation.significance/#bib-Clark1999-nested]: Clark T.E., McCracken M.W.. Tests of Equal Forecast Accuracy and Encompassing for Nested Models. 1999.

[^/time-series/timeseries-evaluation.significance/#bib-Dietterich1997-tests]: Dietterich T.G.. Approximate Statistical Tests for Comparing Supervised Classification Learning Algorithms. 1997. [https://sci2s.ugr.es/keel/pdf/algorithm/articulo/dietterich1998.pdf](<https://sci2s.ugr.es/keel/pdf/algorithm/articulo/dietterich1998.pdf>)

[^/time-series/timeseries-evaluation.significance/#bib-Nadeau1999-inference]: Nadeau C., Bengio Y.. Inference for the Generalization Error. *Advances in Neural Information Processing Systems 12*. 1999.

[^/time-series/timeseries-evaluation.significance/#bib-Benavoli2017-bayes]: Benavoli A., Corani G., Dem\\v sar J. et al.. Time for a Change: a Tutorial for Comparing Multiple Classifiers Through Bayesian Analysis. *Journal of Machine Learning Research*. 2017.

[^/time-series/timeseries-evaluation.significance/#bib-Garcia2008-pairwise]: Garc\\'\\ia S., Herrera F.. An Extension on \`\`Statistical Comparisons of Classifiers over Multiple Data Sets'' for all Pairwise Comparisons. *Journal of Machine Learning Research*. 2008.

[^/time-series/timeseries-evaluation.significance/#bib-White2000-rc]: White H.. A Reality Check for Data Snooping. *Econometrica*. 2000.

[^/time-series/timeseries-evaluation.significance/#bib-Hansen2005-spa]: Hansen P.R.. A Test for Superior Predictive Ability. *Journal of Business and Economic Statistics*. 2005. [https://doi.org/10.1198/073500105000000063](<https://doi.org/10.1198/073500105000000063>)

[^/time-series/timeseries-evaluation.significance/#bib-Benavoli2016-ranks]: Benavoli A., Corani G., Mangili F.. Should We Really Use Post-Hoc Tests Based on Mean-Ranks?. *Journal of Machine Learning Research*. 2016.

[^/time-series/timeseries-evaluation.significance/#bib-Hansen2011-mcs]: Hansen P.R., Lunde A., Nason J.M.. The Model Confidence Set. *Econometrica*. 2011.

[^/time-series/timeseries-evaluation.significance/#bib-Benjamini1995-fdr]: Benjamini Y., Hochberg Y.. Controlling the False Discovery Rate: A Practical and Powerful Approach to Multiple Testing. *Journal of the Royal Statistical Society: Series B (Methodological)*. 1995.

[^/time-series/timeseries-evaluation.significance/#bib-Bouthillier2021-var]: Bouthillier X., Delaunay P., Bronzi M. et al.. Accounting for Variance in Machine Learning Benchmarks. 2021. [http://arxiv.org/abs/2103.03098](<http://arxiv.org/abs/2103.03098>)

[^/time-series/timeseries-evaluation.significance/#bib-Reimers2017-seeds]: Reimers N., Gurevych I.. Reporting Score Distributions Makes a Difference: Performance Study of LSTM-networks for Sequence Tagging. 2017. [http://arxiv.org/abs/1707.09861](<http://arxiv.org/abs/1707.09861>)

[^/time-series/timeseries-evaluation.significance/#bib-Picard2021-seed]: Picard D.. torch.manual\_seed(3407) is all you need: On the influence of random seeds in deep learning architectures for computer vision. 2021. [http://arxiv.org/abs/2109.08203](<http://arxiv.org/abs/2109.08203>)

[^/time-series/timeseries-evaluation.significance/#bib-Henderson2017-matters]: Henderson P., Islam R., Bachman P. et al.. Deep Reinforcement Learning that Matters. 2017. [http://arxiv.org/abs/1709.06560](<http://arxiv.org/abs/1709.06560>)

[^/time-series/timeseries-evaluation.significance/#bib-Melis2017-eval]: Melis G., Dyer C., Blunsom P.. On the State of the Art of Evaluation in Neural Language Models. 2017. [http://arxiv.org/abs/1707.05589](<http://arxiv.org/abs/1707.05589>)

[^/time-series/timeseries-evaluation.significance/#bib-Dodge2019-show]: Dodge J., Gururangan S., Card D. et al.. Show Your Work: Improved Reporting of Experimental Results. 2019. [http://arxiv.org/abs/1909.03004](<http://arxiv.org/abs/1909.03004>)

[^/time-series/timeseries-evaluation.significance/#bib-Agarwal2021-precipice]: Agarwal R., Schwarzer M., Castro P.S. et al.. Deep Reinforcement Learning at the Edge of the Statistical Precipice. 2021. [http://arxiv.org/abs/2108.13264](<http://arxiv.org/abs/2108.13264>)

[^/time-series/timeseries-evaluation.significance/#bib-Colas2018-seeds]: Colas C., Sigaud O., Oudeyer P.. How Many Random Seeds? Statistical Power Analysis in Deep Reinforcement Learning Experiments. 2018. [http://arxiv.org/abs/1806.08295](<http://arxiv.org/abs/1806.08295>)

[^/time-series/timeseries-evaluation.significance/#bib-Bouckaert2004-replic]: Bouckaert R.R., Frank E.. Evaluating the Replicability of Significance Tests for Comparing Learning Algorithms. *Proceedings of the 8th Pacific-Asia Conference on Knowledge Discovery and Data Mining (PAKDD 2004)*. 2004.

[^/time-series/timeseries-evaluation.significance/#bib-Bengio2004-novar]: Bengio Y., Grandvalet Y.. No Unbiased Estimator of the Variance of K-Fold Cross-Validation. *Journal of Machine Learning Research*. 2004.

[^/time-series/timeseries-evaluation.significance/#bib-Bates2021-cv]: Bates S., Hastie T., Tibshirani R.. Cross-validation: what does it estimate and how well does it do it?. 2021. [http://arxiv.org/abs/2104.00673](<http://arxiv.org/abs/2104.00673>)

[^/time-series/timeseries-evaluation.significance/#bib-Miller2024-errorbars]: Miller E.. Adding Error Bars to Evals: A Statistical Approach to Language Model Evaluations. 2024. [http://arxiv.org/abs/2411.00640](<http://arxiv.org/abs/2411.00640>)

[^/time-series/timeseries-evaluation.significance/#bib-Hewamalage2022-sc]: Hewamalage H., Ackermann K., Bergmeir C.. Forecast Evaluation for Data Scientists: Common Pitfalls and Best Practices. 2022. [http://arxiv.org/abs/2203.10716](<http://arxiv.org/abs/2203.10716>)

[^/time-series/timeseries-evaluation.significance/#bib-Cerqueira2019-vt]: Cerqueira V., Torgo L., Mozetic I.. Evaluating time series forecasting models: An empirical study on performance estimation methods. 2019. [http://arxiv.org/abs/1905.11744](<http://arxiv.org/abs/1905.11744>)

[^/time-series/timeseries-evaluation.significance/#bib-Diebold2012-dm]: Diebold F.X.. Comparing Predictive Accuracy, Twenty Years Later: A Personal Perspective on the Use and Abuse of Diebold-Mariano Tests. 2012. [http://www.nber.org/papers/w18391](<http://www.nber.org/papers/w18391>)

[^/time-series/timeseries-evaluation.significance/#bib-Clark2011-advances]: Clark T.E., McCracken M.W.. Advances in Forecast Evaluation. 2012.

[^/time-series/timeseries-evaluation.significance/#bib-Demsar2008-appropriate]: Dem\\v sar J.. On the Appropriateness of Statistical Tests in Machine Learning. 2008.

[^/time-series/timeseries-evaluation.significance/#bib-Gelman2006-difference]: Gelman A., Stern H.. The Difference Between `Significant'' and`Not Significant'' is not Itself Statistically Significant. *The American Statistician*. 2006. [https://doi.org/10.1198/000313006X152649](<https://doi.org/10.1198/000313006X152649>)

[^/time-series/timeseries-evaluation.significance/#bib-Gelman2014-typesm]: Gelman A., Carlin J.. Beyond Power Calculations: Assessing Type S (Sign) and Type M (Magnitude) Errors. *Perspectives on Psychological Science*. 2014. [https://doi.org/10.1177/1745691614551642](<https://doi.org/10.1177/1745691614551642>)

[^/time-series/timeseries-evaluation.significance/#bib-Card2020-power]: Card D., Henderson P., Khandelwal U. et al.. With Little Power Comes Great Responsibility. 2020. [http://arxiv.org/abs/2010.06595](<http://arxiv.org/abs/2010.06595>)

[^/time-series/timeseries-evaluation.significance/#bib-Wasserstein2016-asa]: Wasserstein R.L., Lazar N.A.. The ASA's Statement on p-Values: Context, Process, and Purpose. *The American Statistician*. 2016. [https://doi.org/10.1080/00031305.2016.1154108](<https://doi.org/10.1080/00031305.2016.1154108>)

[^/time-series/timeseries-evaluation.significance/#bib-Dolan2002-profiles]: Dolan E.D., Mor\\'e J.J.. Benchmarking Optimization Software with Performance Profiles. 2004. [http://arxiv.org/abs/cs/0102001](<http://arxiv.org/abs/cs/0102001>)

[^/time-series/timeseries-evaluation.significance/#bib-Dror2019-dominance]: Dror R., Shlomov S., Reichart R.. Deep Dominance - How to Properly Compare Deep Neural Models. *Proceedings of the 57th Annual Meeting of the Association for Computational Linguistics*. 2019.

[^/time-series/timeseries-evaluation.significance/#bib-Dror2018-hitchhiker]: Dror R., Baumer G., Shlomov S. et al.. The Hitchhiker's Guide to Testing Statistical Significance in Natural Language Processing. *Proceedings of the 56th Annual Meeting of the Association for Computational Linguistics*. 2018.
