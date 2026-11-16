# Leakage and Contamination

> [!NOTE]
>
> **In one minute**
>
> An evaluation is only worth something if the model could not have known the answers. Drawing on more than forty papers, the page sorts the ways information about the test data reaches a model into one table and treats six routes: features known only later, dependent splits, choices fitted outside the loop, reuse of the test set, pretraining corpora that contain the benchmark, and language models that have read about the outcome.
>
> **Read first:** [Time Series Forecasting Evaluation](<https://dl.leima.is/time-series/timeseries-evaluation.forecasting/index.md>) · [Time Series Forecasting Tasks](<https://dl.leima.is/time-series/timeseries-forecast.tasks/index.md>)

A forecast is made at one point in time, about a later one. An evaluation simulates that situation with historical data, where the later point has already happened and its value sits in the same table as everything else. Nothing in the data stops a model, or the person building it, from using that value. When it happens, the measured accuracy is higher than anything the model will deliver in use. This is leakage.

Kaufman, Rosset and Perlich define leakage as the introduction of information about the target that should not be legitimately available to learn from. They note that it is usually unintentional, enters through the collection and preparation of the data, and tends to be subtle and indirect[^/time-series/timeseries-evaluation.leakage/#bib-Kaufman2011-leak]. Hewamalage, Ackermann and Bergmeir give the forecaster's version: the inadvertent use, while training a model, of data from the test set or of any data that is not available at inference time. They add that it happens more easily in forecasting than in classification or regression, and is harder to avoid[^/time-series/timeseries-evaluation.leakage/#bib-Hewamalage2022-sc].

The problem is common. Kapoor and Narayanan found leakage errors documented in 17 scientific fields, affecting 329 papers that use machine learning[^/time-series/timeseries-evaluation.leakage/#bib-Kapoor2022-leak]. Arp and coauthors checked 30 papers from the top security conferences against ten pitfalls and found data snooping at least partly present in 73% of them[^/time-series/timeseries-evaluation.leakage/#bib-Arp2020-dos].

## One Table for All Leaks

The papers behind this page come from statistics, bioinformatics, ecology, economics, computer security, natural language processing and forecasting, and each field has its own vocabulary: selection bias, data snooping, look-ahead bias, adaptive overfitting, contamination. Read side by side, they describe one event. **The reported number was computed on data that was not independent of something that shaped the model.** They differ in what carried the information and in how it reached the model.

| What carries the information | It reaches the model through its training data | It reaches the model through decisions |
| --- | --- | --- |
| The test targets themselves | the test split is in the pretraining corpus[^/time-series/timeseries-evaluation.leakage/#bib-Sainz2023-contam][^/time-series/timeseries-evaluation.leakage/#bib-Shchur2025-fevbench][^/time-series/timeseries-evaluation.leakage/#bib-Meyer2025-leak] | tuning or selecting on the test set[^/time-series/timeseries-evaluation.leakage/#bib-Varma2006-cv][^/time-series/timeseries-evaluation.leakage/#bib-Cawley2010-oms][^/time-series/timeseries-evaluation.leakage/#bib-Arp2020-dos][^/time-series/timeseries-evaluation.leakage/#bib-Recht2019-imagenet] |
| Something statistically tied to the test targets | neighbours in time, space or group[^/time-series/timeseries-evaluation.leakage/#bib-Roberts2017-blockcv][^/time-series/timeseries-evaluation.leakage/#bib-Bergmeir2018-cv]; correlated series over the same period[^/time-series/timeseries-evaluation.leakage/#bib-Meyer2025-leak][^/time-series/timeseries-evaluation.leakage/#bib-Hewamalage2022-sc]; the training split of the same series[^/time-series/timeseries-evaluation.leakage/#bib-Goswami2024-moment] | preprocessing fitted on the whole series: scaling, smoothing, decomposition, feature extraction[^/time-series/timeseries-evaluation.leakage/#bib-Hewamalage2022-sc][^/time-series/timeseries-evaluation.leakage/#bib-Lones2021-pitfalls] |
| Knowledge that did not exist at forecast time | revised data in place of the first release[^/time-series/timeseries-evaluation.leakage/#bib-Croushore1999-rtds]; a language model that has read about the outcome[^/time-series/timeseries-evaluation.leakage/#bib-Sarkar2024-lookahead][^/time-series/timeseries-evaluation.leakage/#bib-LopezLira2025-memo] | benchmark questions or series selected with hindsight[^/time-series/timeseries-evaluation.leakage/#bib-Paleka2025-pitfalls][^/time-series/timeseries-evaluation.leakage/#bib-Sarkar2024-lookahead] |

The table is this page's own arrangement of the literature, not taken from one paper. A second observation comes from putting the papers in order of time. The unit that has to be kept clean has grown with the models:

| Period | The unit to keep clean | Representative work |
| --- | --- | --- |
| around 2000 to 2010 | a row inside a cross-validation fold | selection bias from tuning outside the loop[^/time-series/timeseries-evaluation.leakage/#bib-Varma2006-cv][^/time-series/timeseries-evaluation.leakage/#bib-Cawley2010-oms]; data vintages[^/time-series/timeseries-evaluation.leakage/#bib-Croushore1999-rtds] |
| 2011 to 2012 | a feature | leakage named and formalized[^/time-series/timeseries-evaluation.leakage/#bib-Kaufman2011-leak] |
| 2015 to 2019 | the test set, as a resource that is used up | theory[^/time-series/timeseries-evaluation.leakage/#bib-Dwork2015-holdout][^/time-series/timeseries-evaluation.leakage/#bib-Blum2015-ladder], then audits[^/time-series/timeseries-evaluation.leakage/#bib-Recht2019-imagenet][^/time-series/timeseries-evaluation.leakage/#bib-Roelofs2019-meta] |
| 2017 to 2023 | a block of dependent data | block cross-validation[^/time-series/timeseries-evaluation.leakage/#bib-Roberts2017-blockcv][^/time-series/timeseries-evaluation.leakage/#bib-Bergmeir2018-cv][^/time-series/timeseries-evaluation.leakage/#bib-Cerqueira2019-vt]; field-wide pitfall lists[^/time-series/timeseries-evaluation.leakage/#bib-Lones2021-pitfalls][^/time-series/timeseries-evaluation.leakage/#bib-Arp2020-dos][^/time-series/timeseries-evaluation.leakage/#bib-Kapoor2022-leak][^/time-series/timeseries-evaluation.leakage/#bib-Hewamalage2022-sc] |
| 2021 to 2024 | a dataset, against a pretraining corpus | memorization and contamination of language models[^/time-series/timeseries-evaluation.leakage/#bib-Carlini2020-extract][^/time-series/timeseries-evaluation.leakage/#bib-Magar2022-contam][^/time-series/timeseries-evaluation.leakage/#bib-Sainz2023-contam] |
| since 2023 | a period of calendar time | pretrained time series models[^/time-series/timeseries-evaluation.leakage/#bib-Woo2024-moirai][^/time-series/timeseries-evaluation.leakage/#bib-Das2024-timesfm][^/time-series/timeseries-evaluation.leakage/#bib-Ansari2024-chronos][^/time-series/timeseries-evaluation.leakage/#bib-Shchur2025-fevbench][^/time-series/timeseries-evaluation.leakage/#bib-Meyer2025-leak]; language models used as forecasters[^/time-series/timeseries-evaluation.leakage/#bib-Sarkar2024-lookahead][^/time-series/timeseries-evaluation.leakage/#bib-Paleka2025-pitfalls] |

> [!NOTE]
>
> **How This Page Is Organized**
>
> Six routes, each in the same five steps as the other evaluation pages: **the idea**, **the math**, **the picture**, **try it**, **watch out**.
>
> | Route | Question to ask |
> | --- | --- |
> | Features | Was this value known when the forecast was made? |
> | The split | Is anything in the training data tied to the test data? |
> | Choices | Was everything that was fitted, fitted inside the loop? |
> | Reuse | How many times has this test set been looked at? |
> | Pretraining | Has the model seen the benchmark, or something correlated with it? |
> | Language models | Has the model read about the outcome? |

Scores are errors: lower is better.

## Features: The Knowledge Date

**The idea.** Every value in a dataset has two dates: the date it refers to and the date it became known. Only the second one matters for leakage. A model that forecasts Friday's demand on Tuesday may use whatever was known on Tuesday. Thursday's sales refer to a day before the target, and they are still off limits, because on Tuesday they did not exist.

**The math.** Kaufman, Rosset and Perlich call this the no-time-machine requirement: a legitimate model builds only on features with information from a time earlier than that of the target, or, when the problem demands it, a sufficient period earlier[^/time-series/timeseries-evaluation.leakage/#bib-Kaufman2011-leak]. For forecasting, the reference point is the forecast origin. With a forecast made at time \\(t\\) for the target \\(y\_{t+h}\\), and a feature whose value becomes known at time \\(a\\), the condition is

\\\[ a \\le t, \\\]

for every feature and for every training example, each with its own \\(t\\).

To make this checkable, the authors propose a two-stage method. First, tag every observation with what is needed to decide whether it is legitimate for a given target, in practice the time it became known. Second, observe a learn-predict separation: for each target, only inputs that are legitimate for it become features, and only inputs that are legitimate with respect to all evaluation targets are used as training examples[^/time-series/timeseries-evaluation.leakage/#bib-Kaufman2011-leak].

Economists met the same problem with official statistics, which are revised for years after their first release. Croushore and Stark call the data as it existed on one date a vintage and the collection of vintages a real-time data set, built so that a forecast made at any date uses the data available at that date for all past dates[^/time-series/timeseries-evaluation.leakage/#bib-Croushore1999-rtds]. In that vocabulary a value is indexed twice, \\(y\_{s \\mid v}\\), the value for period \\(s\\) as published in vintage \\(v\\), and a forecast made at \\(t\\) may use \\(y\_{s \\mid v}\\) only for \\(v \\le t\\).

**The picture.** The widget lists seven candidate features for forecasting the demand of one day. Each has an open circle on the day it refers to and a square on the day it becomes known. Everything with its square left of the forecast origin is legitimate. The features are made up for this illustration.

**Try it.** Press Play to lengthen the horizon from 1 to 10 days and watch features turn from legitimate to leaking. Then change the publication delay of the price index.

An interactive timeline with seven candidate features for a demand forecast. For each feature it shows the day the value refers to and the day it becomes known, and marks the feature as legitimate when it is known on the day the forecast is made. Sliders set the forecast horizon and the publication delay of one feature. It needs JavaScript. In short: the sales of the day before the target are a legitimate feature for a one-day horizon and a leak for longer horizons, and a statistic about a past period is a leak when it is published after the forecast is made.

**Watch out.**

- The wrong-horizon leak. Hewamalage and coauthors give the example of a one-day-ahead model that uses summary statistics of the whole current day: it can not run before midnight and is in effect a same-day model[^/time-series/timeseries-evaluation.leakage/#bib-Hewamalage2022-sc].
- Revised data is a leak in principle, and its size has to be measured. Croushore and Stark found that the root-mean-square errors of their forecasts were about the same with real-time and with revised data, while the forecasts themselves could differ considerably. They conclude with a warning about studies that claim a new method beats others when only the latest vintage is shown[^/time-series/timeseries-evaluation.leakage/#bib-Croushore1999-rtds].
- Whether a feature is legitimate is a judgement about the problem and needs domain knowledge. Kapoor and Narayanan give no sub-categories for this kind of leakage for that reason, and ask researchers to justify the features they use[^/time-series/timeseries-evaluation.leakage/#bib-Kapoor2022-leak].
- A result that is too good to be true is the classic symptom. Kaufman and coauthors report this as a long-standing rule of thumb in the data mining literature[^/time-series/timeseries-evaluation.leakage/#bib-Kaufman2011-leak].

## The Split: Dependence between Training and Test

**The idea.** In most of machine learning the test set is a random sample of the data. For a time series this puts test points between training points. The model is then asked to fill a gap whose both ends it has seen, which is interpolation. A forecast in use has only one end. Since neighbouring values of a time series are similar, interpolation is far easier. The general statement is about dependence: a test point tells us about new data only to the extent that it is independent of what the model was trained on.

**The math.** Kapoor and Narayanan state the rule for models that predict a future outcome: the test set should not contain any data from a date before the training set[^/time-series/timeseries-evaluation.leakage/#bib-Kapoor2022-leak]. With training times \\(\\mathcal T\_{\\text{train}}\\) and test times \\(\\mathcal T\_{\\text{test}}\\),

\\\[ \\max \\mathcal T\_{\\text{train}} \< \\min \\mathcal T\_{\\text{test}}. \\\]

How much a violation matters depends on the autocorrelation. Take a stationary series with variance 1 and lag-1 autocorrelation \\(\\rho\\), such as the AR(1) process \\(x\_t = \\rho\\, x\_{t-1} + \\varepsilon\_t\\). Predicting \\(x\_t\\) from the previous value alone, with the best linear predictor, leaves an error variance of \\(1 - \\rho^2\\). Predicting it from both neighbours leaves

\\\[ \\frac{1 - \\rho^2}{1 + \\rho^2}. \\\]

Both follow from the standard formula for the best linear predictor of an AR(1) process. For \\(\\rho\\) close to 1, seeing the next value halves the error variance, with no forecasting skill involved.

Three studies map out when a random split is harmful, and they agree once the goal of the prediction is fixed.

| Study | Setting | Finding |
| --- | --- | --- |
| Roberts and coauthors[^/time-series/timeseries-evaluation.leakage/#bib-Roberts2017-blockcv] | data with temporal, spatial, group or phylogenetic structure | random cross-validation seriously underestimates the error; split in blocks, even when the residuals show no correlation |
| Bergmeir, Hyndman and Koo[^/time-series/timeseries-evaluation.leakage/#bib-Bergmeir2018-cv] | one-step autoregression, one stationary series | ordinary K-fold cross-validation is valid as long as the model's errors are uncorrelated |
| Cerqueira, Torgo and Mozetič[^/time-series/timeseries-evaluation.leakage/#bib-Cerqueira2019-vt] | 62 real series and three synthetic stationary ones | cross-validation works for stationary series; for real ones the most accurate estimates come from out-of-sample methods that keep the time order |

Roberts and coauthors care about transfer to new blocks, new places or new periods. Bergmeir and coauthors care about the next step inside one stationary process, where the rows of lagged values behave like regression data if the model is flexible enough. Real series are rarely stationary, which is where Cerqueira and coauthors land. The split strategies that respect the order of time are covered in [Time Series Forecasting Evaluation](<https://dl.leima.is/time-series/timeseries-evaluation.forecasting/index.md>).

**The picture.** The widget draws an autocorrelated series with a quarter of its points picked at random as test points. Each test point gets two predictions from the same simple kind of model: the average of the nearest training points on both sides, and the previous value alone.

**Try it.** Press Play to repeat the experiment on new series and watch the two averages settle. Then lower the autocorrelation to 0.

An interactive chart of an autocorrelated time series in which a quarter of the points are chosen at random as test points. Each test point is predicted twice: from its neighbours on both sides, and from the previous value only. The mean absolute error of both is shown, for one series and averaged over repeated series. A slider sets the autocorrelation. It needs JavaScript. In short: at an autocorrelation of 0.95 the error with access to both neighbours is clearly smaller than the error when only the past is used, and the gap disappears when the autocorrelation is 0.

**Watch out.**

- Standard cross-validation trains on points later than the test points in all but one of its iterations, which Lones calls look-ahead bias; he adds that whether blocked variants are needed depends on whether the series is stationary[^/time-series/timeseries-evaluation.leakage/#bib-Lones2021-pitfalls]. Kapoor and Narayanan list k-fold cross-validation on temporal data among the errors found in published studies[^/time-series/timeseries-evaluation.leakage/#bib-Kapoor2022-leak]. The result of Bergmeir and coauthors is the exception, and it covers one-step autoregression with uncorrelated errors, not multi-step targets or non-stationary series[^/time-series/timeseries-evaluation.leakage/#bib-Bergmeir2018-cv].
- Blocking has a cost. It can force the model to extrapolate beyond the range of the predictors it saw, which overestimates the error of interpolation; which of the two is wanted depends on the goal[^/time-series/timeseries-evaluation.leakage/#bib-Roberts2017-blockcv].
- Series in one collection leak into each other. If the series of a global model are not aligned in time, one series can contain the future of another, and a shock that hits all of them is already in the training set. Hewamalage and coauthors consider this likely in competition datasets whose series are not aligned[^/time-series/timeseries-evaluation.leakage/#bib-Hewamalage2022-sc].
- Rolling-origin evaluation moves data from the test set into the training set by design, each time the model is retrained, which makes the implementation easy to get wrong and hard to check from outside[^/time-series/timeseries-evaluation.leakage/#bib-Hewamalage2022-sc].

## Choices: Everything Fitted Belongs inside the Loop

**The idea.** A model is more than its weights. The features that were kept, the scaling constants, the hyperparameters, the architecture and the stopping point were all chosen with data. Whatever data took part in those choices has been trained on, even if no gradient ever touched it. If that data includes the test set, the test is no longer a test.

**The math.** Varma and Simon state the condition: cross-validation is only proven unbiased if every part of training, including feature selection, the choice of classifier and the tuning of parameters, takes place inside each loop, on the data that was not left out[^/time-series/timeseries-evaluation.leakage/#bib-Varma2006-cv]. Choosing the hyperparameters that minimize the cross-validation error and then reporting that minimum breaks the condition, because the choosing is itself training. The remedy is to treat "tune, then fit" as the procedure under evaluation and to wrap it in a second loop, called nested cross-validation:

\\\[ \\widehat{\\operatorname{Err}} = \\frac{1}{K} \\sum\_{k=1}^{K} L\\left( \\mathcal D\_k, \\; \\operatorname{fit}\\left( \\mathcal D\_{-k}, \\; \\hat\\lambda(\\mathcal D\_{-k}) \\right) \\right), \\\]

where \\(\\mathcal D\_k\\) is the outer test fold, \\(\\mathcal D\_{-k}\\) the rest, and the hyperparameters \\(\\hat\\lambda\\) are tuned by an inner cross-validation that only sees \\(\\mathcal D\_{-k}\\).

The size of the effect, from three sources:

- Varma and Simon simulated data with no difference between the classes, so that the true error is 50%. The cross-validation error of the tuned classifier was below 30% on 18.5% of the simulated datasets for one classifier and on 38% for another. Nested cross-validation gave estimates close to the true error[^/time-series/timeseries-evaluation.leakage/#bib-Varma2006-cv].
- Cawley and Talbot showed that the bias from tuning outside the evaluation loop is often as large as the differences between learning algorithms. In their experiments, switching one of three classifiers to the biased protocol moved its average rank from about 2 to 1.23 and made it statistically superior, a difference they call entirely spurious[^/time-series/timeseries-evaluation.leakage/#bib-Cawley2010-oms].
- Kapoor and Narayanan list the usual violations: preprocessing on the whole dataset, such as imputation or oversampling before the split, feature selection on the whole dataset, and duplicates on both sides of the split[^/time-series/timeseries-evaluation.leakage/#bib-Kapoor2022-leak].

Feature selection shows the mechanism in a case that can be computed. Take \\(p\\) candidate features that are pure noise and \\(n\\) samples. The sample correlation \\(r\\) between one such feature and the target is approximately normal with mean 0 and standard deviation \\(1/\\sqrt{n}\\). The largest \\(\\lvert r \\rvert\\) among all \\(p\\) grows roughly like

\\\[ \\sqrt{\\frac{2 \\ln p}{n}}, \\\]

which for \\(p = 200\\) and \\(n = 100\\) is about 0.33. These are standard approximations for independent normal variables, meant as orders of magnitude; the widget computes the real thing.

**The picture.** Each dot in the widget is one experiment: many noise features, one target, half of the samples held out for testing. The upper row picks the best-looking feature using all the samples, the lower row using the training half only. Both rows show how well the chosen feature correlates with the target on the test half.

**Try it.** Press Play to accumulate experiments. Then raise the number of features and lower the number of samples.

An interactive chart of repeated experiments with noise features that have no relation to the target. In each experiment the feature with the highest correlation is selected, once using all the data and once using only the training half, and its correlation with the target on the test half is shown. Sliders set the number of samples and the number of features. It needs JavaScript. In short: a feature selected with the test data in view looks predictive on the test data although it is noise, and the effect grows with the number of candidate features and shrinks with the number of samples.

**Watch out.**

- Preprocessing counts. For time series, Hewamalage and coauthors name smoothing, decomposition and normalization applied to the whole series before the split, and features computed over the whole series, which have to be computed from training data only and recomputed as time moves on[^/time-series/timeseries-evaluation.leakage/#bib-Hewamalage2022-sc]. Lones adds data augmentation done before splitting[^/time-series/timeseries-evaluation.leakage/#bib-Lones2021-pitfalls], and Arp and coauthors add cleaning, such as removing outliers with statistics of the complete dataset[^/time-series/timeseries-evaluation.leakage/#bib-Arp2020-dos].
- A biased protocol does not shift every method by the same amount. Methods with more hyperparameters gain more from it, so comparisons made under such a protocol are not consistent with each other[^/time-series/timeseries-evaluation.leakage/#bib-Cawley2010-oms].
- The leak can run through the modeler. Kaufman and coauthors point out that choices of features, algorithms and settings made with knowledge of the evaluation data leak in a way an evaluator may not be able to detect, and that in time series prediction, where usually only one history exists, this form is endemic[^/time-series/timeseries-evaluation.leakage/#bib-Kaufman2011-leak].
- To make the separation checkable by others, Kapoor and Narayanan propose model info sheets, in which authors argue for the separation of training and test data, the legitimacy of every feature, and the match between the test set and the claim[^/time-series/timeseries-evaluation.leakage/#bib-Kapoor2022-leak].

## Reuse: The Test Set Wears Out

**The idea.** Suppose every single experiment is clean: the model is trained without the test set and scored on it once. Now repeat this fifty times, each time keeping the changes that raised the test score. No experiment leaked, and the sequence did, because each score that was looked at carried a little information about the test set into the next decision. A whole research community working on one benchmark for years does the same thing on a larger scale. The papers on this question fall into two groups: theory that shows how bad it can get, and audits that measure how bad it has been.

**The math.** Blum and Hardt model a leaderboard as a sequence of adaptively chosen classifiers \\(f\_1, \\cdots, f\_k\\) and released scores \\(R\_1, \\cdots, R\_k\\), and measure it by how far the best released score is from the truth[^/time-series/timeseries-evaluation.leakage/#bib-Blum2015-ladder],

\\\[ \\operatorname{lberr}(R\_1, \\cdots, R\_k) = \\max\_{1 \\le t \\le k} \\left\\lvert \\min\_{1 \\le i \\le t} R\_D(f\_i) - R\_t \\right\\rvert , \\\]

where \\(R\_D\\) is the true loss. For a leaderboard that simply reports the holdout loss on \\(n\\) examples, they construct a sequence of submissions that drives this error to the order of

\\\[ \\sqrt{\\frac{k}{n}} . \\\]

Their remedy, the Ladder, releases a new score only when a submission improves on the best so far by more than a step \\(\\eta\\), and repeats the previous best otherwise. Its error grows only with the logarithm of the number of submissions[^/time-series/timeseries-evaluation.leakage/#bib-Blum2015-ladder]. Dwork and coauthors give a related mechanism, the reusable holdout, which answers a query from the training set whenever training and holdout agree and reveals a noisy holdout value only when they do not. In their demonstration on data with random labels, an analyst who reuses a standard holdout for variable selection reports an accuracy above 63% where 50% is the truth[^/time-series/timeseries-evaluation.leakage/#bib-Dwork2015-holdout].

Recht and coauthors supply the accounting for what a drop on fresh data means. With the original test set \\(S\\) from distribution \\(D\\) and a new test set \\(S'\\) from \\(D'\\)[^/time-series/timeseries-evaluation.leakage/#bib-Recht2019-imagenet],

\\\[ L\_S - L\_{S'} = \\underbrace{(L\_S - L\_D)}\_{\\text{adaptivity gap}} + \\underbrace{(L\_D - L\_{D'})}\_{\\text{distribution gap}} + \\underbrace{(L\_{D'} - L\_{S'})}\_{\\text{generalization gap}} . \\\]

Only the first term is the wear of the test set. The second is that the new data is different, and the third is sampling noise.

The audits:

| Study | What was compared | Finding |
| --- | --- | --- |
| Recht and coauthors[^/time-series/timeseries-evaluation.leakage/#bib-Recht2019-imagenet] | image classifiers on the original and on newly built test sets | accuracy drops of 3% to 15%, but the ranking of models is kept, and later models gain more on new data, which speaks against adaptive overfitting as the cause |
| Roelofs and coauthors[^/time-series/timeseries-evaluation.leakage/#bib-Roelofs2019-meta] | public leaderboard against private test set in more than one hundred Kaggle competitions | little evidence of substantial overfitting; the outliers have a non-random split or very small test sets |
| Zhang and coauthors[^/time-series/timeseries-evaluation.leakage/#bib-Zhang2024-gsm1k] | language models on a public arithmetic benchmark and a newly written twin | drops of up to 8% for some model families, minimal for many models at the frontier |

**The picture.** The widget plays the attack in its simplest form. The labels of the holdout are coin flips, so no classifier can do better than 50%. An attacker submits random predictions, keeps the ones the leaderboard scored above chance, and combines them by majority vote.

**Try it.** Press Play to add submissions and watch the holdout accuracy of the combination climb while its true accuracy stays at 50%. Shrink the holdout, then switch to the Ladder.

An interactive chart of an attack on a leaderboard whose holdout labels are random. Random submissions are scored on the holdout, those scoring above chance are kept, and their majority vote is scored on the same holdout. Sliders set the holdout size and the number of submissions, and a switch replaces the plain leaderboard with the Ladder of Blum and Hardt. It needs JavaScript. In short: with a plain leaderboard the holdout accuracy of the combination rises far above 50% as submissions accumulate, faster for a smaller holdout, while its accuracy on new data stays at 50%; with the Ladder it stays close to 50%.

**Watch out.**

- The worst case and the typical case are different things. The theory is about what an adversary can do. The audits found that large benchmarks with random splits have held up. Neither result covers the long-horizon forecasting benchmarks, whose test sets are a few thousand overlapping windows of one series and so contain far fewer independent samples than their size suggests. None of the papers read for this page reports an audit of them.
- Lones calls the everyday form sequential overfitting: training models in succession, each guided by the test score of the last. His remedy is the classical one, a validation set for guidance and a holdout touched once[^/time-series/timeseries-evaluation.leakage/#bib-Lones2021-pitfalls].
- Small test sets fail first. In the Kaggle study, the competitions with large gaps between public and private scores had private test sets as small as 100 examples[^/time-series/timeseries-evaluation.leakage/#bib-Roelofs2019-meta].
- Brigato and coauthors retrained eight long-horizon forecasting models with equal tuning effort on 14 datasets and found no model that consistently beats the others. Among their recommendations are test sets that are withheld and scored by a third party[^/time-series/timeseries-evaluation.leakage/#bib-Brigato2025-champions].
- Each look at the test set is also one of the comparisons counted in [Is the Difference Real?](<https://dl.leima.is/time-series/timeseries-evaluation.significance/index.md#many-comparisons-at-once>).

## Pretraining: Three Meanings of Zero-Shot

**The idea.** Pretrained forecasting models are trained once on a large collection of public time series and then applied to new series without further training, which is called zero-shot forecasting. The claim "zero-shot" is a claim about the past of the model: that it has not seen what it is tested on. Public benchmarks are built from public datasets, and so are pretraining collections. When they overlap, the model is tested on data it was trained on, and nobody had to make a mistake in their own code for it to happen. Reading the model papers side by side shows that "has not seen" is used with three different meanings.

**The math.** Let \\(\\mathcal C\\) be the pretraining corpus and let the benchmark consist of series \\(y^{(i)}\\), each split at a date \\(\\tau\_i\\) into a training part and a test part.

| Level | Condition | Who uses it |
| --- | --- | --- |
| 1\. No dataset overlap | no part of any \\(y^{(i)}\\) is in \\(\\mathcal C\\) | the zero-shot benchmarks of Chronos[^/time-series/timeseries-evaluation.leakage/#bib-Ansari2024-chronos], Moirai[^/time-series/timeseries-evaluation.leakage/#bib-Woo2024-moirai], TimesFM[^/time-series/timeseries-evaluation.leakage/#bib-Das2024-timesfm], Lag-Llama[^/time-series/timeseries-evaluation.leakage/#bib-Rasul2023-ep], Time-MoE[^/time-series/timeseries-evaluation.leakage/#bib-Shi2024-timemoe] and Sundial[^/time-series/timeseries-evaluation.leakage/#bib-Liu2025-sundial] |
| 2\. No test-split overlap | only \\(y^{(i)}\_{t \\le \\tau\_i}\\) is in \\(\\mathcal C\\) | MOMENT, which pretrains on the training splits of all its datasets[^/time-series/timeseries-evaluation.leakage/#bib-Goswami2024-moment]; Chronos and Moirai report this case separately as in-domain or in-distribution[^/time-series/timeseries-evaluation.leakage/#bib-Ansari2024-chronos][^/time-series/timeseries-evaluation.leakage/#bib-Woo2024-moirai] |
| 3\. No temporal overlap | nothing in \\(\\mathcal C\\) is later than the start of any test period | proposed by Meyer and coauthors as a global temporal barrier[^/time-series/timeseries-evaluation.leakage/#bib-Meyer2025-leak] |

Level 2 keeps the test values out of the corpus and still lets the model learn the earlier part of the very series it will forecast. The fev-bench authors count both cases as leakage: exposure to the test split inflates accuracy directly, and training on the training split violates the zero-shot setting, since the model has already seen the series it is asked to forecast[^/time-series/timeseries-evaluation.leakage/#bib-Shchur2025-fevbench]. Sainz and coauthors drew the same line for language models two years earlier: contamination with the evaluation split invalidates the experiment, and contamination with the training split invalidates the zero-shot claim[^/time-series/timeseries-evaluation.leakage/#bib-Sainz2023-contam].

Level 3 exists because of indirect leakage. Meyer and coauthors separate direct leakage, where the same samples appear in training and test data, from indirect leakage, where a training series and a test series are correlated and overlap in time. A shock that hits many series at once, such as the start of the pandemic, is then in the training data even though the test series is not[^/time-series/timeseries-evaluation.leakage/#bib-Meyer2025-leak].

For the size of the bias, let a fraction \\(f\\) of the benchmark be affected. If the model's expected error is \\(E\_{\\text{seen}}\\) on those series and \\(E\_{\\text{unseen}}\\) on the rest, the benchmark reports

\\\[ E\_{\\text{benchmark}} = f \\, E\_{\\text{seen}} + (1 - f) \\, E\_{\\text{unseen}}, \\\]

while a new user gets \\(E\_{\\text{unseen}}\\). The bias \\(f \\, (E\_{\\text{unseen}} - E\_{\\text{seen}})\\) differs from model to model because \\(f\\) depends on each model's corpus. Magar and Schwartz measured the two factors separately for a language model and called the difference between seen and unseen test items exploitation. They found that it can be zero even when the model has memorized the data: memorized is not always exploited[^/time-series/timeseries-evaluation.leakage/#bib-Magar2022-contam].

How much overlap is there, and what is done about it?

- Meyer and coauthors traced the data of 22 published time series foundation models. Of the 401 datasets involved, 6% had never appeared in any model's pretraining or fine-tuning data, and no dataset was reserved for evaluation by everyone[^/time-series/timeseries-evaluation.leakage/#bib-Meyer2025-leak].
- The Moirai authors name the root problem: there is no standard held-out split, so it is hard to collect datasets that none of the compared models was trained on[^/time-series/timeseries-evaluation.leakage/#bib-Woo2024-moirai].
- Subtraction: TimesFM, ProbTS and Sundial drop, for each model, the datasets it was pretrained on, which leaves different models evaluated on different data[^/time-series/timeseries-evaluation.leakage/#bib-Das2024-timesfm][^/time-series/timeseries-evaluation.leakage/#bib-Zhang2023-probts][^/time-series/timeseries-evaluation.leakage/#bib-Liu2025-sundial].
- Flagging and retraining: GIFT-Eval reports the overlap for three published models, provides a pretraining collection without overlap, and retrains one model on it. On the affected datasets the leaked version scores better in most cases[^/time-series/timeseries-evaluation.leakage/#bib-Aksu2024-gift].
- Imputation: fev-bench asks contributors to declare overlap, shows it as a column of the leaderboard, and replaces a model's results on leaked tasks with those of a reference model, so that all aggregate scores cover the same tasks[^/time-series/timeseries-evaluation.leakage/#bib-Shchur2025-fevbench].

**The picture.** Two widgets. The first takes the extreme case of direct leakage: a model that only memorizes, stores windows of its pretraining series and returns what followed the most similar stored context. On a stored series the forecast is an exact recall. The second shows indirect leakage with the same kind of model: the target series is never in the corpus, but another series driven by the same factor is. Real pretrained models are not pure memorizers, and neither widget is a model of any of them.

**Try it.** In the first widget, press Play to move benchmark series into the pretraining data and watch the benchmark score fall while the model stays the same. The blue line is a rival model that is better on unseen series; the second slider sets its error as a ratio of the memorizer's error on unseen series.

An interactive chart of a model that memorizes its pretraining series, evaluated on a benchmark of 40 series. A slider sets what share of the benchmark is in the pretraining data. The upper panel shows one benchmark series with the model's forecast; the lower panel shows the benchmark error as a function of the share, next to the error of a rival model that has seen none of the benchmark. It needs JavaScript. In short: the benchmark error falls in proportion to the share of the benchmark that was memorized, and beyond some share the memorizing model is ranked above a rival that is better on unseen series.

In the second widget, switch between a corpus that ends at the split date and one that runs through the test period. Then press Play to raise the share of variance that the two series have in common.

An interactive chart of two series that share a common driver with a shock inside the test period. The target series is never in the pretraining data; the other series is, either up to the split date or through the test period. A memorizing model forecasts the target series by recalling the continuation of the most similar context in its corpus. It needs JavaScript. In short: when the other series is known through the test period and the two series share most of their variance, the forecast of the never-seen target follows the shock and its error is much lower; when they share little, there is no gain.

**Watch out.**

- Overlap can not be detected from the outside today. The fev-bench authors point out that automatic detection amounts to membership inference over pretraining corpora, an open problem for language models and harder for numerical data, and fall back on self-reporting[^/time-series/timeseries-evaluation.leakage/#bib-Shchur2025-fevbench]. Meyer and coauthors argue the opposite way: since disclosure can not be verified, the benchmark and not the model has to guarantee the separation[^/time-series/timeseries-evaluation.leakage/#bib-Meyer2025-leak].
- The same data at another sampling frequency: fev-bench does not count it as leakage[^/time-series/timeseries-evaluation.leakage/#bib-Shchur2025-fevbench], Meyer and coauthors say frequency variants of one process should be treated as dependent[^/time-series/timeseries-evaluation.leakage/#bib-Meyer2025-leak].
- A leak does not always raise the score. In GIFT-Eval the leaked model is better in most cases, not all[^/time-series/timeseries-evaluation.leakage/#bib-Aksu2024-gift], and in the experiment of Meyer and coauthors the model with overlapping data was better in eight of ten runs[^/time-series/timeseries-evaluation.leakage/#bib-Meyer2025-leak].
- The designs that rule leakage out have costs. Withheld test data needs rate-limited submissions and gives no per-series errors, and prospective evaluation, where forecasts are submitted before the truth exists, means waiting for months[^/time-series/timeseries-evaluation.leakage/#bib-Shchur2025-fevbench].
- Report in-domain and zero-shot results separately, as the Chronos and Moirai papers do, and say for each compared model which level of the table above it meets[^/time-series/timeseries-evaluation.leakage/#bib-Ansari2024-chronos][^/time-series/timeseries-evaluation.leakage/#bib-Woo2024-moirai].

## Language Models: A Feature with an Unknown Knowledge Date

**The idea.** When a large language model is used to forecast, or to produce features for a forecaster, the first route of this page returns in a new form. The model is a feature whose knowledge date nobody can state exactly. If its training text includes the period being backtested, it may know the outcome, and a backtest then measures recall. Economists call this look-ahead bias. Sarkar and Vafa define it for language models as the case where the pretraining data contains information about the future, which then leaks into an analysis that should only use information from the past[^/time-series/timeseries-evaluation.leakage/#bib-Sarkar2024-lookahead].

**The math.** Lopez-Lira, Tang and Zhu state the problem as one of identification. When the model has seen the realized values, any observed forecast accuracy is consistent with skill and with memorization, so the forecasting ability on that period is not identified by the backtest at all[^/time-series/timeseries-evaluation.leakage/#bib-LopezLira2025-memo]. The remedy is the level-3 condition of the previous section, applied to the model: the test period has to lie after everything the model was trained on.

Because the corpus is usually not available, a family of tests tries to detect exposure from the model's behaviour.

| Test | Signal of exposure | Source |
| --- | --- | --- |
| Before and after the cutoff | performance is better on items released before the training cutoff than after | Roberts and coauthors[^/time-series/timeseries-evaluation.leakage/#bib-RobertsM2023-time], Li and Flanigan[^/time-series/timeseries-evaluation.leakage/#bib-Li2023-task] |
| Unpredictable events | the model predicts something that could not be known from the stated information | Sarkar and Vafa[^/time-series/timeseries-evaluation.leakage/#bib-Sarkar2024-lookahead] |
| Completion | given the first part of a test item, the model reproduces the rest | Golchin and Surdeanu[^/time-series/timeseries-evaluation.leakage/#bib-Golchin2023-timetravel] |
| Slot guessing | the model fills in a masked wrong answer option of a test question | Deng and coauthors[^/time-series/timeseries-evaluation.leakage/#bib-Deng2023-contam] |
| Canonical order | the model finds the published order of a dataset more likely than shuffled orders | Oren and coauthors[^/time-series/timeseries-evaluation.leakage/#bib-Oren2023-proving] |
| Low-probability tokens | the least likely tokens of a text are less unlikely than for unseen text | Shi and coauthors[^/time-series/timeseries-evaluation.leakage/#bib-ShiW2023-mink] |
| Extraction | training sequences can be drawn out of the model verbatim | Carlini and coauthors[^/time-series/timeseries-evaluation.leakage/#bib-Carlini2020-extract] |

The statistic of Shi and coauthors shows how such a test is built. For a text \\(x\\) with tokens \\(x\_1, \\cdots, x\_N\\), take the set \\(\\text{Min-K\\%}(x)\\) of the \\(k\\)% of tokens to which the model assigns the lowest probability, \\(E\\) tokens in all, and average their log-likelihood[^/time-series/timeseries-evaluation.leakage/#bib-ShiW2023-mink],

\\\[ \\text{Min-K\\% Prob}(x) = \\frac{1}{E} \\sum\_{x\_i \\in \\text{Min-K\\%}(x)} \\log p(x\_i \\mid x\_1, \\cdots, x\_{i-1}) . \\\]

A text the model has not seen tends to contain a few tokens it finds very unlikely; a seen text less so.

Of these tests, the first two carry over to time series without change, because they need only dates. The canonical-order test relies on the examples being exchangeable, which the observations of a time series are not.

**The picture.** The widget shows the before-and-after test from the forecaster's side. Each bar is the error for one month. For the months the model has seen, the answer is partly recalled and the error is low. The backtest uses the months after the stated cutoff.

**Try it.** Press Play to move the true extent of the model's knowledge past its stated cutoff and watch the backtest error fall while the model's real forecasting ability stays the same.

An interactive bar chart of forecast error by month for a model with a knowledge cutoff. Months the model has seen have a low error because the answer is recalled. One slider sets the stated cutoff, after which the backtest period starts, and another sets how far the model's knowledge really extends. It needs JavaScript. In short: when the model's knowledge extends beyond the stated cutoff, part of the backtest period is recalled, and the backtest error is lower than the error the model would have on a period it has truly not seen.

**Watch out.**

- A stated cutoff is not a boundary. Paleka and coauthors note that cutoff dates are published to inform users, mean different things for different developers, and are not guarantees; they show a model that reveals an event from after its stated cutoff[^/time-series/timeseries-evaluation.leakage/#bib-Paleka2025-pitfalls].
- Asking the model to forget does not work. Instructions to ignore later information do not remove the bias[^/time-series/timeseries-evaluation.leakage/#bib-Sarkar2024-lookahead], and masking names and dates fails because the model reconstructs them from context[^/time-series/timeseries-evaluation.leakage/#bib-LopezLira2025-memo].
- The tests are one-sided: a detection shows leakage, and a non-detection does not show its absence[^/time-series/timeseries-evaluation.leakage/#bib-Sarkar2024-lookahead]. Evidence of memorization is a lower bound on what the model has encoded[^/time-series/timeseries-evaluation.leakage/#bib-LopezLira2025-memo].
- The effect is not always large or in one direction. Glasserman and Lin found that, inside the training window, anonymizing company names improved trading signals from news sentiment, so that the model's general knowledge about a company was more of a distraction than a help[^/time-series/timeseries-evaluation.leakage/#bib-Glasserman2023-lookahead].
- Hindsight also enters through the benchmark. Questions selected because they have resolved by now, or written after the fact from news reports, carry their answers in the selection[^/time-series/timeseries-evaluation.leakage/#bib-Paleka2025-pitfalls]. Sarkar and Vafa describe the same for training text that has survived, for example about firms that still exist[^/time-series/timeseries-evaluation.leakage/#bib-Sarkar2024-lookahead].
- Prevention is by design. Proposals include test data published in encrypted form[^/time-series/timeseries-evaluation.leakage/#bib-Jacovi2023-stop], benchmarks that are refreshed monthly from recent sources[^/time-series/timeseries-evaluation.leakage/#bib-White2024-livebench], and newly written twins of public test sets[^/time-series/timeseries-evaluation.leakage/#bib-Zhang2024-gsm1k].

> [!NOTE]
>
> **Not Leakage, but the Test Set Still Moves**
>
> Two findings from benchmark studies belong next to this page. Neither passes test information to the model; both change what the reported number means.
>
> - Many long-horizon forecasting codebases batch the test windows and discard the last incomplete batch. Which windows are scored then depends on the batch size. Qiu and coauthors show the reported error of the same models falling steadily as the batch size grows, and do not drop the last batch in their own benchmark[^/time-series/timeseries-evaluation.leakage/#bib-Qiu2024-tfb].
> - Shao and coauthors collected the numbers reported for the same baselines on the same datasets in several widely cited papers and found that they differ, and traced this to normalization, loss masking and the way metrics are computed. They also note that errors reported on normalized data look small[^/time-series/timeseries-evaluation.leakage/#bib-Shao2023-basicts].

## Where the Literature Disagrees

| Question | One position | The other |
| --- | --- | --- |
| Is K-fold cross-validation valid for time series? | yes, for one-step autoregression with uncorrelated errors[^/time-series/timeseries-evaluation.leakage/#bib-Bergmeir2018-cv] | no in general, block the data[^/time-series/timeseries-evaluation.leakage/#bib-Roberts2017-blockcv]; out-of-sample is best for real series[^/time-series/timeseries-evaluation.leakage/#bib-Cerqueira2019-vt] |
| How harmful is reusing a test set? | provably harmful in the worst case[^/time-series/timeseries-evaluation.leakage/#bib-Blum2015-ladder][^/time-series/timeseries-evaluation.leakage/#bib-Dwork2015-holdout] | little harm found in large audits[^/time-series/timeseries-evaluation.leakage/#bib-Recht2019-imagenet][^/time-series/timeseries-evaluation.leakage/#bib-Roelofs2019-meta] |
| Does contamination raise scores? | clearly, in before-and-after comparisons[^/time-series/timeseries-evaluation.leakage/#bib-RobertsM2023-time][^/time-series/timeseries-evaluation.leakage/#bib-Li2023-task][^/time-series/timeseries-evaluation.leakage/#bib-LopezLira2025-memo] | not always: memorized is not always exploited[^/time-series/timeseries-evaluation.leakage/#bib-Magar2022-contam], and one audit of four models found little evidence of pervasive contamination[^/time-series/timeseries-evaluation.leakage/#bib-Oren2023-proving] |
| Is pretraining on the training split acceptable? | yes, it avoids contamination of the test split[^/time-series/timeseries-evaluation.leakage/#bib-Goswami2024-moment] | it invalidates a zero-shot claim[^/time-series/timeseries-evaluation.leakage/#bib-Shchur2025-fevbench][^/time-series/timeseries-evaluation.leakage/#bib-Sainz2023-contam] |
| Who has to guarantee clean data? | model builders, by disclosure and cooperation[^/time-series/timeseries-evaluation.leakage/#bib-Jacovi2023-stop][^/time-series/timeseries-evaluation.leakage/#bib-Shchur2025-fevbench] | the benchmark, because disclosure can not be verified[^/time-series/timeseries-evaluation.leakage/#bib-Meyer2025-leak] |

The disagreements mostly dissolve once the claim is made precise: which prediction task, which notion of "unseen", worst case or typical case. That is the practical lesson of reading the papers together. A statement about leakage needs its definition next to it.

## In Practice

Before trusting a score, including one's own:

1. For every feature: on which date did this value become known, and is that before the forecast origin of every example it is used for?
2. Is anything in the training data tied to the test data: later in time, the same series, or another series over the same period?
3. Was everything that was fitted, including scaling, feature extraction and hyperparameters, fitted without the test data?
4. How many times has this test set been used to make a decision, by you and by the papers you build on?
5. For a pretrained model: which of the three levels of "unseen" holds for each dataset, and for each model in the comparison?
6. For a language model: does the test period lie after everything the model was trained on?

And one symptom: a result far better than a strong [baseline](<https://dl.leima.is/time-series/timeseries-evaluation.baselines/index.md>), or than anything published on the same data, is more often a leak than a breakthrough.

---

**Related:** [Baselines and Skill Scores](<https://dl.leima.is/time-series/timeseries-evaluation.baselines/index.md>) · [Is the Difference Real?](<https://dl.leima.is/time-series/timeseries-evaluation.significance/index.md>) · [Evaluating Probabilistic Forecasts](<https://dl.leima.is/time-series/timeseries-evaluation.probabilistic/index.md>)

[^/time-series/timeseries-evaluation.leakage/#bib-Kaufman2011-leak]: Kaufman S., Rosset S., Perlich C.. Leakage in Data Mining: Formulation, Detection, and Avoidance. *Proceedings of the 17th ACM SIGKDD International Conference on Knowledge Discovery and Data Mining (KDD '11)*. 2011.

[^/time-series/timeseries-evaluation.leakage/#bib-Hewamalage2022-sc]: Hewamalage H., Ackermann K., Bergmeir C.. Forecast Evaluation for Data Scientists: Common Pitfalls and Best Practices. 2022. [http://arxiv.org/abs/2203.10716](<http://arxiv.org/abs/2203.10716>)

[^/time-series/timeseries-evaluation.leakage/#bib-Kapoor2022-leak]: Kapoor S., Narayanan A.. Leakage and the Reproducibility Crisis in ML-based Science. 2022. [http://arxiv.org/abs/2207.07048](<http://arxiv.org/abs/2207.07048>)

[^/time-series/timeseries-evaluation.leakage/#bib-Arp2020-dos]: Arp D., Quiring E., Pendlebury F. et al.. Dos and Don'ts of Machine Learning in Computer Security. 2020. [http://arxiv.org/abs/2010.09470](<http://arxiv.org/abs/2010.09470>)

[^/time-series/timeseries-evaluation.leakage/#bib-Sainz2023-contam]: Sainz O., Campos J.A., Garc\\'\\ia-Ferrero I. et al.. NLP Evaluation in trouble: On the Need to Measure LLM Data Contamination for each Benchmark. 2023. [http://arxiv.org/abs/2310.18018](<http://arxiv.org/abs/2310.18018>)

[^/time-series/timeseries-evaluation.leakage/#bib-Shchur2025-fevbench]: Shchur O., Ansari A.F., Turkmen C. et al.. fev-bench: A Realistic Benchmark for Time Series Forecasting. 2025. [http://arxiv.org/abs/2509.26468](<http://arxiv.org/abs/2509.26468>)

[^/time-series/timeseries-evaluation.leakage/#bib-Meyer2025-leak]: Meyer M., Kaltenpoth S., Zalipski K. et al.. Rethinking Evaluation in the Era of Time Series Foundation Models: (Un)known Information Leakage Challenges. 2025. [http://arxiv.org/abs/2510.13654](<http://arxiv.org/abs/2510.13654>)

[^/time-series/timeseries-evaluation.leakage/#bib-Varma2006-cv]: Varma S., Simon R.. Bias in error estimation when using cross-validation for model selection. *BMC Bioinformatics*. 2006. [https://doi.org/10.1186/1471-2105-7-91](<https://doi.org/10.1186/1471-2105-7-91>)

[^/time-series/timeseries-evaluation.leakage/#bib-Cawley2010-oms]: Cawley G.C., Talbot N.L.C.. On Over-fitting in Model Selection and Subsequent Selection Bias in Performance Evaluation. *Journal of Machine Learning Research*. 2010. [https://www.jmlr.org/papers/v11/cawley10a.html](<https://www.jmlr.org/papers/v11/cawley10a.html>)

[^/time-series/timeseries-evaluation.leakage/#bib-Recht2019-imagenet]: Recht B., Roelofs R., Schmidt L. et al.. Do ImageNet Classifiers Generalize to ImageNet?. 2019. [http://arxiv.org/abs/1902.10811](<http://arxiv.org/abs/1902.10811>)

[^/time-series/timeseries-evaluation.leakage/#bib-Roberts2017-blockcv]: Roberts D.R., Bahn V., Ciuti S. et al.. Cross-validation strategies for data with temporal, spatial, hierarchical, or phylogenetic structure. *Ecography*. 2017. [https://doi.org/10.1111/ecog.02881](<https://doi.org/10.1111/ecog.02881>)

[^/time-series/timeseries-evaluation.leakage/#bib-Bergmeir2018-cv]: Bergmeir C., Hyndman R.J., Koo B.. A Note on the Validity of Cross-Validation for Evaluating Autoregressive Time Series Prediction. 2018. [https://robjhyndman.com/papers/cv-wp.pdf](<https://robjhyndman.com/papers/cv-wp.pdf>)

[^/time-series/timeseries-evaluation.leakage/#bib-Goswami2024-moment]: Goswami M., Szafer K., Choudhry A. et al.. MOMENT: A Family of Open Time-series Foundation Models. *Proceedings of the 41st International Conference on Machine Learning*. 2024. [http://arxiv.org/abs/2402.03885](<http://arxiv.org/abs/2402.03885>)

[^/time-series/timeseries-evaluation.leakage/#bib-Lones2021-pitfalls]: Lones M.A.. How to avoid machine learning pitfalls: a guide for academic researchers. 2021. [http://arxiv.org/abs/2108.02497](<http://arxiv.org/abs/2108.02497>)

[^/time-series/timeseries-evaluation.leakage/#bib-Croushore1999-rtds]: Croushore D., Stark T.. A Real-Time Data Set for Macroeconomists. 1999. [https://www.philadelphiafed.org/-/media/frbp/assets/working-papers/1999/wp99-4.pdf](<https://www.philadelphiafed.org/-/media/frbp/assets/working-papers/1999/wp99-4.pdf>)

[^/time-series/timeseries-evaluation.leakage/#bib-Sarkar2024-lookahead]: Sarkar S.K., Vafa K.. Lookahead Bias in Pretrained Language Models. 2024. [https://conference.nber.org/conf\_papers/f205530/f205530.pdf](<https://conference.nber.org/conf_papers/f205530/f205530.pdf>)

[^/time-series/timeseries-evaluation.leakage/#bib-LopezLira2025-memo]: Lopez-Lira A., Tang Y., Zhu M.. The Memorization Problem: Can We Trust LLMs' Economic Forecasts?. 2025. [http://arxiv.org/abs/2504.14765](<http://arxiv.org/abs/2504.14765>)

[^/time-series/timeseries-evaluation.leakage/#bib-Paleka2025-pitfalls]: Paleka D., Goel S., Geiping J. et al.. Pitfalls in Evaluating Language Model Forecasters. 2025. [http://arxiv.org/abs/2506.00723](<http://arxiv.org/abs/2506.00723>)

[^/time-series/timeseries-evaluation.leakage/#bib-Dwork2015-holdout]: Dwork C., Feldman V., Hardt M. et al.. Generalization in Adaptive Data Analysis and Holdout Reuse. 2015. [http://arxiv.org/abs/1506.02629](<http://arxiv.org/abs/1506.02629>)

[^/time-series/timeseries-evaluation.leakage/#bib-Blum2015-ladder]: Blum A., Hardt M.. The Ladder: A Reliable Leaderboard for Machine Learning Competitions. 2015. [http://arxiv.org/abs/1502.04585](<http://arxiv.org/abs/1502.04585>)

[^/time-series/timeseries-evaluation.leakage/#bib-Roelofs2019-meta]: Roelofs R., Fridovich-Keil S., Miller J. et al.. A Meta-Analysis of Overfitting in Machine Learning. *33rd Conference on Neural Information Processing Systems (NeurIPS 2019)*. 2019.

[^/time-series/timeseries-evaluation.leakage/#bib-Cerqueira2019-vt]: Cerqueira V., Torgo L., Mozetic I.. Evaluating time series forecasting models: An empirical study on performance estimation methods. 2019. [http://arxiv.org/abs/1905.11744](<http://arxiv.org/abs/1905.11744>)

[^/time-series/timeseries-evaluation.leakage/#bib-Carlini2020-extract]: Carlini N., Tram\`er F., Wallace E. et al.. Extracting Training Data from Large Language Models. 2020. [http://arxiv.org/abs/2012.07805](<http://arxiv.org/abs/2012.07805>)

[^/time-series/timeseries-evaluation.leakage/#bib-Magar2022-contam]: Magar I., Schwartz R.. Data Contamination: From Memorization to Exploitation. 2022. [http://arxiv.org/abs/2203.08242](<http://arxiv.org/abs/2203.08242>)

[^/time-series/timeseries-evaluation.leakage/#bib-Woo2024-moirai]: Woo G., Liu C., Kumar A. et al.. Unified Training of Universal Time Series Forecasting Transformers. *Proceedings of the 41st International Conference on Machine Learning*. 2024. [http://arxiv.org/abs/2402.02592](<http://arxiv.org/abs/2402.02592>)

[^/time-series/timeseries-evaluation.leakage/#bib-Das2024-timesfm]: Das A., Kong W., Sen R. et al.. A decoder-only foundation model for time-series forecasting. 2024. [http://arxiv.org/abs/2310.10688](<http://arxiv.org/abs/2310.10688>)

[^/time-series/timeseries-evaluation.leakage/#bib-Ansari2024-chronos]: Ansari A.F., Stella L., Turkmen C. et al.. Chronos: Learning the Language of Time Series. *Transactions on Machine Learning Research*. 2024. [https://openreview.net/forum?id=gerNCVqqtR](<https://openreview.net/forum?id=gerNCVqqtR>)

[^/time-series/timeseries-evaluation.leakage/#bib-Zhang2024-gsm1k]: Zhang H., Da J., Lee D. et al.. A Careful Examination of Large Language Model Performance on Grade School Arithmetic. 2024. [http://arxiv.org/abs/2405.00332](<http://arxiv.org/abs/2405.00332>)

[^/time-series/timeseries-evaluation.leakage/#bib-Brigato2025-champions]: Brigato L., Morand R., Str\\ommen K. et al.. There are no Champions in Supervised Long-Term Time Series Forecasting. *Transactions on Machine Learning Research*. 2026. [http://arxiv.org/abs/2502.14045](<http://arxiv.org/abs/2502.14045>)

[^/time-series/timeseries-evaluation.leakage/#bib-Rasul2023-ep]: Rasul K., Ashok A., Williams A.R. et al.. Lag-Llama: Towards foundation models for time series forecasting. *arXiv \[cs.LG\]*. 2023. [http://arxiv.org/abs/2310.08278](<http://arxiv.org/abs/2310.08278>)

[^/time-series/timeseries-evaluation.leakage/#bib-Shi2024-timemoe]: Shi X., Wang S., Nie Y. et al.. Time-MoE: Billion-Scale Time Series Foundation Models with Mixture of Experts. *International Conference on Learning Representations (ICLR 2025)*. 2025. [http://arxiv.org/abs/2409.16040](<http://arxiv.org/abs/2409.16040>)

[^/time-series/timeseries-evaluation.leakage/#bib-Liu2025-sundial]: Liu Y., Qin G., Shi Z. et al.. Sundial: A Family of Highly Capable Time Series Foundation Models. *Proceedings of the 42nd International Conference on Machine Learning*. 2025. [http://arxiv.org/abs/2502.00816](<http://arxiv.org/abs/2502.00816>)

[^/time-series/timeseries-evaluation.leakage/#bib-Zhang2023-probts]: Zhang J., Wen X., Zhang Z. et al.. ProbTS: Benchmarking Point and Distributional Forecasting across Diverse Prediction Horizons. 2023. [http://arxiv.org/abs/2310.07446](<http://arxiv.org/abs/2310.07446>)

[^/time-series/timeseries-evaluation.leakage/#bib-Aksu2024-gift]: Aksu T., Woo G., Liu J. et al.. GIFT-Eval: A Benchmark For General Time Series Forecasting Model Evaluation. 2024. [http://arxiv.org/abs/2410.10393](<http://arxiv.org/abs/2410.10393>)

[^/time-series/timeseries-evaluation.leakage/#bib-RobertsM2023-time]: Roberts M., Thakur H., Herlihy C. et al.. Data Contamination Through the Lens of Time. 2023. [http://arxiv.org/abs/2310.10628](<http://arxiv.org/abs/2310.10628>)

[^/time-series/timeseries-evaluation.leakage/#bib-Li2023-task]: Li C., Flanigan J.. Task Contamination: Language Models May Not Be Few-Shot Anymore. 2023. [http://arxiv.org/abs/2312.16337](<http://arxiv.org/abs/2312.16337>)

[^/time-series/timeseries-evaluation.leakage/#bib-Golchin2023-timetravel]: Golchin S., Surdeanu M.. Time Travel in LLMs: Tracing Data Contamination in Large Language Models. *International Conference on Learning Representations (ICLR 2024)*. 2024. [http://arxiv.org/abs/2308.08493](<http://arxiv.org/abs/2308.08493>)

[^/time-series/timeseries-evaluation.leakage/#bib-Deng2023-contam]: Deng C., Zhao Y., Tang X. et al.. Investigating Data Contamination in Modern Benchmarks for Large Language Models. 2023. [http://arxiv.org/abs/2311.09783](<http://arxiv.org/abs/2311.09783>)

[^/time-series/timeseries-evaluation.leakage/#bib-Oren2023-proving]: Oren Y., Meister N., Chatterji N. et al.. Proving Test Set Contamination in Black Box Language Models. 2023. [http://arxiv.org/abs/2310.17623](<http://arxiv.org/abs/2310.17623>)

[^/time-series/timeseries-evaluation.leakage/#bib-ShiW2023-mink]: Shi W., Ajith A., Xia M. et al.. Detecting Pretraining Data from Large Language Models. *International Conference on Learning Representations (ICLR 2024)*. 2024. [http://arxiv.org/abs/2310.16789](<http://arxiv.org/abs/2310.16789>)

[^/time-series/timeseries-evaluation.leakage/#bib-Glasserman2023-lookahead]: Glasserman P., Lin C.. Assessing Look-Ahead Bias in Stock Return Predictions Generated By GPT Sentiment Analysis. 2023. [http://arxiv.org/abs/2309.17322](<http://arxiv.org/abs/2309.17322>)

[^/time-series/timeseries-evaluation.leakage/#bib-Jacovi2023-stop]: Jacovi A., Caciularu A., Goldman O. et al.. Stop Uploading Test Data in Plain Text: Practical Strategies for Mitigating Data Contamination by Evaluation Benchmarks. 2023. [http://arxiv.org/abs/2305.10160](<http://arxiv.org/abs/2305.10160>)

[^/time-series/timeseries-evaluation.leakage/#bib-White2024-livebench]: White C., Dooley S., Roberts M. et al.. LiveBench: A Challenging, Contamination-Limited LLM Benchmark. *International Conference on Learning Representations (ICLR 2025)*. 2025. [http://arxiv.org/abs/2406.19314](<http://arxiv.org/abs/2406.19314>)

[^/time-series/timeseries-evaluation.leakage/#bib-Qiu2024-tfb]: Qiu X., Hu J., Zhou L. et al.. TFB: Towards Comprehensive and Fair Benchmarking of Time Series Forecasting Methods. *Proceedings of the VLDB Endowment*. 2024. [https://doi.org/10.14778/3665844.3665863](<https://doi.org/10.14778/3665844.3665863>)

[^/time-series/timeseries-evaluation.leakage/#bib-Shao2023-basicts]: Shao Z., Wang F., Xu Y. et al.. Exploring Progress in Multivariate Time Series Forecasting: Comprehensive Benchmarking and Heterogeneity Analysis. 2023. [http://arxiv.org/abs/2310.06119](<http://arxiv.org/abs/2310.06119>)
