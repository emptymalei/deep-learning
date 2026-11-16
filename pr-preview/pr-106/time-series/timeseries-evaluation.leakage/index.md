# Leakage and Contamination

> [!NOTE]
>
> **In one minute**
>
> An evaluation is only worth something if the model could not have known the answers. The page covers the ways information about the test data reaches a model: features that are not known at forecast time, splits that put the future into the training data, choices made with the test set in view, and pretrained models whose training corpus contains the benchmark.
>
> **Read first:** [Time Series Forecasting Evaluation](<https://dl.leima.is/time-series/timeseries-evaluation.forecasting/index.md>) · [Time Series Forecasting Tasks](<https://dl.leima.is/time-series/timeseries-forecast.tasks/index.md>)

A forecast is made at one point in time, about a later one. An evaluation simulates that situation with historical data, where the later point has already happened and its value sits in the same table as everything else. Nothing in the data stops a model, or the person building it, from using that value. When it happens, the measured accuracy is higher than anything the model will deliver in use. This is leakage.

Kaufman, Rosset and Perlich define leakage as the introduction of information about the target that should not be legitimately available to learn from. They note that it is usually unintentional, enters through the collection and preparation of the data, and tends to be subtle and indirect[^/time-series/timeseries-evaluation.leakage/#bib-Kaufman2011-leak]. Kapoor and Narayanan surveyed the consequences: they found leakage errors documented in 17 scientific fields, affecting 329 papers that use machine learning[^/time-series/timeseries-evaluation.leakage/#bib-Kapoor2022-leak].

```
flowchart LR
    T["What the test<br/>is meant to hide"] -.->|"features known<br/>only later"| M["Model"]
    T -.->|"a split that puts<br/>the future in training"| M
    T -.->|"choices made<br/>with the test set in view"| M
    T -.->|"a pretraining corpus<br/>that contains the benchmark"| M
    M --> S["Score that is<br/>too good"]
```

> [!NOTE]
>
> **How This Page Is Organized**
>
> Each of the four routes in the diagram is treated in the same five steps as the other evaluation pages: **the idea**, **the math**, **the picture**, **try it**, **watch out**.
>
> | Route | Question to ask | In the taxonomy of Kapoor and Narayanan[^/time-series/timeseries-evaluation.leakage/#bib-Kapoor2022-leak] |
> | --- | --- | --- |
> | Features | Was this value known when the forecast was made? | L2, features that are not legitimate |
> | The split | Does any training data come from after the test data? | L3.1, temporal leakage |
> | Choices | Did the test set take part in any decision? | L1, no clean separation of training and test |
> | Pretraining | Has the model seen the benchmark before? | L1, applied to the pretraining corpus |

Scores are errors: lower is better.

## Features: The No-Time-Machine Requirement

**The idea.** Every value in a dataset has two dates: the date it refers to and the date it became known. Only the second one matters for leakage. A model that forecasts Friday's demand on Tuesday may use whatever was known on Tuesday. Thursday's sales refer to a day before the target, and they are still off limits, because on Tuesday they did not exist.

**The math.** Kaufman, Rosset and Perlich call this the no-time-machine requirement: a legitimate model builds only on features with information from a time earlier than that of the target, or, when the problem demands it, a sufficient period earlier[^/time-series/timeseries-evaluation.leakage/#bib-Kaufman2011-leak]. For forecasting, the reference point is the forecast origin. With a forecast made at time \\(t\\) for the target \\(y\_{t+h}\\), and a feature whose value becomes known at time \\(a\\), the condition is

\\\[ a \\le t, \\\]

for every feature and for every training example, each with its own \\(t\\). The horizon \\(h\\) enters through the target: the features for \\(y\_{t+h}\\) may only use what was known \\(h\\) steps before it.

To make this checkable, the authors propose a two-stage method. First, tag every observation with what is needed to decide whether it is legitimate for a given target, in practice the time it became known. Second, observe a learn-predict separation: for each target, only inputs that are legitimate for it become features, and only inputs that are legitimate with respect to all evaluation targets are used as training examples[^/time-series/timeseries-evaluation.leakage/#bib-Kaufman2011-leak].

**The picture.** The widget lists seven candidate features for forecasting the demand of one day. Each has an open circle on the day it refers to and a square on the day it becomes known. Everything with its square left of the forecast origin is legitimate. The features are made up for this illustration.

**Try it.** Press Play to lengthen the horizon from 1 to 10 days and watch features turn from legitimate to leaking. Then change the publication delay of the price index.

An interactive timeline with seven candidate features for a demand forecast. For each feature it shows the day the value refers to and the day it becomes known, and marks the feature as legitimate when it is known on the day the forecast is made. Sliders set the forecast horizon and the publication delay of one feature. It needs JavaScript. In short: the sales of the day before the target are a legitimate feature for a one-day horizon and a leak for longer horizons, and a statistic about a past period is a leak when it is published after the forecast is made.

**Watch out.**

- Lag features are the common case. A model trained with the lag-1 value as a feature and evaluated at horizon 3 has used a value that will not exist when the forecast is needed.
- Revised data leaks too. A table that holds the final, corrected value of a statistic gives the model something the forecaster did not have on the day: on that day only the first release existed.
- Whether a feature is legitimate is a judgement about the problem and needs domain knowledge. Kapoor and Narayanan give no sub-categories for this kind of leakage for that reason, and ask researchers to justify the features they use[^/time-series/timeseries-evaluation.leakage/#bib-Kapoor2022-leak].
- A result that is too good to be true is the classic symptom. Kaufman and coauthors report this as a long-standing rule of thumb in the data mining literature[^/time-series/timeseries-evaluation.leakage/#bib-Kaufman2011-leak].

## The Split: Training on the Future

**The idea.** In most of machine learning the test set is a random sample of the data. For a time series this puts test points between training points. The model is then asked to fill a gap whose both ends it has seen, which is interpolation. A forecast in use has only one end. Since neighbouring values of a time series are similar, interpolation is far easier, and the test error says nothing about forecasting.

**The math.** Kapoor and Narayanan state the rule for models that predict a future outcome: the test set should not contain any data from a date before the training set. Otherwise the model is built with data from the future[^/time-series/timeseries-evaluation.leakage/#bib-Kapoor2022-leak]. With training times \\(\\mathcal T\_{\\text{train}}\\) and test times \\(\\mathcal T\_{\\text{test}}\\),

\\\[ \\max \\mathcal T\_{\\text{train}} \< \\min \\mathcal T\_{\\text{test}}. \\\]

How much a violation matters depends on the autocorrelation. Take a stationary series with variance 1 and lag-1 autocorrelation \\(\\rho\\), such as the AR(1) process \\(x\_t = \\rho\\, x\_{t-1} + \\varepsilon\_t\\). Predicting \\(x\_t\\) from the previous value alone, with the best linear predictor \\(\\rho\\, x\_{t-1}\\), leaves an error variance of

\\\[ 1 - \\rho^2 . \\\]

Predicting it from both neighbours, \\(x\_{t-1}\\) and \\(x\_{t+1}\\), leaves

\\\[ \\frac{1 - \\rho^2}{1 + \\rho^2}. \\\]

Both follow from the standard formula for the best linear predictor of an AR(1) process. For \\(\\rho\\) close to 1, the second is close to half of the first: seeing the next value halves the error variance, with no forecasting skill involved. The split strategies that respect the order of time are covered in [Time Series Forecasting Evaluation](<https://dl.leima.is/time-series/timeseries-evaluation.forecasting/index.md>).

**The picture.** The widget draws an autocorrelated series with a quarter of its points picked at random as test points. Each test point gets two predictions from the same simple kind of model: the average of the nearest training points on both sides, and the previous value alone.

**Try it.** Press Play to repeat the experiment on new series and watch the two averages settle. Then lower the autocorrelation to 0.

An interactive chart of an autocorrelated time series in which a quarter of the points are chosen at random as test points. Each test point is predicted twice: from its neighbours on both sides, and from the previous value only. The mean absolute error of both is shown, for one series and averaged over repeated series. A slider sets the autocorrelation. It needs JavaScript. In short: at an autocorrelation of 0.95 the error with access to both neighbours is clearly smaller than the error when only the past is used, and the gap disappears when the autocorrelation is 0.

**Watch out.**

- Standard k-fold cross validation on a time series has the same problem in every fold but the last. Kapoor and Narayanan list its use on temporal data among the errors found in published studies[^/time-series/timeseries-evaluation.leakage/#bib-Kapoor2022-leak].
- Overlapping windows leak across the boundary. If the last training window and the first test window share time steps, the two are not separated even when the split point is a single date.
- With many series, splitting by series and not by time leaks in another way: the other series were observed during the test period and carry its events. The same paper names non-independence between training and test samples as its own kind of leakage[^/time-series/timeseries-evaluation.leakage/#bib-Kapoor2022-leak].

## Choices: When the Test Set Takes Part

**The idea.** A test set gives an honest estimate once. Each decision that is made after looking at test results, such as which features to keep, how to scale, which architecture, which hyperparameters, when to stop, moves a little information from the test set into the model. No single step looks like cheating. Taken together they fit the model to the test set.

**The math.** Kapoor and Narayanan require a clean separation of training and test data during all preprocessing, modeling and evaluation steps, and list the usual violations[^/time-series/timeseries-evaluation.leakage/#bib-Kapoor2022-leak]:

- preprocessing on the whole dataset, for example imputation or oversampling before the split;
- feature selection on the whole dataset, which uses knowledge of which features do well on the test set;
- duplicates that end up on both sides of the split.

Feature selection shows the size of the effect in a case that can be computed. Take \\(p\\) candidate features that are pure noise, unrelated to the target, and \\(n\\) test samples. The sample correlation \\(r\\) between one such feature and the target is approximately normal with mean 0 and standard deviation \\(1/\\sqrt{n}\\), so \\(\\lvert r \\rvert\\) is around \\(0.8/\\sqrt{n}\\) on average. Choose instead the feature with the largest \\(\\lvert r \\rvert\\) among all \\(p\\): the maximum of \\(p\\) such values grows roughly like

\\\[ \\sqrt{\\frac{2 \\ln p}{n}}, \\\]

which for \\(p = 200\\) and \\(n = 100\\) is about 0.33. A correlation of that size, found by looking at the evaluation data, is produced by noise alone. These are standard approximations for independent normal variables and are meant as orders of magnitude; the widget computes the real thing.

**The picture.** Each dot in the widget is one experiment: many noise features, one target, half of the samples held out for testing. The upper row picks the best-looking feature using all the samples, the lower row using the training half only. Both rows show how well the chosen feature correlates with the target on the test half.

**Try it.** Press Play to accumulate experiments. Then raise the number of features and lower the number of samples.

An interactive chart of repeated experiments with noise features that have no relation to the target. In each experiment the feature with the highest correlation is selected, once using all the data and once using only the training half, and its correlation with the target on the test half is shown. Sliders set the number of samples and the number of features. It needs JavaScript. In short: a feature selected with the test data in view looks predictive on the test data although it is noise, and the effect grows with the number of candidate features and shrinks with the number of samples.

**Watch out.**

- Scaling a series with the mean and standard deviation of its whole history, test period included, is the same error in a milder form. Fit every transformation on the training part.
- The leak can run through the modeler. Kaufman and coauthors point out that choices of features, algorithms and settings made with knowledge of the evaluation data leak in a way an evaluator may not be able to detect, and that in time series prediction, where usually only one history exists, this form is endemic and known as data snooping[^/time-series/timeseries-evaluation.leakage/#bib-Kaufman2011-leak]. Their only prescription is self-control: not to look beyond the separation point.
- A validation set exists for making choices. The test set stays untouched until the choices are final, and each extra look is one of the comparisons counted in [Is the Difference Real?](<https://dl.leima.is/time-series/timeseries-evaluation.significance/index.md#many-comparisons-at-once>).
- To make the separation checkable by others, Kapoor and Narayanan propose model info sheets, in which authors argue for each of three points: the separation of training and test data, the legitimacy of every feature, and that the test set matches the distribution the claim is about[^/time-series/timeseries-evaluation.leakage/#bib-Kapoor2022-leak].

## Pretraining: When the Model Has Seen the Benchmark

**The idea.** Pretrained forecasting models are trained once on a large collection of public time series and then applied to new series without further training, which is called zero-shot forecasting. The claim "zero-shot" is a claim about the past of the model: that it has not seen the series it is tested on. Public benchmarks are built from public datasets, and so are pretraining collections. When they overlap, the model is tested on data it was trained on, and nobody had to make a mistake in their own code for it to happen.

**The math.** Let a fraction \\(f\\) of the benchmark be contained in the pretraining data. If the model's expected error is \\(E\_{\\text{seen}}\\) on those series and \\(E\_{\\text{unseen}}\\) on the rest, the benchmark reports

\\\[ E\_{\\text{benchmark}} = f \\, E\_{\\text{seen}} + (1 - f) \\, E\_{\\text{unseen}}, \\\]

while the quantity of interest for a new user is \\(E\_{\\text{unseen}}\\). The bias is \\(f \\, (E\_{\\text{unseen}} - E\_{\\text{seen}})\\), and it differs from model to model because \\(f\\) depends on each model's corpus. A leaderboard then ranks models partly by how much of the benchmark they were trained on.

Two papers show how this is handled in practice.

- The authors of Chronos split their 55 datasets into three groups before training: 13 used only for training, 15 used for training and evaluation, which they report as in-domain results (Benchmark I), and 27 used only for evaluation, which they report as zero-shot results (Benchmark II). When they compare with other pretrained models on Benchmark II, they mark that the zero-shot setting does not apply to those models, because they were pretrained on some of its datasets[^/time-series/timeseries-evaluation.leakage/#bib-Ansari2024-chronos].
- The authors of the GIFT-Eval benchmark report that the pretraining datasets of three published foundation models partially overlap with its evaluation data. They provide a pretraining collection without overlap, retrain one of the models on it, and compare it with the original. On the datasets affected, they find that the leakage improves the scores in most cases, with a stronger effect at longer prediction lengths, and read this as evidence that such models can memorize training data[^/time-series/timeseries-evaluation.leakage/#bib-Aksu2024-gift].

**The picture.** The widget takes the extreme case to make the mechanism visible: a model that only memorizes. It stores windows of its pretraining series and, for a new context, returns what followed the most similar stored context. On a series it has stored, the forecast is an exact recall, \\(E\_{\\text{seen}} = 0\\). Real pretrained models are not pure memorizers, and the widget is not a model of any of them.

**Try it.** Press Play to move benchmark series into the pretraining data, 5% at a time, and watch the benchmark score fall while the model stays the same. The blue line is a rival model that is better on unseen series.

An interactive chart of a model that memorizes its pretraining series, evaluated on a benchmark of 40 series. A slider sets what share of the benchmark is in the pretraining data. The upper panel shows one benchmark series with the model's forecast; the lower panel shows the benchmark error as a function of the share, next to the error of a rival model that has seen none of the benchmark. It needs JavaScript. In short: the benchmark error falls in proportion to the share of the benchmark that was memorized, and beyond some share the memorizing model is ranked above a rival that is better on unseen series.

**Watch out.**

- Overlap is hard to rule out from the outside. It requires the list of pretraining datasets, and the same series can appear in several collections under different names or at different sampling frequencies.
- A test period held out by time does not help when the corpus contains the complete series, test period included.
- Data published after a model was trained can not be in its corpus. A benchmark built from such data is clean for that model, and stops being clean for the next one.
- Report in-domain and zero-shot results separately, as in the Chronos paper, and say for each compared model which of the two applies[^/time-series/timeseries-evaluation.leakage/#bib-Ansari2024-chronos].

## In Practice

Before trusting a score, including one's own, four questions:

1. For every feature: on which date did this value become known, and is that before the forecast origin of every example it is used for?
2. Is all training data, including the data used for scaling and feature engineering, from before the test period?
3. Which decisions were made after looking at test results, and how many?
4. For a pretrained model: is the list of pretraining datasets known, and does it contain the evaluation data?

And one symptom: a result far better than a strong [baseline](<https://dl.leima.is/time-series/timeseries-evaluation.baselines/index.md>), or than anything published on the same data, is more often a leak than a breakthrough.

---

**Related:** [Baselines and Skill Scores](<https://dl.leima.is/time-series/timeseries-evaluation.baselines/index.md>) · [Is the Difference Real?](<https://dl.leima.is/time-series/timeseries-evaluation.significance/index.md>) · [Evaluating Probabilistic Forecasts](<https://dl.leima.is/time-series/timeseries-evaluation.probabilistic/index.md>)

[^/time-series/timeseries-evaluation.leakage/#bib-Kaufman2011-leak]: Kaufman S., Rosset S., Perlich C.. Leakage in Data Mining: Formulation, Detection, and Avoidance. *Proceedings of the 17th ACM SIGKDD International Conference on Knowledge Discovery and Data Mining (KDD '11)*. 2011.

[^/time-series/timeseries-evaluation.leakage/#bib-Kapoor2022-leak]: Kapoor S., Narayanan A.. Leakage and the Reproducibility Crisis in ML-based Science. 2022. [http://arxiv.org/abs/2207.07048](<http://arxiv.org/abs/2207.07048>)

[^/time-series/timeseries-evaluation.leakage/#bib-Ansari2024-chronos]: Ansari A.F., Stella L., Turkmen C. et al.. Chronos: Learning the Language of Time Series. *Transactions on Machine Learning Research*. 2024. [https://openreview.net/forum?id=gerNCVqqtR](<https://openreview.net/forum?id=gerNCVqqtR>)

[^/time-series/timeseries-evaluation.leakage/#bib-Aksu2024-gift]: Aksu T., Woo G., Liu J. et al.. GIFT-Eval: A Benchmark For General Time Series Forecasting Model Evaluation. 2024. [http://arxiv.org/abs/2410.10393](<http://arxiv.org/abs/2410.10393>)
