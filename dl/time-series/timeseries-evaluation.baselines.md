---
summary: >-
  A reported accuracy number is the result of five choices: what the forecast
  is meant to be, the loss, the scale, how results are averaged, and what they
  are compared with. Drawing on about thirty papers, the page covers each
  choice: simple and tuned baselines, why every error measure rewards a
  different point forecast, scaled errors such as MASE, arithmetic and
  geometric averaging, and skill scores.
prerequisites:
  - time-series/timeseries-forecast.naive.md
  - time-series/timeseries-metrics.forecasting.md
related:
  - time-series/timeseries-evaluation.forecasting.md
  - time-series/timeseries-metrics.forecasting.crps.md
  - time-series/timeseries-evaluation.probabilistic.md
  - time-series/timeseries-evaluation.leakage.md
---

# Baselines and Skill Scores

An MAE of 12 can be excellent or useless. For a series that moves by hundreds from one step to the next it is excellent; for a series that barely moves, repeating the last observation would have done better. A metric value only becomes informative next to the value a simple method gets on the same data. That simple method is the baseline, and this page is about choosing it and comparing against it.


## Five Choices behind Every Benchmark Number

The literature on forecast accuracy is spread over forecasting competitions, statistics, meteorology and, more recently, deep learning benchmarks. Each strand studies one decision. Put together, a number in a results table is the output of five decisions made in sequence.

| Step | The choice | Options in use | Studied by |
| --- | --- | --- | --- |
| 1. Target | which summary of the uncertain future the forecast should be | mean, median, a quantile | Gneiting[@Gneiting2011-point], Kolassa[@Kolassa2023-coherent] |
| 2. Loss | a loss whose best answer is that target | squared, absolute, pinball; percentage errors have odd targets | Gneiting[@Gneiting2011-point], Tofallis[@Tofallis2015-lnq] |
| 3. Scale | what makes different series comparable | the actual value (MAPE), the naive forecast's error (MASE, RMSSE), the benchmark's error (relative errors) | Hyndman and Koehler[@Hyndman2006-ld], Armstrong and Collopy[@Armstrong1992-em], Chen and coauthors[@Chen2017-umbrae] |
| 4. Aggregate | how results are combined over series and datasets | arithmetic mean, geometric mean, median, ranks, win rate | Davydenko and coauthors[@Davydenko2010-wp], Shchur and coauthors[@Shchur2025-fevbench] |
| 5. Reference | what the result is compared with | naive methods, statistical models, tuned linear or tree models, the model with its key part removed | Petropoulos and coauthors[@Petropoulos2022-ftp], Elsayed and coauthors[@Elsayed2021-gbrt], Tan and coauthors[@Tan2024-llm] |

The table is this page's own arrangement of the literature. Steps 3 and 5 are the same idea at two levels: a baseline built into the metric, and a baseline placed next to it.

!!! info "How This Page Is Organized"

    Five concepts, each in the same five steps as the other evaluation pages: **the idea**, **the math**, **the picture**, **try it**, **watch out**. The page starts with the reference, since it is the choice most often skipped.

    | Concept | Step | What it does |
    | --- | --- | --- |
    | Baselines | 5 | puts a simple method's score next to the model's |
    | The target of a metric | 1 and 2 | decides which point forecast counts as best |
    | Scaled errors | 3 | builds the naive forecast into the metric |
    | Averaging across series | 4 | decides which series dominate the result |
    | Skill scores | 3 to 5 | expresses the result as a gain over the reference |


## Baselines

**The idea.** A baseline is a forecasting method that needs little or no modeling effort: repeat the last value, repeat last week, continue the average slope. Whatever accuracy it reaches was available for free, so only the part of a model's accuracy that goes beyond it counts as the model's contribution. The published record shows two things: simple references are beaten less often than one would expect, and a complex model's reported advantage frequently shrinks when the simple side is set up with care.

- Makridakis, Spiliotis and Assimakopoulos compared machine learning methods with eight statistical methods on 1,045 monthly series from the M3 competition. The machine learning methods were less accurate on both measures and at every horizon, and a seasonally adjusted random walk was more accurate than half of them[@Makridakis2018-slmf].
- In the M5 competition on retail sales, 48.4% of the teams beat the naive forecast with their final submission, 35.8% beat the seasonal naive forecast, and 7.5% beat the organizers' exponential smoothing benchmark[@Makridakis2022-hb].
- Zeng and coauthors showed that one linear layer outperformed the Transformer-based long-horizon models of the time on nine datasets[@Zeng2022-ku]. Their paper also includes a model that repeats the last value, which beat all the Transformer models on the exchange rate dataset.
- Qiu and coauthors found a vector autoregression ahead of recent deep models on one dataset and linear regression on another, and note that the papers introducing those models had not compared with either[@Qiu2024-tfb].

**The math.** Hyndman and Athanasopoulos describe four simple methods and use them as benchmarks: a new method that is not better than them is, in their words, "not worth considering"[@Hyndman2021-fpp3]. With $T$ observations $y_1, \cdots, y_T$ and a forecast $\hat y_{T+h\vert T}$ for $h$ steps ahead, the four are

| Method | Forecast |
| --- | --- |
| Mean | $\hat y_{T+h\vert T} = \frac{1}{T}\sum_{t=1}^{T} y_t$ |
| Naive | $\hat y_{T+h\vert T} = y_T$ |
| Seasonal naive | $\hat y_{T+h\vert T} = y_{T+h-m(k+1)}$ |
| Drift | $\hat y_{T+h\vert T} = y_T + h\,\frac{y_T - y_1}{T-1}$ |

where $m$ is the seasonal period and $k$ is the integer part of $(h-1)/m$. The naive forecast is also covered in [Naive Forecasts](timeseries-forecast.naive.md). It is not an arbitrary choice: for a random walk, $y_{t+1} = y_t + \epsilon_t$, the naive forecast is the best forecast there is, and any measured advantage of a complex model on such a series is chance[@Hewamalage2022-sc].

The sources agree that the no-skill methods are a minimum, and they suggest a ladder above them.

| Rung | Baseline | Why it is on the ladder |
| --- | --- | --- |
| 1 | naive, seasonal naive, drift, mean | no parameters; the seasonal naive forecast for seasonal series[@Hewamalage2022-sc] |
| 2 | simple exponential smoothing, Theta | one or two parameters. The Theta method, winner of the M3 competition, was later shown to equal simple exponential smoothing with a drift of half the slope of the fitted trend line[@Hyndman2003-theta] |
| 3 | automatic ETS and ARIMA, and averages of them | the minimum set recommended in a review by some eighty forecasters: naive methods plus ARIMA, exponential smoothing, Holt-Winters and Theta[@Petropoulos2022-ftp] |
| 4 | a linear model on the look-back window | several named linear models are ordinary linear regression in disguise, and the closed-form solution forecast better than the trained versions in 72% of the settings tested[@Toner2024-linear] |
| 5 | gradient-boosted trees on windows | reached or beat eight deep models once set up as a window-based regression[@Elsayed2021-gbrt]; most top entries of M5 were built on it[@Makridakis2022-hb] |
| 6 | the proposed model with its key component removed | removing the language model from three language-model-based forecasters did not hurt accuracy and mostly improved it[@Tan2024-llm] |

The ladder is this page's reading of those sources, not a list from one of them.

For Theta, the result of Hyndman and Billah is a formula worth seeing, because it turns a method described with pages of algebra into a baseline that fits on one line. With $\tilde X_n(h)$ the simple exponential smoothing forecast with parameter $\alpha$ and $\hat b_0$ the slope of a linear trend fitted to the series[@Hyndman2003-theta],

$$
\hat X_n(h) = \tilde X_n(h) + \frac{1}{2}\, \hat b_0 \left( h - 1 + \frac{1}{\alpha} \right).
$$

**The picture.** Each of the four simple methods carries one kind of structure forward and ignores the rest: the mean keeps the level, the naive forecast the latest value, the seasonal naive forecast the last cycle, and the drift forecast the average slope. The widget draws all four from the same forecast origin on a daily series with a trend and a weekly pattern.

**Try it.** Press Play to move the forecast origin through the series and watch which of the four is ahead. Then set the weekly pattern to 0, or the trend to 0, and see the ranking change.

<div class="dl-widget" data-widget="baseline-forecasts" markdown>

An interactive chart of a daily series with a trend and a weekly pattern, and the mean, naive, seasonal naive and drift forecasts for the next 14 days from a movable forecast origin, with the MAE of each. Sliders set the trend, the strength of the weekly pattern and the origin. It needs JavaScript. In short: with a strong weekly pattern the seasonal naive forecast is the best of the four, and without a pattern one of the other three is.

</div>

**Watch out.**

- An untuned baseline is a weaker method under the same name. The tree model that matched the deep models was beaten by far when set up without windows[@Elsayed2021-gbrt]. An automatic ARIMA baseline looked weak in earlier studies because its look-back was limited to five steps; a tuned linear autoregression matched the pretrained time series models tested[@Xu2024-fm]. Tuning one linear baseline on one dataset moved its error by 10% to 15%, as much as the gaps between competing architectures[@Brigato2025-champions].
- Re-run baselines can be far below their published results. Bergmeir traces one deep model through three papers: an error of 0.795 on the M4 data in its own paper, and 0.855 and 0.896 when later papers ran it as a baseline. His advice is to check a claimed win on a competition dataset against the original leaderboard[@Bergmeir2024-llm].
- "Simple beats complex" is narrower than the slogan. The first M-competition found that complex methods were not typically more accurate than simple ones and that combinations of methods did well. Hyndman's reading of M3 is that it did not show simple beating complex: the best two methods were not obviously simple[@Hyndman2020-history]. Deep models did later beat the competition benchmarks, for example N-BEATS on M4 by 11% over the statistical benchmark, as a large ensemble[@Oreshkin2019-nbeats].
- The label "statistical versus machine learning" hides the useful distinction. Januschowski and coauthors propose local versus global: whether parameters are estimated for each series separately or jointly across all series. Neural networks did poorly as local models on short series and better as global ones[@Januschowski2020-criteria].
- Which baseline is strongest changes with the data. Theta, the M3 winner, was the second worst of seven baselines on the first version of the Monash archive, which holds mostly seasonal and high-frequency data[@Godahewa2021-monash]. Report the strongest baseline for the data at hand, evaluated on the same origins, horizons and test periods as the model.


## The Target: Every Metric Rewards a Different Forecast

**The idea.** The future value is uncertain, so behind every point forecast there is a distribution of what might happen. A point forecast is one number chosen to summarize it. Which number is best depends on how errors are punished. Squared errors are smallest on average at the mean of that distribution, absolute errors at its median, and percentage errors somewhere lower still. For a skewed distribution these are different numbers, so the choice of the metric is also the choice of what the model is asked to predict.

**The math.** Gneiting formalizes this. Given a scoring function $S(x, y)$ for a point forecast $x$ and an outcome $y$, the best forecast under a predictive distribution $F$ is the Bayes rule[@Gneiting2011-point]

$$
\hat x = \arg\min_x \; \mathbb E_F \, S(x, Y).
$$

| Scoring function | $S(x, y)$ | Best point forecast |
| --- | --- | --- |
| Squared error | $(x - y)^2$ | the mean of $F$ |
| Absolute error | $\lvert x - y \rvert$ | the median of $F$ |
| Absolute percentage error | $\lvert (x - y) / y \rvert$ | the median of a distribution with density proportional to $f(y)/y$, which lies below the median |
| Pinball loss at level $\alpha$ | $(\mathbb 1\{x \ge y\} - \alpha)(x - y)$ | the $\alpha$-quantile of $F$ |

A scoring function is consistent for a summary such as the mean if that summary always minimizes the expected score, and a summary is elicitable if such a scoring function exists. Gneiting's demand follows: either announce the scoring function in advance, or name the target and score with a function that is consistent for it[@Gneiting2011-point]. MASE and the weighted MAPE are scaled absolute errors, so they reward the median as well[@Kolassa2023-coherent].

Kolassa gives a worked example with a Gamma distribution with shape 2 and scale 1: the best forecast is 2 under squared error, 1.678 under absolute error, and $\log 2 \approx 0.693$ under absolute percentage error[@Kolassa2023-coherent]. Gneiting's own simulation is sharper. For a volatile series he compares a forecaster who always reports the true conditional mean with one who always reports 0.05. Over 100,000 forecasts the constant 0.05 has the lower mean absolute error (0.96 against 0.97) and the lower mean absolute percentage error, and loses clearly under squared error, 7.61 against 5.07[@Gneiting2011-point].

**The picture.** The widget draws one distribution of future demand, a point forecast, and the expected value of three error measures for every possible forecast. Each curve has its lowest point somewhere else.

**Try it.** Press Play to sweep the forecast from high to low and watch the three markers reach their minimum at different moments. Then switch to intermittent demand.

<div class="dl-widget" data-widget="metric-target" markdown>

An interactive chart with a skewed distribution of future demand and a point forecast set by a slider. Three panels show the expected squared error, absolute error and absolute percentage error as functions of the forecast, each with its minimum marked. A switch changes the distribution to intermittent demand with 60% zeros. It needs JavaScript. In short: for a Gamma distribution with shape 2 the best forecast is 2.00 under squared error, 1.68 under absolute error and 0.69 under absolute percentage error; for intermittent demand the forecast with the lowest absolute error is zero.

</div>

**Watch out.**

- Intermittent demand. When more than half of the values are zero, the median is zero, and a measure built on absolute errors can prefer a model that always forecasts zero[@Hewamalage2022-sc]. This is why the M5 competition scored with squared errors: the goal was average sales[@Makridakis2022-hb].
- Selecting models by MAPE favours those that predict too low. Tofallis shows the mechanism and proposes the logarithm of the ratio of prediction to actual, which treats over- and under-prediction alike and rewards the geometric mean[@Tofallis2015-lnq].
- A comparison is unfair when one model was trained with an absolute loss and its competitors with a squared loss, and all are then scored on MAE[@Hewamalage2022-sc].
- Several measures for one point forecast. Reviews of benchmarking practice recommend reporting a diverse set of error measures[@Petropoulos2022-ftp]. Read with the table above in mind, this means a single forecast can not be best on all of them, and a ranking that changes with the measure is the expected result, not a contradiction.
- Only the mean adds up. Forecasts aimed at the mean sum correctly from products to categories; forecasts aimed at the median or at the MAPE target do not[@Kolassa2023-coherent].


## Scaled Errors: A Baseline Inside the Metric

**The idea.** An MAE is in the unit of the data. An MAE of 12 on a series counted in thousands and an MAE of 12 on a series counted in single items can not be averaged, or even compared. Dividing each error by the typical size of a one-step change of the same series removes the unit, and the divisor has a meaning of its own: it is the error the naive forecast made on the training data. MASE therefore reads as "how large are the errors, in units of the naive forecast's error".

**The math.** Three layers of control can be told apart: the MAE has none, a percentage error controls for the scale of the series, and an error relative to a benchmark also controls for how hard the series was to forecast[@Armstrong1992-em]. Armstrong and Collopy's relative absolute error does the last per forecast,

$$
\operatorname{RAE} = \frac{\lvert F_{m,h} - A_h \rvert}{\lvert F_{rw,h} - A_h \rvert},
$$

the error of the method over the error of the random walk at the same horizon. It breaks down when the random walk happens to be exactly right.

Hyndman and Koehler avoid that by scaling with an average. They define the scaled error for a forecast error $e_t$ as[@Hyndman2006-ld]

$$
q_t = \frac{e_t}{\frac{1}{n-1}\sum_{i=2}^{n} \lvert y_i - y_{i-1} \rvert},
$$

where the denominator is the in-sample MAE of the one-step naive forecast, computed on the $n$ training observations. The mean absolute scaled error is

$$
\operatorname{MASE} = \operatorname{mean}\left( \lvert q_t \rvert \right).
$$

The denominator is taken from the training data because it is always available and almost never zero, while an out-of-sample naive error can rest on a single observation[@Hyndman2006-foresight]. For seasonal series, the scaled error can be defined with the seasonal naive forecast instead[@Hyndman2021-fpp3],

$$
q_j = \frac{e_j}{\frac{1}{T-m}\sum_{t=m+1}^{T} \lvert y_t - y_{t-m} \rvert}.
$$

The M5 competition used the squared version, for the reason given in the previous section[@Makridakis2022-hb]:

$$
\operatorname{RMSSE} = \sqrt{ \frac{ \frac{1}{h} \sum_{t=n+1}^{n+h} (y_t - \hat y_t)^2 }{ \frac{1}{n-1} \sum_{t=2}^{n} (y_t - y_{t-1})^2 } } .
$$

**The picture.** MASE is a ratio of two averages over two different parts of the series. In the widget, the blue bars are the absolute one-step changes of the training data, and their mean is the scale. The orange bars are the absolute errors of the forecast on the test data, and their mean is the MAE.

**Try it.** Press Play to change the unit of the data, as if going from thousands to single items, and watch the two means and their ratio.

<div class="dl-widget" data-widget="mase-scale" markdown>

An interactive chart of a series with a forecast for its last 12 steps. Below it, bars show the absolute one-step changes of the training data, whose mean is the scale, and the absolute forecast errors, whose mean is the MAE; MASE is their ratio. Sliders set a unit factor and the size of the model's errors. It needs JavaScript. In short: multiplying the data by a factor multiplies the MAE and the scale by the same factor, so MASE does not change.

</div>

**Watch out.** Points from Hyndman and Koehler[@Hyndman2006-ld]:

- A MASE below one means that the forecasts have, on average, smaller errors than the one-step naive forecast had on the training data. A MASE above one means larger errors.
- The denominator comes from one-step forecasts. Scaled errors of multi-step forecasts are therefore typically larger than one, and a MASE above one at a long horizon is not, on its own, evidence that the naive forecast would have won at that horizon.
- The scaling does not depend on the unit of the data, and it is only undefined when all historical observations are equal. This is why the authors propose scaled errors for comparing accuracy across series on different scales, where averaging the MAE across series makes no sense.
- Among the scaled measures, the authors prefer MASE: it is less sensitive to outliers and easier to interpret than the root mean squared scaled error, and less variable on small samples than the median absolute scaled error.

Later papers add qualifications:

- On intermittent demand MASE is the one measure that stays finite where MAPE and relative errors are infinite or undefined[@Hyndman2006-foresight]. It still rewards the median, which there is often zero[@Kolassa2023-coherent].
- A one-step scale against a long horizon gives values far above one simply because the benchmark was solving an easier problem, and values are not comparable across datasets whose horizons or scalings differ[@Hewamalage2022-sc][@Godahewa2021-monash].
- MASE assumes that the period-to-period changes are stable enough for their average to be a meaningful scale, and it is about as sensitive to a single large error as the MAE[@Chen2017-umbrae].
- The naive and the seasonal naive scaling answer different questions, so a reported MASE should say which one was used.
- Percentage errors, the older way to remove the unit, have their own asymmetries. MAPE punishes forecasts above the actual more than forecasts below, since the error on the low side is bounded at 100%[@Armstrong1992-em]. The symmetric MAPE, introduced to repair this, is not symmetric either: for the same series a 10% under-forecast gets a larger sMAPE than a 10% over-forecast[@Chen2017-umbrae].


## Averaging across Series: A Second Metric Choice

**The idea.** A benchmark has many series, and a paper reports one number. How the per-series results are combined is a decision with consequences as large as the choice of the metric. An arithmetic mean lets a few series with large values decide. Ratios to a benchmark fix the scale but bring a subtler problem: averaged arithmetically, a method that is twice as good on one series and twice as bad on another comes out as worse than the benchmark, and so does the benchmark when the roles are swapped.

**The math.** Let $r_i = \operatorname{MAE}_i / \operatorname{MAE}_i^{b}$ be the relative MAE on series $i$. Davydenko and coauthors point out that with $r_1 = \frac{1}{2}$ and $r_2 = 2$ the arithmetic mean is 1.25 whichever method is called the benchmark, so the penalty for bad forecasting is larger than the reward for good forecasting. Their remedy is the geometric mean, weighted by the number $n_i$ of forecasts in each series[@Davydenko2010-wp],

$$
\operatorname{AvgRelMAE} = \left( \prod_{i=1}^{m} r_i^{\,n_i} \right)^{1 / \sum_i n_i},
$$

for which the two directions give reciprocal results. (The formula is written here as restated by Chen and coauthors[@Chen2017-umbrae]; the working paper read for this page names the measure differently.) Armstrong and Collopy had used the geometric mean of relative absolute errors for the same reason, and noted its weak point: a single zero pulls it to zero, so they clip each ratio to the range from 0.01 to 10[@Armstrong1992-em].

The fev-bench benchmark reports two aggregates side by side for an error matrix $E_{rj}$ over tasks $r$ and models $j$[@Shchur2025-fevbench]. The average win rate counts how often a model beats another,

$$
W_j = \frac{1}{R (M - 1)} \sum_{r=1}^{R} \sum_{k \ne j} \left[ \mathbb 1(E_{rj} < E_{rk}) + \frac{1}{2} \mathbb 1(E_{rj} = E_{rk}) \right],
$$

and the skill score is one minus the geometric mean of clipped errors relative to a baseline $\beta$, the seasonal naive forecast,

$$
S_j = 1 - \exp\left( \frac{1}{R} \sum_{r=1}^{R} \log \operatorname{clip}\left( \frac{E_{rj}}{E_{r\beta}} ;\, 0.01, \, 100 \right) \right).
$$

How much the choice of aggregate matters was measured early. Armstrong and Collopy ranked forecasting methods on subsamples of the M-competition series and asked how well the ranking repeated. The average rank correlation between subsamples was 0.20 for RMSE, 0.50 for MAPE, 0.53 for the geometric mean of relative errors and 0.70 for the share of series on which a method beats the random walk[@Armstrong1992-em]. An earlier re-examination they cite found that 5 of the 1,001 series decided the RMSE ranking of the whole competition.

**The picture.** The widget shows the relative error of a model on eight series, on a logarithmic axis where half and double are the same distance from 1. The solid line is the arithmetic mean, the dashed line the geometric mean.

**Try it.** Press Play to increase the spread between series while the typical ratio stays at 1. Then press "Swap model and benchmark".

<div class="dl-widget" data-widget="aggregate-ratios" markdown>

An interactive chart of the relative errors of a model against a benchmark on eight series, with their arithmetic mean, geometric mean and the number of series on which the model is better. Sliders set the typical ratio and the spread across series, and a button swaps the roles of model and benchmark. It needs JavaScript. In short: when a model is better on half of the series and worse by the same factor on the other half, the geometric mean is 1 in both directions, while the arithmetic mean is above 1 in both directions, so each method looks worse than the other.

</div>

**Watch out.**

- MASE averaged over series is an arithmetic mean of ratios, and so favours the benchmark in the way described above[@Davydenko2010-wp].
- Counts ignore size. Win rates, ranks and "percent better" are robust to outliers, but they say nothing about the size of the differences, and they change when methods are added to the comparison[@Hewamalage2022-sc][@Shchur2025-fevbench]. In fev-bench, one statistical method has a win rate of 33% on the probabilistic metric and a skill score of minus 27%: it is often slightly better than others and occasionally far worse[@Shchur2025-fevbench].
- Geometric means hide outliers as well as resisting them, and a nearly perfect result on one series pulls the whole mean towards zero unless ratios are clipped[@Hewamalage2022-sc].
- Pooled ratios, such as the sum of all absolute errors over the sum of the baseline's, let the largest series dominate[@Garza2023-timegpt]. M5 did this on purpose, weighting each series by its sales in dollars[@Makridakis2022-hb]. A study of that setup found its rankings less stable across comparable datasets than those of simpler measures[@Hewamalage2021-m5].
- A stable ranking is necessary and not sufficient: ranking models alphabetically is perfectly stable[@Hewamalage2021-m5].


## Relative Measures and Skill Scores

**The idea.** A second way to bring in the baseline is to compute the same metric for the model and for the baseline, on the same forecasts, and compare the two numbers. A skill score states the comparison as a share: how much of the baseline's error the model has removed.

**The math.** A relative measure is the ratio of the two. With $\operatorname{MAE}_b$ as the MAE of the benchmark method, the relative MAE is[@Hyndman2006-ld]

$$
\operatorname{RelMAE} = \frac{\operatorname{MAE}}{\operatorname{MAE}_b}.
$$

A value below one means the method is better than the benchmark.

The general skill score comes from meteorology. With $A_f$ the score of the forecasting system, $A_r$ that of a reference and $A_p$ that of a perfect forecast[@Wheatcroft2019-skill],

$$
\operatorname{SS} = \frac{A_f - A_r}{A_p - A_r}.
$$

For errors such as the MAE or the MSE a perfect forecast scores zero, and the expression reduces to the form used by Hyndman and Athanasopoulos, for a score $S$ where lower is better and the benchmark's score $S_b$[@Hyndman2021-fpp3],

$$
\operatorname{Skill} = \frac{S_b - S}{S_b}.
$$

It is the proportion by which the method improves on the benchmark: 0 for no improvement, negative when the method is worse, and 1 for a perfect forecast. With the MAE as the score, $\operatorname{Skill} = 1 - \operatorname{RelMAE}$.

Three uses of the same idea under other names: the M4 competition averaged two metrics, each divided by its value for a seasonally adjusted naive forecast, so that this benchmark scores exactly 1[@Oreshkin2019-nbeats]; fev-bench reports the geometric-mean skill score of the previous section[@Shchur2025-fevbench]; and practitioners track forecast value added, the change in accuracy that each step of a forecasting process contributes over the naive forecast, where a step that adds nothing is waste[@Petropoulos2022-ftp].

**The picture.** A skill score is a statement about two forecasts, so it changes when the baseline changes, with the model's forecast untouched. The first widget shows a daily series with a weekly pattern, a model that forecasts the next two weeks, and two baselines, with the skill against each. The second shows what the skill score does when it is computed from few forecasts.

**Try it.** Press Play to strengthen the weekly pattern step by step. The model's errors stay the same; watch its skill against the naive forecast rise while its skill against the seasonal naive forecast does not.

<div class="dl-widget" data-widget="baseline-skill" markdown>

An interactive chart of a daily series with a weekly pattern, a model's forecast for 14 days, and the naive and seasonal naive forecasts. Sliders set the strength of the weekly pattern and the size of the model's errors; the MAE of all three, the skill against each baseline and MASE with both scalings are shown. It needs JavaScript. In short: with a strong weekly pattern a model can have a high skill against the naive forecast and almost none against the seasonal naive forecast.

</div>

A skill score is a ratio of two noisy averages, and a ratio of averages is biased. Wheatcroft shows that for two forecasters of exactly equal quality, compared on $N$ forecasts with squared errors, the expected skill score is not zero[@Wheatcroft2019-skill]:

$$
\mathbb E \left( \operatorname{MSESS} \right) = 1 - \frac{N}{N - 2} = -\frac{2}{N - 2}.
$$

In the next widget, press Play to lengthen the test set from 5 forecasts to 200.

<div class="dl-widget" data-widget="skill-bias" markdown>

An interactive histogram of the mean squared error skill score of one forecaster against another of exactly the same quality, over 3,000 repeated experiments. A slider sets the number of forecasts in the test set. It needs JavaScript. In short: with 10 forecasts the average skill score is about minus 0.25 although the true skill is zero, and the spread is wide; with 200 forecasts the average is close to zero.

</div>

**Watch out.**

- Hyndman and Koehler point out a limit of relative measures: they need several forecasts on the same series to compute the MAE at all, so they can not be used for a single forecast horizon on a single series[@Hyndman2006-ld].
- They also warn against only counting on how many series a method beats the benchmark, because the count says nothing about the size of the differences[@Hyndman2006-ld].
- A skill score is only as informative as its baseline is strong. In fev-bench the naive forecast has a skill of minus 17% against the seasonal naive forecast, so any skill quoted against the naive forecast starts with that head start[@Shchur2025-fevbench].
- A skill of 1 may be out of reach. When the observations themselves are noisy, no forecast can be perfect, and the score no longer reads as the share of the achievable gain. Wheatcroft's conclusion is to report raw scores in meaningful units unless there is a reason for the skill form[@Wheatcroft2019-skill].
- Relative gains are not value. Halving an error from 40 to 20 counts the same as from 10 to 5, which is why Armstrong and Collopy consider relative errors unsuited to judging the economic benefit of a forecast[@Armstrong1992-em].
- For probabilistic forecasts, a skill score built from a proper scoring rule is in general not proper itself; see [Evaluating Probabilistic Forecasts](timeseries-evaluation.probabilistic.md#proper-scoring-rules).


## How Benchmark Design Changes the Winner

The five choices are usually made by whoever builds the benchmark. Several studies have measured how far the conclusions move when they are made differently.

| Design choice | Finding |
| --- | --- |
| Which datasets and horizons are reported | Over 5,000 random subsets of datasets and horizons, one long-horizon model won about 56% of the time; after removing the single dataset with a clear gap, the models' results overlap. Pooling four related datasets raised one model's win rate from 33% to 71%[@Brigato2025-champions] |
| How much each model is tuned | With equal tuning for eight models on 14 datasets, no model is best everywhere, and a statistical test finds no difference between them[@Brigato2025-champions] |
| Whether errors are computed on normalized data | Errors on normalized series look small: an MAE of 0.43 on one normalized dataset corresponds to a MAPE of 64% on the original scale[@Shao2023-basicts] |
| Whether the last test batch is dropped | The reported error of one model fell from 0.42 to 0.35 as the batch size went from 1 to 512, with nothing else changed[@Qiu2024-tfb] |
| Whose baseline numbers are used | The same baseline on the same dataset is reported with different errors across widely cited papers, and does up to 33% better in a unified pipeline[@Shao2023-basicts] |
| How long the benchmark has been in use | Methods studied on one benchmark for long enough end up fitted to its test data[@Hyndman2020-history]; see [Leakage and Contamination](timeseries-evaluation.leakage.md#reuse-the-test-set-wears-out) |

The remedies proposed in these papers converge: report every dataset and horizon, tune every method, test whether differences are significant, and give uncertainty for the aggregate, for example by resampling the tasks of a benchmark[@Brigato2025-champions][@Shchur2025-fevbench][@Petropoulos2022-ftp]. The tests are the subject of [Is the Difference Real?](timeseries-evaluation.significance.md).


## Where the Literature Disagrees

| Question | One position | The other |
| --- | --- | --- |
| Is MASE the standard measure? | yes: scale free, defined almost everywhere, easy to interpret[@Hyndman2006-ld] | its arithmetic averaging favours the benchmark[@Davydenko2010-wp]; it is as outlier-prone as the MAE[@Chen2017-umbrae]; it rewards the median, which is wrong for intermittent demand[@Kolassa2023-coherent] |
| One measure or several? | several, to guard against a result that depends on one[@Armstrong1992-em][@Petropoulos2022-ftp] | the one measure that is consistent with the forecast's target[@Gneiting2011-point][@Kolassa2023-coherent] |
| Are skill scores good practice? | yes, they are headline numbers at weather centres[@Wheatcroft2019-skill] and in recent forecasting benchmarks[@Shchur2025-fevbench] | report raw scores; the ratio is biased and its upper end may be unreachable[@Wheatcroft2019-skill] |
| Did the competitions show that simple beats complex? | complex methods were not typically more accurate in the early competitions | M3 did not show this[@Hyndman2020-history], and later competitions were won by ensembles of tree models and neural networks[@Makridakis2022-hb][@Oreshkin2019-nbeats] |
| Do pretrained models beat tuned simple models? | a tuned linear autoregression matched the models tested in 2024[@Xu2024-fm] | in a 2025 benchmark of 100 tasks pretrained models lead, with a skill of 35% where the statistical ensemble has 16%[@Shchur2025-fevbench] |

On the last row, the two studies use different tasks and different generations of models, and the author lists of the benchmark and of its top-ranked model family overlap.


## In Practice

1. Say what the forecast is meant to be, the mean, the median or a quantile, and score it with a measure that rewards exactly that.
2. Report at least the naive forecast, and the seasonal naive forecast when the data has a season, next to every model, on exactly the same origins, horizons and test periods.
3. Climb the ladder: add a tuned statistical model, a linear model on the window and, for a new architecture, the model without its new component.
4. Tune the baselines with the same care as the model, and check re-run baselines against their published results.
5. When averaging over series on different scales, use a scaled error, say which scaling was used, and prefer a geometric mean for ratios.
6. Report a count, such as a win rate, and a size, such as a skill score, since they can disagree.
7. Give the raw metric next to any skill score, and the number of forecasts it was computed from.
