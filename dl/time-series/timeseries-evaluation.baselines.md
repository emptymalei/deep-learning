---
summary: >-
  A metric value means little until it is compared with what a method with no
  skill would score. The page covers which simple baselines to use, how MASE
  builds the naive forecast into the metric, and how relative measures and
  skill scores express a model's error as a fraction of a baseline's.
prerequisites:
  - time-series/timeseries-forecast.naive.md
  - time-series/timeseries-metrics.forecasting.md
related:
  - time-series/timeseries-evaluation.forecasting.md
  - time-series/timeseries-metrics.forecasting.crps.md
---

# Baselines and Skill Scores

An MAE of 12 can be excellent or useless. For a series that moves by hundreds from one step to the next it is excellent; for a series that barely moves, repeating the last observation would have done better. A metric value only becomes informative next to the value a simple method gets on the same data. That simple method is the baseline, and this page is about choosing it and comparing against it.


!!! info "How This Page Is Organized"

    Like the other evaluation pages, each of the three concepts is treated in the same five steps: **the idea**, **the math**, **the picture**, **try it**, **watch out**.

    | Concept | What it does with the baseline |
    | --- | --- |
    | Simple baselines | puts its score next to the model's |
    | Scaled errors (MASE) | builds it into the metric |
    | Relative measures and skill scores | divides by it |


## Simple Baselines

**The idea.** A baseline is a forecasting method that needs no modeling effort: repeat the last value, repeat last week, continue the average slope. Whatever accuracy it reaches was available for free, so only the part of a model's accuracy that goes beyond it counts as the model's contribution. Two published comparisons show what is at stake.

Makridakis, Spiliotis and Assimakopoulos compared machine learning methods with eight statistical methods on 1,045 monthly series from the M3 competition, using sMAPE and MASE. The machine learning methods were less accurate on both measures and at every forecasting horizon. The authors also report that Naive 2, a random walk adjusted for seasonality, was more accurate than half of the machine learning methods[@Makridakis2018-slmf].

Zeng and coauthors compared a set of one-layer linear models with Transformer-based models for long-term forecasting on nine datasets. They report that the linear models outperformed the Transformer-based ones in all cases[@Zeng2022-ku].

Neither result says that complex models cannot work. Both say that a result reported without simple baselines can not be interpreted: the reader has no way to tell how much of the accuracy was already available for free.

**The math.** Hyndman and Athanasopoulos describe four simple methods and use them as benchmarks: a new method that is not better than them is, in their words, "not worth considering"[@Hyndman2021-fpp3]. With $T$ observations $y_1, \cdots, y_T$ and a forecast $\hat y_{T+h\vert T}$ for $h$ steps ahead, the four are

| Method | Forecast |
| --- | --- |
| Mean | $\hat y_{T+h\vert T} = \frac{1}{T}\sum_{t=1}^{T} y_t$ |
| Naive | $\hat y_{T+h\vert T} = y_T$ |
| Seasonal naive | $\hat y_{T+h\vert T} = y_{T+h-m(k+1)}$ |
| Drift | $\hat y_{T+h\vert T} = y_T + h\,\frac{y_T - y_1}{T-1}$ |

where $m$ is the seasonal period and $k$ is the integer part of $(h-1)/m$. The seasonal naive forecast repeats the most recent observation from the same point in the season. The naive forecast is also covered in [Naive Forecasts](timeseries-forecast.naive.md).

**The picture.** Each of the four methods carries one kind of structure forward and ignores the rest: the mean keeps the level, the naive forecast the latest value, the seasonal naive forecast the last cycle, and the drift forecast the average slope. The widget draws all four from the same forecast origin on a daily series with a trend and a weekly pattern.

**Try it.** Press Play to move the forecast origin through the series and watch which of the four is ahead. Then set the weekly pattern to 0, or the trend to 0, and see the ranking change.

<div class="dl-widget" data-widget="baseline-forecasts" markdown>

An interactive chart of a daily series with a trend and a weekly pattern, and the mean, naive, seasonal naive and drift forecasts for the next 14 days from a movable forecast origin, with the MAE of each. Sliders set the trend, the strength of the weekly pattern and the origin. It needs JavaScript. In short: with a strong weekly pattern the seasonal naive forecast is the best of the four, and without a pattern one of the other three is.

</div>

**Watch out.**

- The baseline has to match the structure of the data. On a series with a strong weekly pattern, the naive forecast is easy to beat, and beating it shows little. The seasonal naive forecast is the fair comparison there.
- Baselines are evaluated like the model: the same forecast origins, the same horizons, the same test periods.
- Which baseline is strongest changes with the series and the horizon, as the widget shows. Report the strongest one.


## Scaled Errors: A Baseline Inside the Metric

**The idea.** An MAE is in the unit of the data. An MAE of 12 on a series counted in thousands and an MAE of 12 on a series counted in single items can not be averaged, or even compared. Dividing each error by the typical size of a one-step change of the same series removes the unit, and the divisor has a meaning of its own: it is the error the naive forecast made on the training data. MASE therefore reads as "how large are the errors, in units of the naive forecast's error".

**The math.** Hyndman and Koehler define the scaled error for a forecast error $e_t$ as[@Hyndman2006-ld]

$$
q_t = \frac{e_t}{\frac{1}{n-1}\sum_{i=2}^{n} \lvert y_i - y_{i-1} \rvert},
$$

where the denominator is the in-sample MAE of the one-step naive forecast, computed on the $n$ training observations. The mean absolute scaled error is

$$
\operatorname{MASE} = \operatorname{mean}\left( \lvert q_t \rvert \right).
$$

For seasonal series, the scaled error can be defined with the seasonal naive forecast instead[@Hyndman2021-fpp3],

$$
q_j = \frac{e_j}{\frac{1}{T-m}\sum_{t=m+1}^{T} \lvert y_t - y_{t-m} \rvert}.
$$

**The picture.** MASE is a ratio of two averages over two different parts of the series. In the widget, the blue bars are the absolute one-step changes of the training data, and their mean is the scale. The orange bars are the absolute errors of the forecast on the test data, and their mean is the MAE.

**Try it.** Press Play to change the unit of the data, as if going from thousands to single items, and watch the two means and their ratio.

<div class="dl-widget" data-widget="mase-scale" markdown>

An interactive chart of a series with a forecast for its last 12 steps. Below it, bars show the absolute one-step changes of the training data, whose mean is the scale, and the absolute forecast errors, whose mean is the MAE; MASE is their ratio. Sliders set a unit factor and the size of the model's errors. It needs JavaScript. In short: multiplying the data by a factor multiplies the MAE and the scale by the same factor, so MASE does not change.

</div>

**Watch out.** Points from the paper[@Hyndman2006-ld]:

- A MASE below one means that the forecasts have, on average, smaller errors than the one-step naive forecast had on the training data. A MASE above one means larger errors.
- The denominator comes from one-step forecasts. Scaled errors of multi-step forecasts are therefore typically larger than one, and a MASE above one at a long horizon is not, on its own, evidence that the naive forecast would have won at that horizon.
- The scaling does not depend on the unit of the data, and it is only undefined when all historical observations are equal. This is why the authors propose scaled errors for comparing accuracy across series on different scales, where averaging the MAE across series makes no sense.
- Among the scaled measures, the authors prefer MASE: it is less sensitive to outliers and easier to interpret than the root mean squared scaled error, and less variable on small samples than the median absolute scaled error.

The naive and the seasonal naive scaling answer different questions, so a reported MASE should say which one was used.


## Relative Measures and Skill Scores

**The idea.** A second way to bring in the baseline is to compute the same metric for the model and for the baseline, on the same forecasts, and compare the two numbers.

**The math.** A relative measure is their ratio. With $\operatorname{MAE}_b$ as the MAE of the benchmark method, the relative MAE is[@Hyndman2006-ld]

$$
\operatorname{RelMAE} = \frac{\operatorname{MAE}}{\operatorname{MAE}_b}.
$$

A value below one means the method is better than the benchmark.

A skill score expresses the same comparison as an improvement. For a score $S$ where lower is better, and the benchmark's score $S_b$,

$$
\operatorname{Skill} = \frac{S_b - S}{S_b}.
$$

It is the proportion by which the method improves on the benchmark: 0 for no improvement, and negative when the method is worse. Hyndman and Athanasopoulos introduce it with the CRPS and note that it can be used with any accuracy measure[@Hyndman2021-fpp3]. With the MAE as the score, the two definitions above give $\operatorname{Skill} = 1 - \operatorname{RelMAE}$.

!!! note "A Perfect Forecast"

    From the definition, a forecast with zero error has a skill of 1, whatever the benchmark is. A skill of 0.3 reads as "30% lower error than the benchmark".

**The picture.** A skill score is a statement about two forecasts, so it changes when the baseline changes, with the model's forecast untouched. The widget shows a daily series with a weekly pattern, a model that forecasts the next two weeks, and two baselines, with the skill against each.

**Try it.** Press Play to strengthen the weekly pattern step by step. The model's errors stay the same; watch its skill against the naive forecast rise while its skill against the seasonal naive forecast does not. Then change how large the model's errors are.

<div class="dl-widget" data-widget="baseline-skill" markdown>

An interactive chart of a daily series with a weekly pattern, a model's forecast for 14 days, and the naive and seasonal naive forecasts. Sliders set the strength of the weekly pattern and the size of the model's errors; the MAE of all three, the skill against each baseline and MASE with both scalings are shown. It needs JavaScript. In short: with a strong weekly pattern a model can have a high skill against the naive forecast and almost none against the seasonal naive forecast.

</div>

**Watch out.**

- Hyndman and Koehler point out a limit of relative measures: they need several forecasts on the same series to compute the MAE at all, so they can not be used for a single forecast horizon on a single series[@Hyndman2006-ld].
- They also warn against only counting on how many series a method beats the benchmark ("percent better"), because the count says nothing about the size of the differences[@Hyndman2006-ld].
- A skill score is only as informative as its baseline is strong. The same model has a high skill against a weak baseline and a low one against a strong baseline, as the widget shows.
- For probabilistic forecasts, a skill score built from a proper scoring rule is in general not proper itself; see [Evaluating Probabilistic Forecasts](timeseries-evaluation.probabilistic.md#proper-scoring-rules).


## In Practice

The following is a short checklist that follows from the sections above.

1. Report at least the naive forecast, and the seasonal naive forecast when the data has a season, next to every model.
2. Evaluate the baselines on exactly the same test periods and horizons as the model.
3. Report the metric of the baseline or a skill score, not only the metric of the model.
4. When averaging over series with different scales, use a scaled error such as MASE, and say which scaling was used.
5. Compare with the strongest simple baseline, not the weakest.
